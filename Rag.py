import os
import uuid
import shutil
import tempfile

import numpy as np
import streamlit as st

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq


# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="PDF RAG Chat", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def get_chroma_client(persist_directory: str):
    return chromadb.PersistentClient(path=persist_directory)


# ----------------------------------------------------------------------------
# Core classes (adapted from the notebook)
# ----------------------------------------------------------------------------
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = load_embedding_model(model_name)

    def generate_embeddings(self, texts):
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_directory="chroma_db_streamlit"):
        self.client = get_chroma_client(persist_directory)
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents, embeddings, batch_size=500):
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have equal lengths.")
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start:start + batch_size]
            batch_embs = embeddings[start:start + batch_size]
            ids, texts, metas, embs = [], [], [], []
            for offset, (doc, emb) in enumerate(zip(batch_docs, batch_embs)):
                ids.append(f"chunk_{start + offset}_{uuid.uuid4().hex[:8]}")
                texts.append(doc.page_content)
                metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif value is not None:
                        metadata[key] = str(value)
                metas.append(metadata or {"source_file": "unknown"})
                embs.append(emb.tolist())
            self.collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)


class RAGRetriever:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query, top_k=5, score_threshold=0.0):
        if not query.strip():
            return []
        count = self.vector_store.collection.count()
        if count == 0:
            return []
        query_embedding = self.embedder.generate_embeddings([query])[0].tolist()
        result = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        found = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        for rank, (doc_id, text, metadata, distance) in enumerate(zip(ids, docs, metas, distances), 1):
            score = 1.0 - float(distance)
            if score >= score_threshold:
                found.append({
                    "id": doc_id,
                    "content": text,
                    "metadata": metadata or {},
                    "similarity_score": score,
                    "distance": float(distance),
                    "rank": rank,
                })
        return found


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    if not documents:
        raise ValueError("The documents list is empty. Load PDFs first.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def rag_advanced(query, retriever, llm, top_k=5, min_score=0.0, return_context=False):
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        output = {"answer": "No relevant context found in the indexed PDFs.", "sources": [], "confidence": 0.0}
        if return_context:
            output["context"] = ""
        return output

    context = "\n\n".join(
        f"[Source: {r['metadata'].get('source_file', 'unknown')}, page: {r['metadata'].get('page', 'unknown')}]\n"
        f"{r['content']}"
        for r in results
    )

    sources = [
        {
            "source": r["metadata"].get("source_file", "unknown"),
            "page": r["metadata"].get("page", "unknown"),
            "similarity": round(r["similarity_score"], 4),
        }
        for r in results
    ]

    if llm is None:
        answer = "Relevant passages were retrieved, but no LLM is configured to generate an answer."
    else:
        prompt = f"""You are a helpful assistant. Answer using only the supplied document context.
If the context does not contain the answer, clearly say that you cannot determine it from these documents.

Context:
{context}

Question: {query}

Answer:"""
        response = llm.invoke(prompt)
        answer = response.content

    confidence = max(r["similarity_score"] for r in results)

    output = {"answer": answer, "sources": sources, "confidence": confidence}
    if return_context:
        output["context"] = context
    return output


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "work_dir" not in st.session_state:
    st.session_state.work_dir = tempfile.mkdtemp(prefix="rag_streamlit_")


# ----------------------------------------------------------------------------
# Sidebar - configuration
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 PDF RAG Chat")
    st.caption("Upload PDFs, build an index, then ask questions grounded in your documents.")

    st.subheader("1. Groq API key")
    default_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input("Groq API key", value=default_key, type="password",
                                  help="Get a free key at console.groq.com. Retrieval works without it; answers need it.")

    st.subheader("2. Upload PDFs")
    uploaded_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)

    st.subheader("3. Index settings")
    chunk_size = st.slider("Chunk size", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=500, value=200, step=50)

    build_clicked = st.button("🔨 Build / Rebuild Index", use_container_width=True, type="primary")

    st.subheader("4. Retrieval settings")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)
    min_score = st.slider("Minimum similarity score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    show_context = st.checkbox("Show retrieved context with each answer", value=False)

    if st.session_state.indexed_files:
        st.success(f"Indexed: {', '.join(st.session_state.indexed_files)}\n\n{st.session_state.chunk_count} chunks")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------------------------------
# Build index
# ----------------------------------------------------------------------------
def build_index(files, chunk_size, chunk_overlap):
    upload_dir = Path(st.session_state.work_dir) / "uploads"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for f in files:
        path = upload_dir / f.name
        path.write_bytes(f.getbuffer())
        saved_paths.append(path)

    all_docs = []
    progress = st.progress(0.0, text="Loading PDFs...")
    for i, path in enumerate(saved_paths, 1):
        try:
            pages = PyPDFLoader(str(path)).load()
            for page in pages:
                page.metadata["source_file"] = path.name
            all_docs.extend(pages)
        except Exception as exc:
            st.warning(f"Could not load {path.name}: {exc}")
        progress.progress(i / len(saved_paths), text=f"Loaded {path.name}")

    if not all_docs:
        st.error("No PDF pages were loaded. Check your files and try again.")
        progress.empty()
        return

    progress.progress(0.6, text="Splitting into chunks...")
    chunks = split_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    progress.progress(0.75, text="Generating embeddings...")
    embedder = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.generate_embeddings(texts)

    progress.progress(0.9, text="Storing in vector database...")
    persist_dir = str(Path(st.session_state.work_dir) / "chroma_db")
    vectorstore = VectorStore(persist_directory=persist_dir)
    vectorstore.add_documents(chunks, embeddings)

    st.session_state.retriever = RAGRetriever(vectorstore, embedder)
    st.session_state.indexed_files = [p.name for p in saved_paths]
    st.session_state.chunk_count = len(chunks)

    progress.progress(1.0, text="Done!")
    progress.empty()
    st.success(f"Indexed {len(saved_paths)} file(s) into {len(chunks)} chunks.")


if build_clicked:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF first.")
    else:
        with st.spinner("Building index..."):
            build_index(uploaded_files, chunk_size, chunk_overlap)

# (Re)configure the LLM whenever a key is present
if groq_api_key:
    try:
        st.session_state.llm = ChatGroq(
            api_key=groq_api_key,
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_tokens=1024,
        )
    except Exception as exc:
        st.sidebar.error(f"Could not configure Groq: {exc}")
        st.session_state.llm = None
else:
    st.session_state.llm = None


# ----------------------------------------------------------------------------
# Main chat area
# ----------------------------------------------------------------------------
st.title("Chat with your PDFs")

if not st.session_state.retriever:
    st.info("👈 Upload one or more PDFs and click **Build / Rebuild Index** to get started.")
elif not groq_api_key:
    st.warning("Retrieval is ready, but no Groq API key is set — answers will be disabled until you add one.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources (confidence: {msg.get('confidence', 0):.2f})"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** (page {s['page']}) — similarity {s['similarity']}")
            if msg.get("context"):
                with st.expander("🧩 Retrieved context"):
                    st.text(msg["context"])

question = st.chat_input("Ask a question about your documents...")

if question:
    if not st.session_state.retriever:
        st.error("Please build the index first (upload PDFs and click 'Build / Rebuild Index').")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = rag_advanced(
                    question,
                    st.session_state.retriever,
                    st.session_state.llm,
                    top_k=top_k,
                    min_score=min_score,
                    return_context=show_context,
                )
            st.markdown(result["answer"])
            if result.get("sources"):
                with st.expander(f"📚 Sources (confidence: {result.get('confidence', 0):.2f})"):
                    for s in result["sources"]:
                        st.markdown(f"- **{s['source']}** (page {s['page']}) — similarity {s['similarity']}")
            if show_context and result.get("context"):
                with st.expander("🧩 Retrieved context"):
                    st.text(result["context"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "context": result.get("context", "") if show_context else "",
        })
