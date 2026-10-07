"""RAG (Retrieval-Augmented Generation) System using LangChain.

Implements hybrid retrieval combining BM25 sparse retrieval and dense vector search.
Uses HuggingFace embeddings and Chroma vector store.
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


class RAGRetriever:
    """Hybrid RAG system combining sparse (BM25) and dense (vector) retrieval."""

    def __init__(
        self,
        docs_path: str = "dataAnalysis/agronomy_docs",
        chroma_path: str = "data/chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize RAG retriever.

        Args:
            docs_path: Path to directory containing PDF documents
            chroma_path: Path to store Chroma vector database
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.docs_path = Path(docs_path)
        self.chroma_path = Path(chroma_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None

        self._build_or_load()

    def _build_or_load(self) -> None:
        """Build new RAG system or load existing Chroma database."""
        if self.chroma_path.exists() and list(self.chroma_path.glob("**/*")):
            self._load_vectorstore()
        else:
            self._build_new()

    def _build_new(self) -> None:
        """Build RAG system from scratch using PDFs."""
        docs = self._load_documents()

        if not docs:
            raise ValueError(f"No PDF documents found in {self.docs_path}")

        chunks = self._chunk_documents(docs)
        self._create_vectorstore(chunks)
        self._create_bm25_retriever(chunks)
        self._create_ensemble_retriever()

    def _load_documents(self) -> list:
        """Load PDF documents from directory.

        Returns:
            List of Document objects
        """
        loader = PyPDFDirectoryLoader(str(self.docs_path))
        docs = loader.load()
        return docs

    def _chunk_documents(self, docs: list) -> list:
        """Split documents into overlapping chunks.

        Args:
            docs: List of documents

        Returns:
            List of chunked documents
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        return chunks

    def _create_vectorstore(self, chunks: list) -> None:
        """Create Chroma vector store from chunks.

        Args:
            chunks: List of text chunks
        """
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.chroma_path)
        )
        self.vectorstore.persist()

    def _load_vectorstore(self) -> None:
        """Load existing Chroma vector store."""
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_path)
        )

    def _create_bm25_retriever(self, chunks: list) -> None:
        """Create BM25 sparse retriever.

        Args:
            chunks: List of text chunks
        """
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 5

    def _create_ensemble_retriever(self) -> None:
        """Create hybrid ensemble retriever combining BM25 and vector search."""
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve documents for a query.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant documents with content and metadata
        """
        if not self.ensemble_retriever:
            self._build_or_load()

        results = self.ensemble_retriever.invoke(query)

        output = []
        for doc in results[:k]:
            output.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None)
            })

        return output

    def get_context_string(self, query: str, k: int = 5) -> str:
        """Get retrieved documents as a single context string.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            Formatted string of retrieved documents
        """
        docs = self.retrieve(query, k)

        context = "## Retrieved Agricultural Documents\n\n"
        for i, doc in enumerate(docs, 1):
            source = doc["source"].split("/")[-1] if doc["source"] else "unknown"
            context += f"### Document {i}: {source}\n"
            context += f"{doc['content']}\n\n"

        return context

    def search(self, query: str, k: int = 3) -> list[str]:
        """Simple search returning text content only.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of text content strings
        """
        docs = self.retrieve(query, k)
        return [doc["content"] for doc in docs]
