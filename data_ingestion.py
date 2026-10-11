"""Data Ingestion Pipeline.

Loads and processes:
1. PDF documents into ChromaDB vector store
2. CSV tabular data into SQLite database

Run: python data_ingestion.py
"""
import sqlite3
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class DataIngestionPipeline:
    """Handles data ingestion for PDFs and tabular data."""

    def __init__(
        self,
        pdf_dir: str = "data/agronomy_docs",
        csv_dir: str = "data/raw",
        db_path: str = "data/farmer_profile.db",
        chroma_path: str = "data/chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize data ingestion pipeline.

        Args:
            pdf_dir: Directory containing PDF documents
            csv_dir: Directory containing CSV files
            db_path: Path to SQLite database
            chroma_path: Path to Chroma vector store
            chunk_size: Text chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.pdf_dir = Path(pdf_dir)
        self.csv_dir = Path(csv_dir)
        self.db_path = Path(db_path)
        self.chroma_path = Path(chroma_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.parent.mkdir(parents=True, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

    def ingest_pdfs(self) -> None:
        """Load PDFs and store in ChromaDB."""
        logger.info(f"Loading PDFs from {self.pdf_dir}")

        if not self.pdf_dir.exists():
            logger.warning(f"PDF directory not found: {self.pdf_dir}")
            return

        pdf_files = list(self.pdf_dir.glob("**/*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return

        logger.info(f"Found {len(pdf_files)} PDF files")

        try:
            loader = PyPDFDirectoryLoader(str(self.pdf_dir))
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} document pages")

            if not documents:
                logger.warning("No documents loaded from PDFs")
                return

            logger.info("Splitting documents into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} text chunks")

            logger.info("Creating Chroma vector store...")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(self.chroma_path)
            )
            if hasattr(vectorstore, "persist"):
                vectorstore.persist()
            logger.info(f"Stored {len(chunks)} chunks in ChromaDB at {self.chroma_path}")

        except Exception as e:
            logger.error(f"Error ingesting PDFs: {e}", exc_info=True)
            raise

    def ingest_tabular_data(self) -> None:
        """Load CSV files and store in SQLite."""
        logger.info(f"Loading tabular data from {self.csv_dir}")

        if not self.csv_dir.exists():
            logger.warning(f"CSV directory not found: {self.csv_dir}")
            return

        csv_files = list(self.csv_dir.glob("**/*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {self.csv_dir}")
            return

        logger.info(f"Found {len(csv_files)} CSV files")

        try:
            self._init_sqlite_db()

            for csv_file in csv_files:
                logger.info(f"Processing {csv_file.name}")
                self._load_csv_to_sqlite(csv_file)

            logger.info("Tabular data ingestion complete")

        except Exception as e:
            logger.error(f"Error ingesting tabular data: {e}", exc_info=True)
            raise

    def _init_sqlite_db(self) -> None:
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crop_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season TEXT NOT NULL,
                crop_planted TEXT NOT NULL,
                yield_achieved REAL,
                nitrogen_used REAL,
                profit REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        logger.info(f"Initialized SQLite database at {self.db_path}")
        conn.close()

    def _load_csv_to_sqlite(self, csv_file: Path) -> None:
        """Load CSV file into SQLite.

        Args:
            csv_file: Path to CSV file
        """
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} rows from {csv_file.name}")

            table_name = csv_file.stem.lower()

            conn = sqlite3.connect(self.db_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()

            logger.info(f"Stored {len(df)} rows in table '{table_name}'")

        except Exception as e:
            logger.error(f"Error loading {csv_file.name}: {e}")
            raise

    def verify_ingestion(self) -> dict:
        """Verify successful data ingestion.

        Returns:
            Dict with ingestion statistics
        """
        stats = {
            "chroma_db_exists": self.chroma_path.exists(),
            "sqlite_db_exists": self.db_path.exists(),
            "pdf_documents": 0,
            "csv_files": 0,
            "sqlite_tables": []
        }

        if self.pdf_dir.exists():
            stats["pdf_documents"] = len(list(self.pdf_dir.glob("**/*.pdf")))

        if self.csv_dir.exists():
            stats["csv_files"] = len(list(self.csv_dir.glob("**/*.csv")))

        if self.db_path.exists():
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                stats["sqlite_tables"] = [row[0] for row in cursor.fetchall()]
                conn.close()
            except Exception as e:
                logger.error(f"Error verifying SQLite: {e}")

        return stats

    def run(self) -> dict:
        """Run complete data ingestion pipeline.

        Returns:
            Dict with ingestion statistics
        """
        logger.info("Starting data ingestion pipeline")

        try:
            self.ingest_pdfs()
            self.ingest_tabular_data()

            stats = self.verify_ingestion()

            logger.info("="*60)
            logger.info("DATA INGESTION SUMMARY")
            logger.info("="*60)
            logger.info(f"ChromaDB created: {stats['chroma_db_exists']}")
            logger.info(f"SQLite created: {stats['sqlite_db_exists']}")
            logger.info(f"PDF documents processed: {stats['pdf_documents']}")
            logger.info(f"CSV files processed: {stats['csv_files']}")
            logger.info(f"SQLite tables: {', '.join(stats['sqlite_tables'])}")
            logger.info("="*60)

            return stats

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    pipeline = DataIngestionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
