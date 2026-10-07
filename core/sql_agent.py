"""SQL Agent for Farm History Queries.

Uses LangChain to translate natural language to SQL queries.
Manages SQLite database for crop history and farm operations.
"""
import sqlite3
from pathlib import Path
from typing import Optional, Any
from langchain.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain.prompts.chat import ChatPromptTemplate


class FarmHistoryAgent:
    """Manages farm history database and natural language SQL queries."""

    def __init__(self, db_path: str = "data/farmer_profile.db"):
        """Initialize the farm history agent.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.sql_db = SQLDatabase.from_uri(f"sqlite:///{self.db_path.absolute()}")

    def _init_database(self) -> None:
        """Initialize SQLite database with crop_history table."""
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
        conn.close()

    def add_farm_record(
        self,
        season: str,
        crop_planted: str,
        yield_achieved: Optional[float] = None,
        nitrogen_used: Optional[float] = None,
        profit: Optional[float] = None
    ) -> dict[str, Any]:
        """Add a new farm record to history.

        Args:
            season: Growing season (e.g., 'Kharif', 'Rabi')
            crop_planted: Name of crop planted
            yield_achieved: Yield in tons/hectare
            nitrogen_used: Nitrogen used in kg/hectare
            profit: Profit in currency units

        Returns:
            Dict with record id and status
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO crop_history
            (season, crop_planted, yield_achieved, nitrogen_used, profit)
            VALUES (?, ?, ?, ?, ?)
        """, (season, crop_planted, yield_achieved, nitrogen_used, profit))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return {"id": record_id, "status": "success"}

    def ask_farm_history(self, natural_query: str) -> dict[str, Any]:
        """Translate natural language query to SQL and execute.

        Args:
            natural_query: Natural language question about farm history

        Returns:
            Dict with query results and natural language summary
        """
        try:
            chain = create_sql_query_chain(self.llm, self.sql_db)

            system_prompt = """Given an input question, first create a syntactically correct SQLite query.
Always use table named 'crop_history' with columns: id, season, crop_planted, yield_achieved, nitrogen_used, profit.
Only select columns that are relevant to the question.
Limit results to 10 rows.
Do not make any DML statements (INSERT, UPDATE, DELETE)."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            chain_with_prompt = prompt | chain
            sql_query = chain_with_prompt.invoke({"input": natural_query})

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            conn.close()

            results = [dict(row) for row in rows]

            summary_prompt = f"""Based on these database results: {results}

Answer the original question: {natural_query}

Provide a clear, concise natural language summary."""

            summary_response = self.llm.invoke(summary_prompt)

            return {
                "query": sql_query,
                "results": results,
                "summary": summary_response.content,
                "row_count": len(results)
            }

        except Exception as e:
            return {
                "error": str(e),
                "query": None,
                "results": [],
                "summary": f"Could not process query: {str(e)}"
            }

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics of farm history.

        Returns:
            Dict with aggregate statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT crop_planted) as unique_crops,
                COUNT(DISTINCT season) as unique_seasons,
                ROUND(AVG(yield_achieved), 2) as avg_yield,
                ROUND(AVG(profit), 2) as avg_profit,
                ROUND(AVG(nitrogen_used), 2) as avg_nitrogen
            FROM crop_history
        """)

        row = cursor.fetchone()
        conn.close()

        return {
            "total_records": row[0],
            "unique_crops": row[1],
            "unique_seasons": row[2],
            "avg_yield": row[3],
            "avg_profit": row[4],
            "avg_nitrogen": row[5]
        }
