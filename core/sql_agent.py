import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain.chains import create_sql_query_chain
except ImportError:
    from langchain_classic.chains import create_sql_query_chain

from langchain_core.prompts import ChatPromptTemplate


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

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            try:
                self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
            except Exception:
                self.llm = None
        else:
            self.llm = None

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

    def _sanitize_sql(self, sql_str: str) -> str:
        """Strip markdown formatting and explanations from LLM SQL output."""
        cleaned = sql_str.strip()
        # Remove markdown code fences
        cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        # Remove SQLQuery: prefix
        if "SQLQuery:" in cleaned:
            cleaned = cleaned.split("SQLQuery:")[-1]
        cleaned = cleaned.strip()
        # Take only the first SQL statement if multiple lines
        lines = [line.strip() for line in cleaned.splitlines() if line.strip() and not line.strip().startswith("--")]
        cleaned = " ".join(lines)
        if ";" in cleaned:
            cleaned = cleaned.split(";")[0] + ";"
        return cleaned

    def ask_farm_history(self, natural_query: str) -> dict[str, Any]:
        """Translate natural language query to SQL and execute.

        Args:
            natural_query: Natural language question about farm history

        Returns:
            Dict with query results and natural language summary
        """
        # Try LLM-based SQL generation first
        if self.llm is not None:
            try:
                chain = create_sql_query_chain(self.llm, self.sql_db)

                system_prompt = """Given an input question, create a syntactically correct SQLite query.
Always use table named 'crop_history' with columns: id, season, crop_planted, yield_achieved, nitrogen_used, profit.
Only select columns that are relevant to the question.
Limit results to 10 rows.
Do not make any DML statements (INSERT, UPDATE, DELETE). Output only valid SQL."""

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])

                chain_with_prompt = prompt | chain
                raw_sql = chain_with_prompt.invoke({"input": natural_query})
                sql_query = self._sanitize_sql(raw_sql)

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
                summary_text = getattr(summary_response, "content", str(summary_response))

                return {
                    "query": sql_query,
                    "results": results,
                    "summary": summary_text,
                    "row_count": len(results)
                }

            except Exception as e:
                # Fall back to heuristic rule-based query execution
                pass

        # Robust Heuristic SQL fallback
        return self._heuristic_query(natural_query)

    def _heuristic_query(self, natural_query: str) -> dict[str, Any]:
        """Execute heuristic SQL queries based on keywords in user question."""
        q = natural_query.lower()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if "highest yield" in q or "max yield" in q or "top yield" in q or "best yield" in q:
            sql = "SELECT crop_planted, yield_achieved, season FROM crop_history ORDER BY yield_achieved DESC LIMIT 5"
            cursor.execute(sql)
            rows = cursor.fetchall()
            results = [dict(r) for r in rows]
            top = results[0] if results else None
            summary = f"The highest yield recorded was {top['yield_achieved']} t/ha with {top['crop_planted']} in {top['season']} season." if top else "No yield records found."

        elif "highest profit" in q or "max profit" in q or "most profit" in q or "best profit" in q:
            sql = "SELECT crop_planted, profit, season FROM crop_history ORDER BY profit DESC LIMIT 5"
            cursor.execute(sql)
            rows = cursor.fetchall()
            results = [dict(r) for r in rows]
            top = results[0] if results else None
            summary = f"The most profitable crop was {top['crop_planted']} in {top['season']} season generating ₹{top['profit']:,.0f}." if top else "No profit records found."

        elif "kharif" in q:
            sql = "SELECT crop_planted, yield_achieved, profit, nitrogen_used FROM crop_history WHERE LOWER(season) LIKE '%kharif%' LIMIT 10"
            cursor.execute(sql)
            rows = cursor.fetchall()
            results = [dict(r) for r in rows]
            summary = f"Found {len(results)} records for Kharif season."

        elif "rabi" in q:
            sql = "SELECT crop_planted, yield_achieved, profit, nitrogen_used FROM crop_history WHERE LOWER(season) LIKE '%rabi%' LIMIT 10"
            cursor.execute(sql)
            rows = cursor.fetchall()
            results = [dict(r) for r in rows]
            summary = f"Found {len(results)} records for Rabi season."

        elif "average" in q or "avg" in q:
            sql = "SELECT ROUND(AVG(yield_achieved), 2) as avg_yield, ROUND(AVG(profit), 2) as avg_profit, ROUND(AVG(nitrogen_used), 2) as avg_nitrogen FROM crop_history"
            cursor.execute(sql)
            rows = cursor.fetchall()
            results = [dict(r) for r in rows]
            row = results[0] if results else {}
            summary = f"Average Yield: {row.get('avg_yield', 0)} t/ha, Average Profit: ₹{row.get('avg_profit', 0):,.0f}, Average Nitrogen: {row.get('avg_nitrogen', 0)} kg/ha."

        else:
            sql = "SELECT id, season, crop_planted, yield_achieved, profit, nitrogen_used FROM crop_history ORDER BY id DESC LIMIT 10"
            cursor.execute(sql)
            rows = cursor.fetchall()
            results = [dict(r) for r in rows]
            summary = f"Found {len(results)} farm history records."

        conn.close()
        return {
            "query": sql,
            "results": results,
            "summary": summary,
            "row_count": len(results)
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

    def get_table_names(self) -> list[str]:
        """Get all user table names in the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Get schema information for a specific table."""
        # Sanitize table_name against alphanumeric and underscore
        if not re.match(r"^[A-Za-z0-9_]+$", table_name):
            return []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        rows = cursor.fetchall()
        conn.close()
        # cid, name, type, notnull, dflt_value, pk
        return [
            {
                "column_id": r[0],
                "name": r[1],
                "type": r[2],
                "not_null": bool(r[3]),
                "default_value": r[4],
                "is_primary_key": bool(r[5])
            }
            for r in rows
        ]

    def execute_safe_query(self, sql_query: str) -> dict[str, Any]:
        """Execute a read-only SQL query with safety validation and telemetry.

        Args:
            sql_query: Read-only SQL query (SELECT or PRAGMA)

        Returns:
            Dict containing results, column names, execution latency, or error
        """
        import time
        clean_q = sql_query.strip().rstrip(";")
        upper_q = clean_q.upper()

        # Reject destructive or modifying operations
        prohibited_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "CREATE", "ATTACH", "DETACH"]
        for kw in prohibited_keywords:
            if re.search(rf"\b{kw}\b", upper_q):
                return {
                    "success": False,
                    "error": f"Security Violation: '{kw}' operations are prohibited in read-only console."
                }

        if not (upper_q.startswith("SELECT") or upper_q.startswith("PRAGMA") or upper_q.startswith("EXPLAIN")):
            return {
                "success": False,
                "error": "Only SELECT, PRAGMA, or EXPLAIN statements are permitted."
            }

        t0 = time.time()
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(clean_q)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            conn.close()

            exec_ms = round((time.time() - t0) * 1000, 2)
            results = [dict(r) for r in rows]

            return {
                "success": True,
                "query": clean_q,
                "columns": columns,
                "results": results,
                "row_count": len(results),
                "execution_time_ms": exec_ms
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": round((time.time() - t0) * 1000, 2)
            }

