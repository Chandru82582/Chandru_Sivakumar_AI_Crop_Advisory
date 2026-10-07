# Phase 2 Quick Reference Guide

## Essential Commands

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Load data into ChromaDB + SQLite
python data_ingestion.py

# 3. Test individual modules
python -c "from core.ml_engine import MLEngine; engine = MLEngine(); print('ML Engine Ready')"
```

## Core Modules at a Glance

| Module | Purpose | Key Method | Input | Output |
|--------|---------|------------|-------|--------|
| `ml_engine.py` | Crop predictions | `predict(features)` | Dict with N,P,K,pH,temp,rain,humidity | Top 3 crops + confidence |
| `rag_retriever.py` | Document search | `retrieve(query, k=5)` | Search text | List of relevant chunks |
| `sql_agent.py` | Farm history | `ask_farm_history(query)` | Natural language | SQL results + summary |
| `guardrails.py` | Safety checks | `validate_features(features)` | Feature dict | (is_valid, warnings) |

## File Locations (After Setup)

```
Before ingestion:
  dataAnalysis/agronomy_docs/  → PDF source documents
  dataAnalysis/data/           → CSV source data
  artifact/                    → Pre-trained ML model

After ingestion:
  data/chroma_db/              → Vector embeddings (auto-created)
  data/farmer_profile.db       → SQLite database (auto-created)
```

## Quick Code Examples

### 1. Get Crop Prediction
```python
from core.ml_engine import MLEngine

engine = MLEngine()
result = engine.predict({
    "N": 50, "P": 20, "K": 30, "pH": 6.5,
    "avg_temp_c": 25.0, "total_rainfall_mm": 800.0,
    "avg_humidity_percent": 65.0
})
print(f"Top crop: {result['top_crop']} ({result['overall_confidence']:.1%})")
```

### 2. Search Agricultural Documents
```python
from core.rag_retriever import RAGRetriever

retriever = RAGRetriever()
docs = retriever.retrieve("How to prevent crop diseases?", k=3)
for doc in docs:
    print(f"Source: {doc['source']}\n{doc['content'][:200]}...")
```

### 3. Query Farm History
```python
from core.sql_agent import FarmHistoryAgent

agent = FarmHistoryAgent()
agent.add_farm_record(season="Kharif", crop_planted="Rice", 
                      yield_achieved=5.5, nitrogen_used=80)
result = agent.ask_farm_history("Which season had highest profit?")
print(result['summary'])
```

### 4. Validate Input Safety
```python
from core.guardrails import AdvisoryGuardrails

features = {"pH": 6.5, "N": 50, "P": 20, "K": 30}
is_valid, warnings = AdvisoryGuardrails.validate_features(features)
print(f"Valid: {is_valid}, Warnings: {warnings}")
```

## Environment Setup

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "your-api-key"

# Windows (CMD)
set GEMINI_API_KEY=your-api-key

# Linux/Mac (Bash)
export GEMINI_API_KEY=your-api-key
```

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: langchain` | Run `pip install -r requirements.txt --upgrade` |
| `ChromaDB not found` | Run `python data_ingestion.py` to create vector store |
| `GEMINI_API_KEY error` | Set environment variable with your API key |
| `SQLite locked` | Close other Python processes accessing the database |

## Database Schema

### ChromaDB Collections
- **Documents:** PDF chunks with embeddings
- **Metadata:** source file, page number
- **Retrieval:** BM25 (50%) + Dense vectors (50%)

### SQLite Tables
```sql
crop_history (
  id INTEGER PRIMARY KEY,
  season TEXT,
  crop_planted TEXT,
  yield_achieved REAL,
  nitrogen_used REAL,
  profit REAL,
  created_at TIMESTAMP
)
```

## Integration Flow (Phase 3+)

```
User Input
    ↓
ML Engine (Prediction)
    ↓
RAG Retriever (Context)
    ↓
LangChain Agent (Orchestration)
    ↓
PDF Builder (Report)
    ↓
API Gateway / Streamlit UI
```

## Performance Notes

- **First initialization:** ~30 seconds (PDF chunking + embedding)
- **Subsequent loads:** <1 second (ChromaDB cache)
- **ML predictions:** <100ms per query
- **RAG retrieval:** <200ms per query
- **SQL queries:** <500ms per query

## Next Steps for Development

1. **Phase 3:** Create `core/agents.py` for orchestration
2. **Phase 4:** Create `api/main.py` for FastAPI
3. **Phase 5:** Create `app.py` for Streamlit UI

---

**Last Updated:** Phase 2 Implementation Complete | LangChain 0.2.0+
