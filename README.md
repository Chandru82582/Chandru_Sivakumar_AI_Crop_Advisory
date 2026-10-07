# AI-Powered Agriculture Crop Advisory Assistant

An intelligent agricultural advisory system that combines machine learning predictions with retrieval-augmented generation (RAG) to provide crop recommendations, yield forecasting, and farm history management.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit UI (Phase 5)                    │
├─────────────────────────────────────────────────────────────┤
│                   FastAPI Gateway (Phase 4)                 │
├─────────────────────────────────────────────────────────────┤
│            LangChain Orchestration (Phase 3)                │
├──────────────────┬──────────────────────────┬────────────────┤
│  ML Engine       │  RAG Retriever           │  SQL Agent     │
│  (XGBoost)       │  (BM25 + Dense + Chroma) │  (Gemini LLM)  │
└──────────────────┴──────────────────────────┴────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Create .env file with Gemini API key
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

### 3. Ingest Data

```bash
# Load PDFs to ChromaDB and tabular data to SQLite
python data_ingestion.py
```

### 4. Train ML Model (if needed)

```bash
python ml/train.py --out-dir artifact
```

## Core Modules

### `core/ml_engine.py`

Machine learning predictions with XGBoost.

- Load pre-trained crop recommender model
- Predict top 3 crops with confidence scores
- Get SHAP feature importances
- Input validation and drift detection

**Usage:**

```python
from core.ml_engine import MLEngine

engine = MLEngine(model_path="artifact/crop_recommender.joblib")
result = engine.predict({
    "N": 50, "P": 20, "K": 30, "pH": 6.5,
    "avg_temp_c": 25.0, "total_rainfall_mm": 800.0,
    "avg_humidity_percent": 65.0
})
```

### `core/rag_retriever.py`

Hybrid RAG system combining sparse (BM25) and dense (embeddings) retrieval.

- Loads PDFs from `dataAnalysis/agronomy_docs/`
- Stores documents in ChromaDB vector store
- BM25 + Dense embedding ensemble retrieval
- Persistent storage for fast re-initialization

**Usage:**

```python
from core.rag_retriever import RAGRetriever

retriever = RAGRetriever()
docs = retriever.retrieve("best fertilizer for wheat", k=5)
context = retriever.get_context_string("crop disease treatment")
```

### `core/sql_agent.py`

Natural language interface to farm history database.

- SQLite database: `data/farmer_profile.db`
- LangChain SQL query generation via Gemini API
- Add and query farm records
- Summary statistics

**Usage:**

```python
from core.sql_agent import FarmHistoryAgent

agent = FarmHistoryAgent()
agent.add_farm_record(
    season="Kharif", 
    crop_planted="Rice",
    yield_achieved=5.2,
    nitrogen_used=80,
    profit=45000
)
result = agent.ask_farm_history("What was my rice yield last year?")
```

### `core/guardrails.py`

Safety validation and content sanitization.

- Feature validation (pH ranges, humidity limits)
- Harmful content detection
- Input warnings for anomalies

**Usage:**

```python
from core.guardrails import AdvisoryGuardrails

is_valid, warnings = AdvisoryGuardrails.validate_features(features)
safe_text = AdvisoryGuardrails.sanitize_advisory_text(advisory)
```

## Data Structure

### Documents (ChromaDB)

- **Storage:** `data/chroma_db/`
- **Source:** PDFs from `dataAnalysis/agronomy_docs/`
- **Processing:**
  - PyPDFDirectoryLoader for PDF extraction
  - RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
  - HuggingFace embeddings (all-MiniLM-L6-v2)
  - BM25 + Dense ensemble retrieval

### Tabular Data (SQLite)

- **Database:** `data/farmer_profile.db`
- **Tables:**
  - `crop_history`: id, season, crop_planted, yield_achieved, nitrogen_used, profit, created_at

### ML Models (Artifacts)

- **Location:** `artifact/`
- **Model:** `crop_recommender.joblib` (calibrated Extra Trees classifier)
- **Classes:** 25 crop types
- **Features:** N, P, K, pH, temperature, rainfall, humidity

## File Structure

```
D:\CropAdvisory/
├── core/
│   ├── __init__.py
│   ├── ml_engine.py           # XGBoost predictions + SHAP
│   ├── rag_retriever.py       # Hybrid RAG (BM25 + Dense)
│   ├── sql_agent.py           # NL-SQL via Gemini
│   └── guardrails.py          # Safety & validation
├── data/
│   ├── farmer_profile.db      # SQLite (auto-created)
│   ├── chroma_db/             # Vector store (auto-created)
│   └── (other data files)
├── artifact/
│   └── crop_recommender.joblib  # Trained ML model
├── dataAnalysis/
│   ├── agronomy_docs/         # 4 PDF documents
│   └── data/                  # Training CSV files
├── data_ingestion.py          # PDF + tabular data loader
├── requirements.txt
├── README.md
└── ai_implementation_specification.md
```

## Data Ingestion

The `data_ingestion.py` script handles:

1. **PDF Processing:**

   - Loads all PDFs from `dataAnalysis/agronomy_docs/`
   - Chunks using RecursiveCharacterTextSplitter
   - Embeds with HuggingFace (all-MiniLM-L6-v2)
   - Stores in ChromaDB
2. **Tabular Data:**

   - Reads CSV files from `dataAnalysis/data/`
   - Creates SQLite database
   - Initializes crop_history table

**Run ingestion:**

```bash
python data_ingestion.py
```

## Phase Development

### ✅ Phase 2: Core Modules

- ML Engine with SHAP
- SQL Agent with Gemini
- RAG Retriever (Hybrid)
- Guardrails

### Phase 3: Orchestration (Next)

- LangChain agents for advisory generation
- PDF report builder
- Integration of all Phase 2 modules

### Phase 4: API Gateway

- FastAPI endpoints
- Request/response handling
- Model serving

### Phase 5: UI

- Streamlit dashboard
- Interactive tabs
- PDF download

## Environment Variables

```bash
GEMINI_API_KEY=<your-google-api-key>  # Required for SQL queries via LLM
```

## Dependencies

See `requirements.txt` for full list. Key packages:

- **LangChain:** `langchain`, `langchain-community`, `langchain-google-genai`
- **RAG:** `chromadb`, `sentence-transformers`, `rank-bm25`
- **ML:** `xgboost`, `scikit-learn`, `shap`
- **Data:** `pandas`, `sqlalchemy`
- **API:** `fastapi`, `uvicorn`
- **UI:** `streamlit`

## Testing

Run the test suite:

```bash
python test_phase2.py
```

Tests cover:

- ML predictions
- RAG document retrieval
- SQL agent operations
- Guardrails validation

## Development Notes

1. **ChromaDB Persistence:** Vector store is persisted at `data/chroma_db/` for fast re-initialization.
2. **SQLite Database:** Auto-initializes on first run with crop_history schema.
3. **LLM Integration:** Requires GEMINI_API_KEY in environment for semantic SQL translation.
4. **Feature Standardization:** All numeric features should be normalized before prediction.

## Next Steps

1. Implement Phase 3 orchestration layer
2. Add API endpoints (Phase 4)
3. Build Streamlit UI (Phase 5)
4. End-to-end testing and deployment

---

**Status:** Phase 2 Complete | Phase 3 In Progress
