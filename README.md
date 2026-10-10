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

### `core/plant_doctor.py`

Computer vision leaf disease diagnosis and treatment recommendations.

- Automatic pathology diagnosis from leaf images
- Organic and chemical treatment protocols
- Preventive agronomic practices

## Phase Development

### ✅ All Phases Complete: Production Ready (v3.5)

- **Phase 1: ML Model Training:** Calibrated ensemble crop recommender (`artifact/crop_recommender.joblib`).
- **Phase 2: Core Intelligent Modules:** ML Engine with SHAP, RAG Retriever, SQL Agent, Guardrails, Plant Doctor.
- **Phase 3: LangChain Orchestration:** Dynamic multi-stage advisory and companion planting engine.
- **Phase 4: FastAPI Gateway:** REST API endpoints serving predictions, simulations, history, diagnosis, and PDF export (`api/main.py`).
- **Phase 5: Reactive Streamlit UI:** Production unified dashboard with real-time What-If simulator, interactive timeline checklists, vision doctor, and analytics (`app.py`).

## Environment Variables

```bash
GEMINI_API_KEY=<your-google-api-key>  # For LLM queries and advisory generation
# Note: The platform also includes comprehensive agronomic fallbacks if API key is not present.
```

## Running the Application

### 🌾 Streamlit Production UI

```bash
streamlit run app.py
```

*Access at: `http://localhost:8501`*

### ⚡ FastAPI Gateway

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

*Interactive Swagger docs at: `http://localhost:8000/docs`*

## Testing

Run the module test suite:

```bash
python test_files/test_phase2.py
```

Tests cover:

- ML predictions & SHAP explainability
- Hybrid RAG document retrieval (BM25 + Vector)
- SQL agent operations & stats
- Safety guardrails validation
- Plant Doctor vision diagnosis

---

**Status:** Complete & Production Ready 🌾
