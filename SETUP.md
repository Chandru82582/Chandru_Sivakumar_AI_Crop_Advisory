# Phase 2 Setup and Data Ingestion Guide

## Project Structure (After Cleanup)

```
D:\CropAdvisory/
├── core/                          # Core modules (Phase 2)
│   ├── __init__.py
│   ├── ml_engine.py               # XGBoost predictions + SHAP
│   ├── rag_retriever.py           # Hybrid RAG (BM25 + Dense vectors)
│   ├── sql_agent.py               # Natural Language SQL queries
│   └── guardrails.py              # Safety & input validation
├── data/                          # Generated during ingestion
│   ├── farmer_profile.db          # SQLite (auto-created)
│   └── chroma_db/                 # Vector store (auto-created)
├── dataAnalysis/
│   ├── agronomy_docs/             # 4 PDF reference documents
│   └── data/                      # CSV files for ingestion
├── artifact/
│   └── crop_recommender.joblib    # Trained ML model
├── ml/
│   └── train.py                   # Model training script
├── data_ingestion.py              # PDF + CSV ingestion pipeline
├── requirements.txt               # Python dependencies
├── README.md                      # Complete project documentation
└── ai_implementation_specification.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Create .env file (if not exists)
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

### 3. Run Data Ingestion
```bash
python data_ingestion.py
```

This will:
- **PDF Processing:**
  - Load PDFs from `dataAnalysis/agronomy_docs/`
  - Split using RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
  - Embed with HuggingFace all-MiniLM-L6-v2
  - Store in ChromaDB at `data/chroma_db/`

- **Tabular Data Processing:**
  - Read CSV files from `dataAnalysis/data/`
  - Create SQLite database at `data/farmer_profile.db`
  - Initialize crop_history table schema

## What Each Module Does

### `data_ingestion.py` (NEW)
Central pipeline for loading all data:
- **PDF Processing:**
  - `PyPDFDirectoryLoader` reads all PDFs
  - `RecursiveCharacterTextSplitter` chunks documents intelligently
  - `HuggingFaceEmbeddings` creates dense vectors
  - `Chroma` vector store persists data for fast retrieval

- **Tabular Data Processing:**
  - Reads CSV files with pandas
  - Creates SQLite database schema
  - Stores data in normalized tables

- **Verification:**
  - Confirms ChromaDB creation
  - Validates SQLite schema
  - Logs ingestion statistics

### `core/rag_retriever.py` (Updated)
Hybrid retrieval system:
```python
from core.rag_retriever import RAGRetriever

retriever = RAGRetriever()  # Auto-loads existing ChromaDB
docs = retriever.retrieve("fertilizer for wheat", k=5)
context = retriever.get_context_string("disease treatment")
```

- Combines BM25 (sparse) and Dense (embeddings) retrieval
- 50/50 ensemble weighting
- Retrieves up to k documents
- Returns with metadata (source, page)

### `core/sql_agent.py`
Natural language SQL interface:
```python
from core.sql_agent import FarmHistoryAgent

agent = FarmHistoryAgent()
agent.add_farm_record(season="Kharif", crop_planted="Rice", ...)
result = agent.ask_farm_history("What was my yield last year?")
```

- Uses Gemini API for semantic SQL translation
- Secure query generation via LangChain
- Returns results with natural language summaries

### `core/ml_engine.py`
Machine learning predictions:
```python
from core.ml_engine import MLEngine

engine = MLEngine(model_path="artifact/crop_recommender.joblib")
predictions = engine.predict(features)
shap_values = engine.get_shap_values(features)
```

- Loads calibrated Extra Trees classifier
- Returns top 3 crops with probabilities
- Generates SHAP feature importances
- Validates input data quality

### `core/guardrails.py`
Safety and validation:
```python
from core.guardrails import AdvisoryGuardrails

is_valid, warnings = AdvisoryGuardrails.validate_features(features)
safe_text = AdvisoryGuardrails.sanitize_advisory_text(text)
```

- Validates feature ranges (pH, humidity, etc.)
- Detects harmful content
- Generates warnings for anomalies

## Data Files

### PDFs (Source)
Located in `dataAnalysis/agronomy_docs/`:
1. **Final-Bulletin_77.pdf** - Agricultural extension bulletin
2. **ICAR En-Kharif Agro-Advisories for Farmers 2025.pdf** - Kharif season guidance
3. **ICAR-ISRI_Drought_contingency_plan_for_Sugarcane_crop.pdf** - Drought management
4. **Natural Farming for Sustainable Agriculture.pdf** - Organic farming practices

### CSV Files (Source)
Located in `dataAnalysis/data/`:
- `crop_yield.csv` - Historical yield data
- `state_soil_data.csv` - Soil characteristics by state
- `state_weather_data_1997_2020.csv` - Weather history

### Generated Databases

**ChromaDB** (`data/chroma_db/`):
- Vector embeddings of PDF chunks
- Persisted for fast initialization
- Created by `data_ingestion.py`

**SQLite** (`data/farmer_profile.db`):
```sql
-- Auto-created schema
CREATE TABLE crop_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season TEXT NOT NULL,
    crop_planted TEXT NOT NULL,
    yield_achieved REAL,
    nitrogen_used REAL,
    profit REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Additional tables from CSV ingestion
-- (auto-created from CSV filenames)
```

## Usage Examples

### 1. Ingest Data
```bash
python data_ingestion.py
# Output: Logs ingestion progress and statistics
```

### 2. Query Farm History
```python
from core.sql_agent import FarmHistoryAgent

agent = FarmHistoryAgent()
result = agent.ask_farm_history("Which crop gave me the highest profit?")
print(result['summary'])
```

### 3. Retrieve Agricultural Context
```python
from core.rag_retriever import RAGRetriever

retriever = RAGRetriever()
context = retriever.get_context_string("How to prevent powdery mildew?")
print(context)
```

### 4. Get Crop Predictions
```python
from core.ml_engine import MLEngine

engine = MLEngine()
features = {
    "N": 50, "P": 20, "K": 30, "pH": 6.5,
    "avg_temp_c": 25.0, "total_rainfall_mm": 800.0,
    "avg_humidity_percent": 65.0
}
result = engine.predict(features)
print(f"Top crop: {result['top_crop']}")
```

## Next Steps

1. **Phase 3:** Implement orchestration layer
   - `core/agents.py` - LangChain agent orchestration
   - `core/pdf_builder.py` - PDF report generation
   - Combines all Phase 2 modules into advisory system

2. **Phase 4:** Build API Gateway
   - `api/main.py` - FastAPI endpoints
   - `/api/v1/recommend` - Get recommendations
   - `/api/v1/simulate` - What-if analysis
   - `/api/v1/history` - Query farm history
   - `/api/v1/export_pdf` - Download report

3. **Phase 5:** Create Streamlit UI
   - `app.py` - Interactive dashboard
   - 4 tabs: Advisory, Operations, Crop Doctor, Farm History

## Troubleshooting

### ChromaDB not found
```bash
python data_ingestion.py --force-rebuild
```

### SQLite locked
Close any other Python processes accessing the database

### Gemini API errors
- Verify GEMINI_API_KEY is set in .env
- Check internet connection
- Ensure API key has SQL query permissions

### Import errors
```bash
pip install -r requirements.txt --upgrade
```

---

**Status:** Phase 2 Core Modules + Data Ingestion Ready ✅
