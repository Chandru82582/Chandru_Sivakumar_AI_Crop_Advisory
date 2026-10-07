# AI-Powered Agriculture Crop Advisory Assistant

Colour palette

#3368A0

#66A3BF

#C8DFDB

#F2EFE7

## Master Implementation Specification (AI Code Generation Guide)

**Instructions for the AI Coding Agent:**
This document serves as the complete system architecture and step-by-step implementation guide. The user has already downloaded the required datasets (`Crop_Yield_Data.csv`) and RAG PDF documents into the `data/` folder.

Your task is to generate the entire Python codebase exactly as specified in the **File-by-File Implementation Guide** (Section 5). You must strictly adhere to the directory structure, implement all requested UI tabs (including the visual What-If simulator and PDF export), and wire up the LangChain SQL, Vision, ML, and LangChain RAG components.

---

## 1. High-Level Architecture Diagram

```text
                                  ┌─────────────────────────────────────────────────────────────┐
                                  │                  Client Tier (Streamlit UI)                 │
                                  │  - Tab 1: Advisory & What-If Soil Simulator (SHAP visual)   │
                                  │  - Tab 2: Dynamic Timetable & Checklist (Organic/Inorganic) │
                                  │  - Tab 3: Crop Doctor (Leaf Image Upload & Diagnosis)       │
                                  │  - Tab 4: Farm History (Natural Language SQL queries)       │
                                  │  - Sidebar: Parameter Sliders & PDF Report Download Button  │
                                  └──────────────────────────────▲──────────────────────────────┘
                                                                 │ HTTP / JSON
                                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       API Gateway & Microservice Backend (FastAPI)                                             │
├────────────────────────────────┬───────────────────────────────────────────┬───────────────────────────────────────────────────┤
│ ENGINE A: LangChain RAG System │ ENGINE B: ML Predictive & XAI Engine      │ ENGINE C: Semantic-to-SQL & History               │
│                                │                                           │                                                   │
│ ┌───────────────────────────┐  │ ┌──────────────────────────────────────┐  │ ┌──────────────────────────────────────────────┐  │
│ │ LangChain EnsembleRetriever│  │ │ XGBoost Multi-Class / Regressor      │  │ │ SQLite Database (Farm Logs & Yields)         │  │
│ │ - BM25Retriever (Sparse)   │  │ │ - Predicts: Crop Suitability & Yield │  │ │ LangChain SQLDatabaseChain                   │  │
│ │ - Chroma VectorStore (Dense│  │ ├──────────────────────────────────────┤  │ │ - Translates "What did I plant last year?"   │  │
│ │ - HuggingFace Embeddings   │  │ │ SHAP Explainer & What-If Sandbox     │  │ │   into SQL queries securely.                 │  │
│ └───────────────────────────┘  │ │ - Waterfall feature attributions     │  │ └──────────────────────────────────────────────┘  │
│                                │ └──────────────────────────────────────┘  │ ENGINE D: Vision (Crop Doctor)                    │
│                                │                                           │ ┌──────────────────────────────────────────────┐  │
│                                │                                           │ │ Local CNN (PyTorch / torchvision MobileNet)  │  │
│                                │                                           │ │ Detects leaf disease -> Triggers RAG cure    │  │
│                                │                                           │ └──────────────────────────────────────────────┘  │
├────────────────────────────────┴───────────────────────────────────────────┴───────────────────────────────────────────────────┤
│ ENGINE E: Multi-Agent LLM Orchestration & Operations (Google GenAI SDK / LangChain LCEL)                                       │
│   ├── Agronomy Agent     ──► Reads RAG docs & generates a Day 1 to Harvest dynamic timetable (Organic + Chemical options).     │
│   ├── Risk / Econ Agent  ──► Evaluates ML risk metrics and suggests companion crops / intercropping.                           │
│   └── PDF Generator      ──► ReportLab script to compile the complete advisory into a downloadable PDF dossier.                │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Assumed Pre-existing Data (Do Not Generate)

The user has the following files locally. Your code must read from these paths:

1. **Tabular Data:** `data/raw/Crop_Yield_Data.csv` (Columns: `Crop`, `Crop_Year`, `Season`, `State`, `Area`, `Production`, `Yield`, `Fertilizer`, `Pesticide`, `N`, `P`, `K`, `pH`, `Temperature`, `Rainfall`, `Humidity`).
2. **RAG PDFs:** `data/agronomy_docs/*.pdf` (Various agricultural extension manuals).
3. **Database:** `data/farmer_profile.db` (Will be auto-initialized by your code).

---

## 3. Technology Stack & Dependencies

Create a `requirements.txt` file with the following dependencies:

```text
fastapi>=0.110.0
uvicorn>=0.28.0
pydantic>=2.6.0
pandas>=2.2.0
scikit-learn>=1.4.0
xgboost>=2.0.0
chromadb>=0.4.24
rank-bm25>=0.2.2
sentence-transformers>=2.5.0
langchain>=0.1.12
langchain-community>=0.0.28
langchain-chroma>=0.1.0
langchain-huggingface>=0.0.1
langchain-experimental>=0.0.54
langchain-google-genai>=0.0.11
streamlit>=1.32.0
shap>=0.45.0
pypdf>=4.1.0
reportlab>=4.1.0
torch>=2.2.0
torchvision>=0.17.0
pillow>=10.2.0
matplotlib>=3.8.0
```

---

## 4. Strict Directory Structure

```text
crop-intelligence-system/
├── data/
│   ├── raw/Crop_Yield_Data.csv
│   ├── agronomy_docs/
│   └── farmer_profile.db (Auto-generated SQLite)
├── models/
│   ├── train_xgboost.py
│   └── (yield_model.joblib, crop_model.joblib, explainer.joblib will be saved here)
├── core/
│   ├── __init__.py
│   ├── guardrails.py
│   ├── rag_retriever.py
│   ├── ml_engine.py
│   ├── sql_agent.py
│   ├── plant_doctor.py
│   ├── pdf_builder.py
│   └── agents.py
├── api/
│   ├── __init__.py
│   └── main.py
├── app.py
└── requirements.txt
```

---

## 5. File-by-File Implementation Guide (Prompts for Code Gen)

### Phase 1: Machine Learning & XAI

**File: `models/train_xgboost.py`**

* Load `data/raw/Crop_Yield_Data.csv`.
* Features: `N, P, K, pH, Temperature, Rainfall, Area`.
* Targets: `Yield` (Regressor) and `Crop` (Classifier).
* Train XGBoost models.
* **Crucial XAI Step:** Initialize a `shap.TreeExplainer` on the trained Classifier and save it as `models/explainer.joblib` alongside the models.

### Phase 2: Core Data, Vision & LangChain RAG Modules

**File: `core/sql_agent.py`**

* Initialize `sqlite3` connection to `data/farmer_profile.db`. Create a table `crop_history` (id, season, crop_planted, yield_achieved, nitrogen_used, profit).
* Use `langchain_experimental.sql.SQLDatabaseChain` or `create_sql_query_chain` paired with `ChatGoogleGenerativeAI` (Gemini).
* Implement `ask_farm_history(natural_query: str)` which translates text to SQL, runs it, and returns a natural language summary.

**File: `core/rag_retriever.py`**

* **Strictly use LangChain components here.**
* Implement a `build_or_load_vectorstore()` function:
  * Use `PyPDFDirectoryLoader` to read `data/agronomy_docs/`.
  * Use `RecursiveCharacterTextSplitter` to chunk the documents.
  * Use `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` to embed them.
  * Store them in `langchain_chroma.Chroma`.
* Implement a Hybrid RRF Retriever (`get_hybrid_retriever()`):
  * Initialize a `BM25Retriever` from the chunked documents.
  * Initialize the Chroma retriever (`vectorstore.as_retriever()`).
  * Combine both using `langchain.retrievers.EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5])`.

**File: `core/ml_engine.py`**

* Load `.joblib` models and the `shap.TreeExplainer`.
* `predict(features: dict)`: Returns expected yield, top 3 crops, and risk bounds (P10/P90).
* `get_shap_values(features: dict)`: Returns the SHAP base value and feature contributions for the What-If UI waterfall chart.

### Phase 3: Orchestration & PDF Export

**File: `core/agents.py`**

* Implement `AdvisoryOrchestrator` using LangChain's Expression Language (LCEL) or custom chains.
* The orchestrator must take the User Query, query the `EnsembleRetriever` from `rag_retriever.py` for context, take the predictions from `ml_engine.py`, and pass them all into a `ChatPromptTemplate` targeting `ChatGoogleGenerativeAI`.
* Prompt instructions must explicitly demand a **Day-to-Day Dynamic Timetable** formatted as a checklist.
* Prompt must include instructions for **Intercropping / Companion planting recommendations**.

**File: `core/pdf_builder.py`**

* Implement `generate_advisory_pdf(farmer_data, ml_metrics, agent_text)`.
* Use `reportlab.platypus`.
* Format the advisory, expected yield, and the day-to-day timetable into a cleanly styled 2-page PDF. Save to memory buffer (`io.BytesIO()`).

### Phase 4: API Gateway

**File: `api/main.py`**

* Initialize `FastAPI`.
* **`POST /api/v1/recommend`**: Calls ML -> LangChain Retriever -> Orchestrator Agent. Returns JSON.
* **`POST /api/v1/simulate`**: Accepts tweaked parameters (for What-If), calls `ml_engine.get_shap_values()`.
* **`POST /api/v1/history`**: Passes natural language to `sql_agent`.
* **`POST /api/v1/diagnose`**: Accepts an UploadFile, calls `plant_doctor`, returns disease name and RAG treatment.
* **`POST /api/v1/export_pdf`**: Returns the PDF bytes as a `StreamingResponse`.

### Phase 5: The Streamlit Application

**File: `app.py`**

* **Sidebar:** Input sliders (N, P, K, pH, Rain, Temp, Area). A "Download PDF Report" button that hits `/api/v1/export_pdf`.
* **Main Layout (`st.tabs`):**
  * **Tab 1: Advisory & What-If Simulator:**
    * Display the Master Agent's text.
    * Render a `st.pyplot` matplotlib chart of the SHAP waterfall values returned from `/simulate`.
    * Add interactive "What-If" sliders here that instantly re-trigger the `/simulate` endpoint to show how probabilities change.
  * **Tab 2: Smart Operations:**
    * Display the dynamic timetable checklist (populated via the LangChain RAG pipeline). Allow the user to select an "Organic" vs "Chemical" radio button.
  * **Tab 3: Crop Doctor:**
    * `st.file_uploader` for leaf images. Send to `/diagnose` and display the identified pathology.
  * **Tab 4: Farm History:**
    * `st.chat_message` interface for natural language to SQL communication via LangChain.

---

## 6. Execution Sequence (How to run)

1. `pip install -r requirements.txt`
2. `python models/train_xgboost.py` (Wait for `.joblib` files).
3. `export GOOGLE_API_KEY="your-google-api-key"` (LangChain relies on this env variable).
4. Terminal 1: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
5. Terminal 2: `streamlit run app.py`
