# AI-Powered Agriculture Crop Advisory Assistant

## Complete Engineering Blueprint & System Architecture

This document serves as the master specification for building, deploying, and evaluating a locally runnable Agricultural Decision Support System (DSS). It combines Tabular Machine Learning (XGBoost) for predictive accuracy with a Multi-Agent Retrieval-Augmented Generation (RAG) pipeline for domain-specific agronomic reasoning.

## 1. Project Datasets & RAG Knowledge Base

To run this system, you will ingest tabular data for the ML model and unstructured PDF documents for the RAG vector database.

### A. Primary Tabular Datasets (For XGBoost Training & Case Matching)

1. **Primary Kaggle Dataset:** [Crop Yield Data with Soil and Weather Dataset by Anshu Mishra](https://www.kaggle.com/datasets/anshumish/crop-yield-data-with-soil-and-weather-dataset)

   * *Usage:* Train the XGBoost Regressor/Classifier on `N`, `P`, `K`, `pH`, `Rainfall`, `Temperature` to predict `Crop` and `Yield`.

2. **Alternative Dataset:** [FAOSTAT Crop Production, Yield and Harvested Area](https://www.fao.org/faostat/en/#data/QCL)

   * *Usage:* Global baseline data for fallback yield predictions.

### B. Unstructured Agronomy Documents (For ChromaDB RAG Vector Store)

These publicly available Indian agricultural extension guidelines serve as the ground truth for your Agronomy and Risk LLM Agents. Download these as PDFs and place them in your `data/agronomy_docs/` folder.

1. **ICAR Organic Farming Guidelines:** Essential concepts on organic soil management, crop residue recycling, and bio-fertilizer usage (e.g., Beejamrit, Jeevamrit). [Download ICAR-IIVR Organic Farming Bulletin](https://icariivr.org.in/wp-content/uploads/2025/10/Final-Bulletin_77.pdf)

2. **MANAGE - Natural Farming for Sustainable Agriculture:** Guidelines on chemical-free farming, soil ecosystem health, and utilizing locally sourced inputs for climate resilience. [Download MANAGE E-Book](https://www.manage.gov.in/publications/eBooks/Natural%20Farming%20for%20Sustainable%20Agriculture.pdf)

3. **ICAR Kharif Agro-Advisories for Farmers:** Specific advisories for the Kharif season, including optimal planting windows and irrigation spacing. [Download ICAR Advisory](https://icar.gov.in/sites/default/files/Circulars/ICAR%20En-Kharif%20Agro-Advisories%20for%20Farmers%202025.pdf)

4. **ICAR Drought Contingency Plan:** Strategies for managing dry spells, such as frequent light irrigations, dust mulching, and alternative crop suggestions. [Download ICAR-ISRI Plan](https://www.isri.res.in/iisr/download/miscellaneous/ICAR-ISRI_Drought_contingency_plan_for_Sugarcane_crop.pdf)

## 2. High-Level Architecture Diagram

```
                                  ┌─────────────────────────────────────────────────────────────┐
                                  │                  Client Tier (Streamlit UI)                 │
                                  │  - Natural Language Query Input & Parameter Sliders         │
                                  │  - Tabs: Advisory | Yield Metrics | Historical Cases        │
                                  │  - Interactive "What-If" Soil Simulator Sandbox             │
                                  └──────────────────────────────▲──────────────────────────────┘
                                                                 │ HTTP / JSON
                                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       API Gateway & Microservice Backend (FastAPI)                                             │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ [Layer 0: Input Guardrails] ──► Pydantic validation (e.g., pH bounds [3-10], Rainfall [0-4000mm], NPK [0-500])                 │
├────────────────────────────────┬───────────────────────────────────────────────┬───────────────────────────────────────────────┤
│ ENGINE A: Semantic Retrieval   │ ENGINE B: ML Predictive & Risk Engine         │ ENGINE C: Knowledge Base RAG                  │
│                                │                                               │                                               │
│ ┌───────────────────────────┐  │ ┌──────────────────────────────────────────┐  │ ┌──────────────────────────────────────────┐  │
│ │ Hybrid Search (RRF)       │  │ │ XGBoost Multi-Class / Regressor          │  │ │ Vector Store (ChromaDB)                  │  │
│ │ - BM25 Sparse Index       │  │ │ - Predicts: Crop Suitability & Yield     │  │ │ - Embeddings: all-MiniLM-L6-v2           │  │
│ │ - Dense Vector Embeddings │  │ ├──────────────────────────────────────────┤  │ │ - Corpus: ICAR/FAO Extension PDFs        │  │
│ │ (Matches historical Kaggle│  │ │ Risk Quantile Engine                     │  │ ├──────────────────────────────────────────┤  │
│ │ case rows by similarity)  │  │ │ - Calculates P10/P90 Yield bounds        │  │ │ Metadata Filter Engine                   │  │
│ └───────────────────────────┘  │ └──────────────────────────────────────────┘  │ │ - {"method": "organic", "season": "rabi"}│  │
│                                │                                               │ └──────────────────────────────────────────┘  │
├────────────────────────────────┴───────────────────────────────────────────────┴───────────────────────────────────────────────┤
│ ENGINE D: Multi-Agent LLM Orchestration Layer (LangChain / Gemini API)                                                         │
│                                                                                                                                │
│   ├── Case Retrieval Agent ──► Aggregates top-k similar historical records & mandi returns.                                    │
│   ├── Agronomy Agent       ──► Reads RAG docs & designs day-to-day organic/inorganic crop schedules.                             │
│   ├── Risk / Econ Agent    ──► Evaluates XGBoost P10 yield risks, water stress, and market contingencies.                        │
│   └── Master Synthesizer   ──► Reconciles outputs, compares historical cases, and formats final JSON advisory.                   │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ ENGINE E: Observability & Continuous Evaluation                                                                                │
│                                                                                                                                │
│   ├── DeepEval / G-Eval    ──► LLM-as-a-Judge to evaluate semantic safety (preventing toxic fertilizer recommendations).         │
│   └── Audit Database       ──► SQLite logging of queries, farmer feedback, and realized yields for future XGBoost retraining.    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘


```

## 3. Engineering Design Decisions & Trade-Offs

### A. Hybrid Search vs. Pure SQL vs. Pure Vector RAG

* **Decision:** We use Reciprocal Rank Fusion (RRF) to combine BM25 (keyword search) with Dense Embeddings (ChromaDB) for historical case matching.

* **Trade-Off:**

  * *Why not pure SQL?* SQL requires exact matches. If a query states `Rainfall = 820mm`, but the database only has `815mm` and `825mm`, a strict SQL `WHERE` clause returns 0 results.

  * *Why not pure Vectors?* Pure dense embeddings often blur categorical constraints (e.g., mixing up "Punjab" and "Haryana" because they are geographically similar). Hybrid RRF ensures we respect strict categorical boundaries while matching semantic soil conditions.

### B. XGBoost Predictive Modeling vs. LLM Tabular Guessing

* **Decision:** We use a locally trained XGBoost model to predict crop yield and suitability, feeding its output *into* the LLM.

* **Trade-Off:** LLMs are autoregressive text generators; they hallucinate heavily when asked to perform multi-variable non-linear regression on numerical arrays (N, P, K, pH). XGBoost provides sub-millisecond, mathematically bounded predictions (yielding a 15-30% accuracy improvement over raw database lookup), while the LLM provides the natural language explanation.

### C. Multi-Agent LLM Architecture vs. Monolithic Prompting

* **Decision:** We split reasoning into an Agronomy Agent, a Risk Agent, and an Orchestrator.

* **Trade-Off:** A single massive prompt instructing an LLM to "analyze soil, predict risk, compare historical cases, and generate an organic schedule" leads to context saturation—the LLM will inevitably skip steps. Sub-agents increase API calls (and slight latency) but guarantee modular, comprehensive, and highly accurate outputs.

## 4. Technology Stack

* **Front-End:** `Streamlit` (Rapid Python-based UI for data science widgets, sliders, and charting).

* **Back-End API:** `FastAPI` (High-performance async microservice) & `Uvicorn`.

* **Machine Learning:** `Scikit-Learn`, `XGBoost`, `SHAP` (for Explainable AI).

* **Vector Database:** `ChromaDB` (Local, lightweight vector store requiring no cloud infrastructure).

* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Runs locally on CPU).

* **LLM Orchestration:** `LangChain` / `Google GenAI SDK` (Gemini 2.5 Flash for high-speed agentic reasoning).

* **LLM Evaluation:** `DeepEval` (For automated G-Eval agronomic safety checks).

* **Relational DB:** `SQLite` (For storing user feedback and audit logs natively without a heavy database server).

## 5. Project Directory Structure

```
crop-intelligence-system/
│
├── data/
│   ├── raw/
│   │   └── Crop_Yield_Data.csv              # Kaggle Dataset
│   ├── agronomy_docs/                       # Downloaded ICAR/FAO PDFs
│   │   ├── ICAR_Organic_Farming.pdf
│   │   ├── MANAGE_Natural_Farming.pdf
│   │   ├── ICAR_Kharif_Advisory.pdf
│   │   └── ICAR_Drought_Plan.pdf
│   └── feedback_audit.db                    # SQLite DB (auto-generated)
│
├── models/
│   ├── train_xgboost.py                     # Script to train ML model
│   └── yield_model.joblib                   # Serialized XGBoost model
│
├── core/
│   ├── __init__.py
│   ├── config.py                            # API keys & global thresholds
│   ├── guardrails.py                        # Pydantic schemas for IO validation
│   ├── hybrid_retriever.py                  # BM25 + ChromaDB search logic
│   ├── ml_engine.py                         # XGBoost inference & SHAP explainability
│   ├── agents.py                            # Multi-agent LLM prompts & chains
│   └── deepeval_audit.py                    # LLM-as-a-judge quality testing
│
├── api/
│   ├── __init__.py
│   └── main.py                              # FastAPI backend routes
│
├── app.py                                   # Streamlit Front-End Interface
├── requirements.txt                         # Python dependencies
└── README.md                                # Setup instructions


```

## 6. Implementation Steps

### Step 1: Model Training (`models/train_xgboost.py`)

1. Load the Kaggle dataset.

2. Preprocess features: `N, P, K, pH, Rainfall, Temperature`.

3. Train an XGBoost Regressor predicting `Yield`.

4. Train an XGBoost Classifier predicting `Crop`.

5. Serialize the models using `joblib`.

### Step 2: Core Components (`core/`)

1. **`guardrails.py`:** Define Pydantic models for request/response payloads to ensure inputs are within valid agronomic bounds (e.g., pH between 3-10).

2. **`hybrid_retriever.py`:** Combine BM25 with `ChromaDB`. Parse the Kaggle dataset into a searchable format and index the PDFs in `data/agronomy_docs/`. Implement Reciprocal Rank Fusion (RRF).

3. **`ml_engine.py`:** Load the `.joblib` models. Create functions to predict yield and calculate P10/P90 risk bounds based on environmental stressors. Include SHAP logic for explainability.

4. **`agents.py`:** Define prompts and chain them using the Gemini SDK. The orchestrator must pass the ML predictions and RAG documents to the sub-agents and compile the final response.

5. **`deepeval_audit.py`:** Set up DeepEval metrics (Faithfulness, Answer Relevancy, G-Eval) to evaluate the output against ground truth.

### Step 3: API Gateway (`api/main.py`)

1. Initialize FastAPI.

2. Create the `POST /api/v1/advisory/recommend` endpoint to handle the orchestrated workflow.

3. Create the `POST /api/v1/feedback` endpoint to log user interactions to SQLite.

### Step 4: Streamlit UI (`app.py`)

1. Build the sidebar for input parameters (Soil NPK, pH, Rainfall, Temperature, State, Season).

2. Create the main tabs: Advisory (Agent Output), Yield Metrics (XGBoost Output), and Historical Cases (Hybrid Search Output).

3. Connect the UI to the FastAPI endpoints.

## 7. Local Execution & Setup Guide

Ensure you have Python 3.9+ installed.

### Step 1: Install Dependencies

Create a `requirements.txt` file:

```
fastapi>=0.110.0
uvicorn>=0.28.0
pydantic>=2.6.0
pandas>=2.2.0
scikit-learn>=1.4.0
xgboost>=2.0.0
chromadb>=0.4.24
rank-bm25>=0.2.2
sentence-transformers>=2.5.0
google-genai>=0.1.1
streamlit>=1.32.0
deepeval>=1.2.0
shap>=0.45.0=0.1.12`
`langchain-experimental>=0.0.54`
`reportlab>=4.1.0`
`torch>=2.2.0`
`torchvision>=0.17.0`" type="suggestion">


```

Run installation:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


```

### Step 2: Environment Variables

Export your free Gemini API key for the LLM agents:

```
export GEMINI_API_KEY="your_api_key_here"


```

### Step 3: Train the Model & Ingest Documents

Run your training script to generate the `.joblib` models and initialize the ChromaDB collections.

```
python models/train_xgboost.py


```

### Step 4: Run the Microservice (Backend)

Open a terminal and start the FastAPI server:

```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload


```

*You can view the interactive Swagger API documentation at `http://localhost:8000/docs`.*

### Step 5: Run the UI (Frontend)

Open a second terminal window, activate the virtual environment, and run Streamlit:

```
streamlit run app.py


```

*The interactive dashboard will open in your browser at `http://localhost:8501`.*