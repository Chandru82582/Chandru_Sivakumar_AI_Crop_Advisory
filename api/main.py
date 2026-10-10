"""FastAPI Gateway for Crop Advisory System.

Exposes agricultural advisory, ML predictions, leaf diagnosis, and RAG retrieval.
Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crop Advisory AI Platform API",
    description="REST API Gateway for Precision Agriculture, Crop Recommendations, Leaf Vision Diagnosis, and RAG Retrieval.",
    version="3.6.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS for browser Swagger UI and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Granular lazy singletons for high performance and fast sub-second startup
_ML_ENGINE = None
_GUARDRAILS = None
_SQL_AGENT = None
_DOCTOR = None
_RETRIEVER = None
_ORCHESTRATOR = None


def get_ml_engine():
    global _ML_ENGINE
    if _ML_ENGINE is None:
        from core.ml_engine import MLEngine
        _ML_ENGINE = MLEngine()
    return _ML_ENGINE


def get_guardrails():
    global _GUARDRAILS
    if _GUARDRAILS is None:
        from core.guardrails import AdvisoryGuardrails
        _GUARDRAILS = AdvisoryGuardrails()
    return _GUARDRAILS


def get_sql_agent():
    global _SQL_AGENT
    if _SQL_AGENT is None:
        from core.sql_agent import FarmHistoryAgent
        _SQL_AGENT = FarmHistoryAgent()
    return _SQL_AGENT


def get_doctor():
    global _DOCTOR
    if _DOCTOR is None:
        from core.plant_doctor import PlantDoctor
        _DOCTOR = PlantDoctor(use_pretrained=False)
    return _DOCTOR


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        from core.rag_retriever import RAGRetriever
        _RETRIEVER = RAGRetriever()
    return _RETRIEVER


def get_orchestrator():
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        from core.agents import AdvisoryOrchestrator
        _ORCHESTRATOR = AdvisoryOrchestrator(
            ml_engine=get_ml_engine(),
            retriever=get_retriever(),
            sql_agent=get_sql_agent()
        )
    return _ORCHESTRATOR


# =========================================================================
# WEB DASHBOARD & ROOT ROUTE
# =========================================================================
@app.get("/", response_class=HTMLResponse)
async def api_root():
    """Web landing page linking directly to interactive Swagger docs."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Crop Advisory AI API Gateway</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #F8FAF8; color: #1E293B; margin: 0; padding: 40px 20px; }
            .card { max-width: 760px; margin: 0 auto; background: white; padding: 36px; border-radius: 14px; border: 1px solid #CBD5E1; box-shadow: 0 4px 20px rgba(0,0,0,0.06); }
            h1 { color: #1B4D3E; margin-top: 0; font-size: 1.8rem; }
            p { font-size: 1rem; line-height: 1.6; }
            .btn { display: inline-block; background: #1B4D3E; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; margin-top: 14px; margin-right: 12px; }
            .btn:hover { background: #2D6A4F; }
            .btn-secondary { background: #2D6A4F; }
            .badge { display: inline-block; background: #E8F5E9; color: #166534; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }
            ul { line-height: 1.8; }
            code { background: #F1F5F9; padding: 3px 8px; border-radius: 4px; color: #0F172A; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <div class="card">
            <span class="badge">API v3.6 Active</span>
            <h1>🌾 AgriAdvisory AI Platform API Gateway</h1>
            <p>Welcome to the Agricultural Advisory Decision Support REST API. All endpoints are fully operational.</p>
            <div>
                <a href="/docs" class="btn">📖 Open Swagger Documentation (/docs)</a>
                <a href="/redoc" class="btn btn-secondary">📑 Open ReDoc Specification (/redoc)</a>
            </div>
            <hr style="margin: 28px 0; border: none; border-top: 1px solid #E2E8F0;">
            <h3 style="color: #1B4D3E;">Featured Endpoints:</h3>
            <ul>
                <li><code>GET /health</code> — System health check</li>
                <li><code>POST /api/v1/recommend</code> — Comprehensive ML recommendation & advisory</li>
                <li><code>POST /api/v1/simulate</code> — What-If environmental factor sensitivity simulator</li>
                <li><code>POST /api/v1/predict</code> — Fast ML crop recommendation only</li>
                <li><code>POST /api/v1/diagnose</code> — Computer vision leaf pathology diagnosis</li>
                <li><code>POST /api/v1/history</code> — Natural language farm history query</li>
                <li><code>GET /api/v1/history/stats</code> — Aggregate farm statistics</li>
                <li><code>POST /api/v1/export_pdf</code> — Generate publication-grade PDF report</li>
            </ul>
        </div>
    </body>
    </html>
    """


# =========================================================================
# SCHEMAS
# =========================================================================
class FeaturesInput(BaseModel):
    """Input features for predictions."""
    N: float
    P: float
    K: float
    pH: float
    avg_temp_c: float
    total_rainfall_mm: float
    avg_humidity_percent: float
    season_Kharif: int = 1
    season_Rabi: int = 0


class AdvisoryRequest(BaseModel):
    """Request for comprehensive advisory."""
    features: FeaturesInput
    farmer_name: Optional[str] = "Murugan (Farmer, Tamil Nadu)"
    location: Optional[str] = "Thanjavur, Tamil Nadu"
    farmer_context: Optional[str] = None
    crop_preference: Optional[str] = None


class SimulateRequest(BaseModel):
    """Request for what-if analysis."""
    features: FeaturesInput
    parameter: str
    new_value: float


class HistoryQuery(BaseModel):
    """Farm history query."""
    query: str


class FarmRecordInput(BaseModel):
    """Farm record input."""
    season: str
    crop_planted: str
    yield_achieved: Optional[float] = None
    nitrogen_used: Optional[float] = None
    profit: Optional[float] = None


# =========================================================================
# REST ENDPOINTS
# =========================================================================
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Crop Advisory API",
        "version": "3.6.0"
    }


@app.post("/api/v1/recommend")
async def recommend(request: AdvisoryRequest):
    """Generate crop recommendation with comprehensive advisory."""
    try:
        orch = get_orchestrator()
        features_dict = request.features.dict()

        result = orch.generate_advisory(
            features=features_dict,
            farmer_context=request.farmer_context or f"Farmer: {request.farmer_name}, Location: {request.location}",
            crop_preference=request.crop_preference
        )
        return result
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/simulate")
async def simulate(request: SimulateRequest):
    """Perform what-if analysis with SHAP values."""
    try:
        orch = get_orchestrator()
        features_dict = request.features.dict()

        result = orch.get_what_if_analysis(
            base_features=features_dict,
            parameter=request.parameter,
            new_value=request.new_value
        )
        return result
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/history")
async def farm_history(query: HistoryQuery):
    """Query farm history using natural language."""
    try:
        sql = get_sql_agent()
        result = sql.ask_farm_history(query.query)
        return result
    except Exception as e:
        logger.error(f"History query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/history/add")
async def add_farm_record(record: FarmRecordInput):
    """Add a new farm record to history."""
    try:
        sql = get_sql_agent()
        result = sql.add_farm_record(
            season=record.season,
            crop_planted=record.crop_planted,
            yield_achieved=record.yield_achieved,
            nitrogen_used=record.nitrogen_used,
            profit=record.profit
        )
        return {
            "status": "success",
            "record_id": result["id"],
            "message": "Farm record added successfully"
        }
    except Exception as e:
        logger.error(f"Record insertion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/history/stats")
async def farm_statistics():
    """Get farm history statistics."""
    try:
        sql = get_sql_agent()
        stats = sql.get_summary_stats()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrieve")
async def retrieve_documents(query: str, k: int = 5):
    """Retrieve agricultural documents from knowledge base."""
    try:
        ret = get_retriever()
        docs = ret.retrieve(query, k=k)
        return {
            "status": "success",
            "query": query,
            "count": len(docs),
            "documents": docs
        }
    except Exception as e:
        logger.error(f"Document retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict")
async def predict(features: FeaturesInput):
    """Get fast crop predictions from ML model."""
    try:
        ml_eng = get_ml_engine()
        grd = get_guardrails()
        features_dict = features.dict()

        is_valid, warnings = grd.validate_features(features_dict)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid features: {'; '.join(warnings)}"
            )

        result = ml_eng.predict(features_dict)
        return {
            "status": "success",
            "predictions": result["predictions"],
            "top_crop": result["top_crop"],
            "confidence": result["overall_confidence"],
            "warnings": result.get("warnings", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/export_pdf")
async def export_pdf(request: AdvisoryRequest):
    """Export comprehensive advisory as PDF."""
    try:
        from core.pdf_builder import PDFReportBuilder
        orch = get_orchestrator()
        ml_eng = get_ml_engine()
        features_dict = request.features.dict()

        advisory_result = orch.generate_advisory(
            features=features_dict,
            farmer_context=request.farmer_context,
            crop_preference=request.crop_preference
        )

        if advisory_result["status"] == "error":
            raise HTTPException(
                status_code=400,
                detail=advisory_result["message"]
            )

        ml_result = ml_eng.predict(features_dict)

        builder = PDFReportBuilder()
        pdf_buffer = builder.generate_advisory_pdf(
            farmer_name=request.farmer_name,
            location=request.location,
            advisory_data=advisory_result,
            ml_metrics=ml_result
        )

        return StreamingResponse(
            iter([pdf_buffer.getvalue()]),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=crop_advisory.pdf"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/companion_crops")
async def get_companion_crops(crop: str):
    """Get companion crop recommendations for intercropping."""
    try:
        orch = get_orchestrator()
        result = orch.get_companion_crops(crop)
        return result
    except Exception as e:
        logger.error(f"Companion crop query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/info")
async def model_info():
    """Get ML model information."""
    try:
        ml_eng = get_ml_engine()
        info = ml_eng.get_model_info()
        return {
            "status": "success",
            "model_info": info
        }
    except Exception as e:
        logger.error(f"Model info retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/validate")
async def validate_input(features: FeaturesInput):
    """Validate input features for data quality."""
    try:
        grd = get_guardrails()
        features_dict = features.dict()
        is_valid, warnings = grd.validate_features(features_dict)

        return {
            "status": "success",
            "is_valid": is_valid,
            "warnings": warnings,
            "features_count": len(features_dict)
        }
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/diagnose")
async def diagnose_leaf(file: UploadFile = File(...)):
    """Diagnose crop disease from uploaded leaf image."""
    try:
        doc = get_doctor()
        content = await file.read()
        res = doc.diagnose(content)
        return {
            "status": "success",
            "filename": file.filename,
            "diagnosis": res
        }
    except Exception as e:
        logger.error(f"Diagnosis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
