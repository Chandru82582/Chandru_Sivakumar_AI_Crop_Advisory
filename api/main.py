"""FastAPI Gateway for Crop Advisory System.

Exposes all Phase 2-3 functionality through REST endpoints.
Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import logging
from typing import Optional
from pydantic import BaseModel

from core.agents import AdvisoryOrchestrator
from core.ml_engine import MLEngine
from core.rag_retriever import RAGRetriever
from core.sql_agent import FarmHistoryAgent
from core.guardrails import AdvisoryGuardrails

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crop Advisory API",
    description="AI-powered agricultural advisory system",
    version="1.0.0"
)

orchestrator = AdvisoryOrchestrator()
ml_engine = MLEngine()
retriever = RAGRetriever()
sql_agent = FarmHistoryAgent()
guardrails = AdvisoryGuardrails()


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
    farmer_name: Optional[str] = None
    location: Optional[str] = None
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Crop Advisory API",
        "version": "1.0.0"
    }


@app.post("/api/v1/recommend")
async def recommend(request: AdvisoryRequest):
    """Generate crop recommendation with advisory.

    Combines ML predictions, RAG context, and farm history into
    comprehensive recommendation with timetable.
    """
    try:
        features_dict = request.features.dict()

        result = orchestrator.generate_advisory(
            features=features_dict,
            farmer_context=request.farmer_context,
            crop_preference=request.crop_preference
        )

        return result

    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/simulate")
async def simulate(request: SimulateRequest):
    """Perform what-if analysis with SHAP values.

    Shows how changing one parameter affects crop recommendations
    and displays feature importance changes.
    """
    try:
        features_dict = request.features.dict()

        result = orchestrator.get_what_if_analysis(
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
    """Query farm history using natural language.

    Translates natural language to SQL and returns results
    with automatic summarization.
    """
    try:
        result = sql_agent.ask_farm_history(query.query)
        return result

    except Exception as e:
        logger.error(f"History query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/history/add")
async def add_farm_record(record: FarmRecordInput):
    """Add a new farm record to history."""
    try:
        result = sql_agent.add_farm_record(
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
        stats = sql_agent.get_summary_stats()
        return {
            "status": "success",
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Statistics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrieve")
async def retrieve_documents(query: str, k: int = 5):
    """Retrieve agricultural documents from knowledge base.

    Uses hybrid RAG (BM25 + embeddings) for document retrieval.
    """
    try:
        docs = retriever.retrieve(query, k=k)
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
    """Get crop predictions from ML model.

    Returns top 3 crop recommendations with confidence scores.
    """
    try:
        features_dict = features.dict()

        is_valid, warnings = guardrails.validate_features(features_dict)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid features: {'; '.join(warnings)}"
            )

        result = ml_engine.predict(features_dict)

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
    """Export comprehensive advisory as PDF.

    Generates a professional PDF report with recommendations,
    timetable, and metrics.
    """
    try:
        from core.pdf_builder import PDFReportBuilder

        features_dict = request.features.dict()

        advisory_result = orchestrator.generate_advisory(
            features=features_dict,
            farmer_context=request.farmer_context,
            crop_preference=request.crop_preference
        )

        if advisory_result["status"] == "error":
            raise HTTPException(
                status_code=400,
                detail=advisory_result["message"]
            )

        ml_result = ml_engine.predict(features_dict)

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
    """Get companion crop recommendations for intercropping.

    Uses RAG to retrieve best intercropping combinations.
    """
    try:
        result = orchestrator.get_companion_crops(crop)
        return result

    except Exception as e:
        logger.error(f"Companion crop query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/info")
async def model_info():
    """Get ML model information."""
    try:
        info = ml_engine.get_model_info()
        return {
            "status": "success",
            "model_info": info
        }

    except Exception as e:
        logger.error(f"Model info retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/validate")
async def validate_input(features: FeaturesInput):
    """Validate input features for data quality.

    Checks feature ranges and data quality issues.
    """
    try:
        features_dict = features.dict()
        is_valid, warnings = guardrails.validate_features(features_dict)

        return {
            "status": "success",
            "is_valid": is_valid,
            "warnings": warnings,
            "features_count": len(features_dict)
        }

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
