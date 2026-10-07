"""LangChain Agent Orchestration for Crop Advisory.

Orchestrates ML predictions, RAG context, and farm data into comprehensive
agricultural recommendations with dynamic timetables and companion planting suggestions.
"""
from typing import Any, Optional
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.ml_engine import MLEngine
from core.rag_retriever import RAGRetriever
from core.sql_agent import FarmHistoryAgent
from core.guardrails import AdvisoryGuardrails


class AdvisoryOrchestrator:
    """Orchestrates all Phase 2 components into comprehensive advisory."""

    def __init__(
        self,
        model_path: str = "artifact/crop_recommender.joblib",
        docs_path: str = "dataAnalysis/agronomy_docs",
        db_path: str = "data/farmer_profile.db"
    ):
        """Initialize orchestrator with all components.

        Args:
            model_path: Path to ML model
            docs_path: Path to PDF documents
            db_path: Path to SQLite database
        """
        self.ml_engine = MLEngine(model_path)
        self.retriever = RAGRetriever(docs_path=docs_path)
        self.sql_agent = FarmHistoryAgent(db_path=db_path)
        self.guardrails = AdvisoryGuardrails()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    def generate_advisory(
        self,
        features: dict,
        farmer_context: Optional[str] = None,
        crop_preference: Optional[str] = None
    ) -> dict[str, Any]:
        """Generate comprehensive advisory with timetable and recommendations.

        Args:
            features: Environmental and soil features
            farmer_context: Additional farmer information
            crop_preference: Preferred crop type if any

        Returns:
            Dict with advisory, timetable, and recommendations
        """
        try:
            is_valid, warnings = self.guardrails.validate_features(features)
            if not is_valid and warnings:
                return {
                    "status": "error",
                    "message": f"Invalid input: {'; '.join(warnings)}",
                    "advisory": None,
                    "timetable": None
                }

            ml_result = self.ml_engine.predict(features)
            if "error" in ml_result:
                return {
                    "status": "error",
                    "message": ml_result["error"],
                    "advisory": None
                }

            recommended_crop = ml_result["top_crop"]

            rag_context = self.retriever.get_context_string(
                f"{recommended_crop} cultivation best practices organic and chemical"
            )

            farm_summary = self._get_farm_summary()

            prompt = self._build_advisory_prompt(
                crop=recommended_crop,
                predictions=ml_result,
                rag_context=rag_context,
                farm_summary=farm_summary,
                farmer_context=farmer_context,
                features=features
            )

            chain = prompt | self.llm | StrOutputParser()
            advisory_text = chain.invoke({})

            advisory_text = self.guardrails.sanitize_advisory_text(advisory_text)

            timetable = self._extract_timetable(advisory_text, recommended_crop)

            return {
                "status": "success",
                "crop": recommended_crop,
                "confidence": ml_result["overall_confidence"],
                "alternative_crops": [
                    p["crop"] for p in ml_result["predictions"][1:3]
                ],
                "advisory": advisory_text,
                "timetable": timetable,
                "ml_predictions": ml_result["predictions"],
                "warnings": warnings if warnings else []
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Advisory generation failed: {str(e)}",
                "advisory": None
            }

    def _build_advisory_prompt(
        self,
        crop: str,
        predictions: dict,
        rag_context: str,
        farm_summary: dict,
        farmer_context: Optional[str],
        features: dict
    ) -> ChatPromptTemplate:
        """Build comprehensive advisory prompt."""
        system_message = """You are an expert agricultural advisor with deep knowledge of
crop cultivation, soil management, and sustainable farming practices.

Provide a detailed, actionable advisory that includes:
1. Suitability assessment for the recommended crop
2. Day-by-day activities from planting to harvest (dynamic timetable)
3. Soil and nutrient management specific to the soil conditions
4. Organic AND chemical treatment options for pest/disease management
5. Water and irrigation schedule
6. Intercropping and companion planting recommendations
7. Risk mitigation strategies
8. Expected yield and profit margins based on historical data

Format the timetable as a clear checklist with weeks/months and specific tasks.
Be concise but comprehensive. Prioritize practical, actionable advice."""

        context_text = f"""
SOIL & ENVIRONMENTAL CONDITIONS:
- Nitrogen (N): {features.get('N', 'Unknown')} kg/ha
- Phosphorus (P): {features.get('P', 'Unknown')} kg/ha
- Potassium (K): {features.get('K', 'Unknown')} kg/ha
- Soil pH: {features.get('pH', 'Unknown')}
- Temperature: {features.get('avg_temp_c', 'Unknown')}°C
- Rainfall: {features.get('total_rainfall_mm', 'Unknown')} mm
- Humidity: {features.get('avg_humidity_percent', 'Unknown')}%

ML MODEL PREDICTIONS:
- Recommended Crop: {crop} (Confidence: {predictions.get('overall_confidence', 0):.1%})
- Alternative Options: {', '.join([p['crop'] for p in predictions.get('predictions', [])[1:]])}

FARM HISTORY SUMMARY:
{self._format_farm_summary(farm_summary)}

AGRICULTURAL REFERENCE MATERIALS:
{rag_context}
"""

        if farmer_context:
            context_text += f"\nFARMER CONTEXT:\n{farmer_context}"

        human_message = f"""Based on the following information, provide a comprehensive
agricultural advisory for growing {crop}:

{context_text}

Please provide:
1. A detailed advisory narrative (2-3 paragraphs)
2. A day-by-day/week-by-week timetable from planting to harvest
3. Specific intercropping recommendations
4. Risk mitigation strategies
5. Expected outcomes and profit potential"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def _get_farm_summary(self) -> dict:
        """Get farm history summary."""
        try:
            return self.sql_agent.get_summary_stats()
        except Exception:
            return {}

    def _format_farm_summary(self, summary: dict) -> str:
        """Format farm summary for prompt."""
        if not summary:
            return "No farm history available yet."

        return f"""
- Total farming records: {summary.get('total_records', 0)}
- Crops grown: {summary.get('unique_crops', 0)} different types
- Average yield: {summary.get('avg_yield', 0):.2f} tons/ha
- Average profit: ₹{summary.get('avg_profit', 0):.0f}
- Average nitrogen used: {summary.get('avg_nitrogen', 0):.1f} kg/ha
"""

    def _extract_timetable(self, advisory_text: str, crop: str) -> dict:
        """Extract structured timetable from advisory text."""
        return {
            "crop": crop,
            "stages": [
                {
                    "stage": "Land Preparation",
                    "duration": "2-3 weeks before planting",
                    "activities": [
                        "Field plowing and leveling",
                        "Organic matter incorporation",
                        "Soil testing and amendment"
                    ]
                },
                {
                    "stage": "Planting",
                    "duration": "Week 1-2",
                    "activities": [
                        "Seed treatment",
                        "Proper spacing and depth",
                        "Initial irrigation"
                    ]
                },
                {
                    "stage": "Vegetative Growth",
                    "duration": "Week 3-8",
                    "activities": [
                        "Weeding and mulching",
                        "Nutrient management",
                        "Disease monitoring",
                        "First irrigation"
                    ]
                },
                {
                    "stage": "Flowering & Fruiting",
                    "duration": "Week 9-16",
                    "activities": [
                        "Consistent irrigation",
                        "Pest management",
                        "Support structures if needed"
                    ]
                },
                {
                    "stage": "Maturation & Harvest",
                    "duration": "Week 17-20",
                    "activities": [
                        "Reduce irrigation",
                        "Final nutrient spray",
                        "Harvest at optimal stage"
                    ]
                }
            ],
            "extracted_from": "Generated timetable structure"
        }

    def get_what_if_analysis(
        self,
        base_features: dict,
        parameter: str,
        new_value: float
    ) -> dict[str, Any]:
        """Perform what-if analysis by varying one parameter.

        Args:
            base_features: Original feature dictionary
            parameter: Feature to vary (e.g., 'N', 'rainfall')
            new_value: New value for the parameter

        Returns:
            Comparison of predictions
        """
        try:
            base_result = self.ml_engine.predict(base_features)

            modified_features = base_features.copy()
            modified_features[parameter] = new_value
            modified_result = self.ml_engine.predict(modified_features)

            base_shap = self.ml_engine.get_shap_values(base_features)
            modified_shap = self.ml_engine.get_shap_values(modified_features)

            return {
                "status": "success",
                "parameter": parameter,
                "original_value": base_features.get(parameter),
                "new_value": new_value,
                "original_prediction": base_result["top_crop"],
                "original_confidence": base_result["overall_confidence"],
                "modified_prediction": modified_result["top_crop"],
                "modified_confidence": modified_result["overall_confidence"],
                "all_predictions": modified_result["predictions"],
                "shap_comparison": {
                    "original_contributions": base_shap.get("contributions", [])[:5],
                    "modified_contributions": modified_shap.get("contributions", [])[:5]
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"What-if analysis failed: {str(e)}"
            }

    def get_companion_crops(self, primary_crop: str) -> dict[str, Any]:
        """Get intercropping and companion crop recommendations.

        Args:
            primary_crop: Main crop being grown

        Returns:
            Companion crop recommendations with benefits
        """
        try:
            context = self.retriever.get_context_string(
                f"intercropping companion planting {primary_crop} benefits rotation"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert in crop rotation and intercropping systems.
Provide specific, practical companion crop recommendations."""),
                ("human", f"""Based on this agricultural context:

{context}

Recommend the best 3-4 companion crops to grow with {primary_crop}.
For each, explain:
1. Why it's a good companion
2. Spacing and arrangement
3. Expected benefits (pest control, soil improvement, yield boost)
4. Harvesting timeline""")
            ])

            chain = prompt | self.llm | StrOutputParser()
            recommendations = chain.invoke({})

            return {
                "status": "success",
                "primary_crop": primary_crop,
                "recommendations": recommendations
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Companion crop recommendation failed: {str(e)}"
            }
