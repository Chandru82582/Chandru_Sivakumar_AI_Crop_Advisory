"""Test script for Phase 2 modules.

Tests the SQL agent, RAG retriever, and ML engine.
Run with: python test_phase2.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def test_ml_engine():
    """Test ML engine predictions."""
    print("\n" + "="*60)
    print("Testing ML Engine")
    print("="*60)

    from core.ml_engine import MLEngine

    engine = MLEngine(model_path="artifact/crop_recommender.joblib")

    test_features = {
        "N": 50,
        "P": 20,
        "K": 30,
        "pH": 6.5,
        "avg_temp_c": 25.0,
        "total_rainfall_mm": 800.0,
        "avg_humidity_percent": 65.0,
        "season_Kharif": 1,
        "season_Rabi": 0
    }

    result = engine.predict(test_features)
    print(f"Prediction Result:")
    print(f"  Top Crop: {result['top_crop']}")
    print(f"  Overall Confidence: {result['overall_confidence']:.2%}")
    print(f"  All Predictions:")
    for pred in result['predictions']:
        print(f"    - {pred['crop']}: {pred['confidence_pct']:.1f}%")

    model_info = engine.get_model_info()
    print(f"\nModel Info:")
    print(f"  Model Name: {model_info['model_name']}")
    print(f"  Number of Classes: {model_info['n_classes']}")
    print(f"  Number of Features: {model_info['n_features']}")
    print(f"  Created At: {model_info['created_at']}")

    return engine


def test_rag_retriever():
    """Test RAG retriever."""
    print("\n" + "="*60)
    print("Testing RAG Retriever")
    print("="*60)

    from core.rag_retriever import RAGRetriever

    retriever = RAGRetriever(
        docs_path="dataAnalysis/agronomy_docs",
        chroma_path="data/chroma_db"
    )

    test_query = "What is the best fertilizer for wheat?"
    print(f"\nQuery: {test_query}\n")

    results = retriever.retrieve(test_query, k=3)
    print(f"Retrieved {len(results)} documents:\n")

    for i, doc in enumerate(results, 1):
        source = doc["source"].split("\\")[-1] if "\\" in doc["source"] else doc["source"]
        content_preview = doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
        print(f"Document {i}: {source}")
        print(f"Content: {content_preview}\n")

    return retriever


def test_sql_agent():
    """Test SQL agent."""
    print("\n" + "="*60)
    print("Testing SQL Agent")
    print("="*60)

    from core.sql_agent import FarmHistoryAgent

    agent = FarmHistoryAgent(db_path="data/farmer_profile.db")

    print("Adding sample farm records...")

    records = [
        {
            "season": "Kharif",
            "crop_planted": "Rice",
            "yield_achieved": 5.2,
            "nitrogen_used": 80,
            "profit": 45000
        },
        {
            "season": "Rabi",
            "crop_planted": "Wheat",
            "yield_achieved": 4.8,
            "nitrogen_used": 100,
            "profit": 52000
        },
        {
            "season": "Kharif",
            "crop_planted": "Corn",
            "yield_achieved": 6.1,
            "nitrogen_used": 90,
            "profit": 48000
        }
    ]

    for record in records:
        result = agent.add_farm_record(**record)
        print(f"  Added record ID: {result['id']}")

    print("\nGetting summary statistics...")
    stats = agent.get_summary_stats()
    print(f"  Total Records: {stats['total_records']}")
    print(f"  Unique Crops: {stats['unique_crops']}")
    print(f"  Unique Seasons: {stats['unique_seasons']}")
    print(f"  Average Yield: {stats['avg_yield']:.2f}")
    print(f"  Average Profit: ₹{stats['avg_profit']:.0f}")
    print(f"  Average Nitrogen Used: {stats['avg_nitrogen']:.2f} kg/ha")

    return agent


def test_guardrails():
    """Test safety guardrails."""
    print("\n" + "="*60)
    print("Testing Guardrails")
    print("="*60)

    from core.guardrails import AdvisoryGuardrails

    test_features = {
        "N": 50,
        "P": 20,
        "K": 30,
        "pH": 6.5,
        "avg_temp_c": 25.0,
        "total_rainfall_mm": 800.0
    }

    is_valid, warnings = AdvisoryGuardrails.validate_features(test_features)
    print(f"Validation Result: {'Valid' if is_valid else 'Invalid'}")
    print(f"Warnings: {warnings if warnings else 'None'}")

    test_text = "This fertilizer is toxic and should be banned"
    sanitized = AdvisoryGuardrails.sanitize_advisory_text(test_text)
    print(f"\nOriginal: {test_text}")
    print(f"Sanitized: {sanitized}")


def test_plant_doctor():
    """Test plant doctor."""
    print("\n" + "="*60)
    print("Testing Plant Doctor")
    print("="*60)

    from core.plant_doctor import PlantDoctor

    doctor = PlantDoctor(use_pretrained=False)

    diagnosis = doctor._mock_diagnosis("test_leaf.jpg")
    print(f"Disease: {diagnosis['disease']}")
    print(f"Confidence: {diagnosis['confidence']:.2%}")
    print(f"Recommendations:")
    for rec in diagnosis['recommendations']:
        print(f"  - {rec}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Phase 2 Module Testing")
    print("="*60)

    try:
        print("\n[1] Testing ML Engine...")
        ml_engine = test_ml_engine()
    except Exception as e:
        print(f"❌ ML Engine Error: {e}")

    try:
        print("\n[2] Testing RAG Retriever...")
        rag_retriever = test_rag_retriever()
    except Exception as e:
        print(f"❌ RAG Retriever Error: {e}")

    try:
        print("\n[3] Testing SQL Agent...")
        sql_agent = test_sql_agent()
    except Exception as e:
        print(f"❌ SQL Agent Error: {e}")

    try:
        print("\n[4] Testing Guardrails...")
        test_guardrails()
    except Exception as e:
        print(f"❌ Guardrails Error: {e}")

    try:
        print("\n[5] Testing Plant Doctor...")
        test_plant_doctor()
    except Exception as e:
        print(f"❌ Plant Doctor Error: {e}")

    print("\n" + "="*60)
    print("Phase 2 Testing Complete!")
    print("="*60 + "\n")
