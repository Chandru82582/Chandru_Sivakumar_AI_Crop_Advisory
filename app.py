"""Streamlit UI for Crop Advisory System.

Interactive dashboard with 4 main tabs:
1. Advisory & What-If Simulator
2. Smart Operations & Timetable
3. Crop Doctor
4. Farm History

Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Optional
import requests
import json

from core.ml_engine import MLEngine
from core.rag_retriever import RAGRetriever
from core.sql_agent import FarmHistoryAgent
from core.agents import AdvisoryOrchestrator
from core.pdf_builder import PDFReportBuilder

st.set_page_config(
    page_title="Crop Advisory System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_components():
    """Load and cache all components."""
    return {
        "orchestrator": AdvisoryOrchestrator(),
        "ml_engine": MLEngine(),
        "retriever": RAGRetriever(),
        "sql_agent": FarmHistoryAgent(),
        "pdf_builder": PDFReportBuilder()
    }

def main():
    st.title("🌾 AI-Powered Crop Advisory Assistant")
    st.markdown("---")

    components = load_components()

    with st.sidebar:
        st.header("📊 Input Parameters")

        col1, col2 = st.columns(2)
        with col1:
            nitrogen = st.slider("Nitrogen (N)", 0, 200, 50, step=5)
            phosphorus = st.slider("Phosphorus (P)", 0, 100, 20, step=2)
            potassium = st.slider("Potassium (K)", 0, 100, 30, step=2)

        with col2:
            ph = st.slider("Soil pH", 0.0, 14.0, 6.5, step=0.1)
            temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, step=1.0)
            rainfall = st.slider("Rainfall (mm)", 0, 3000, 800, step=50)

        humidity = st.slider("Humidity (%)", 0, 100, 65, step=5)

        st.divider()
        st.header("👤 Farmer Information")
        farmer_name = st.text_input("Farmer Name (optional)")
        location = st.text_input("Farm Location (optional)")

        st.divider()

        season = st.radio("Select Season", ["Kharif", "Rabi"])
        season_kharif = 1 if season == "Kharif" else 0
        season_rabi = 1 if season == "Rabi" else 0

        features = {
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "pH": ph,
            "avg_temp_c": temp,
            "total_rainfall_mm": rainfall,
            "avg_humidity_percent": humidity,
            "season_Kharif": season_kharif,
            "season_Rabi": season_rabi
        }

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Advisory & What-If",
        "📅 Smart Operations",
        "🔍 Crop Doctor",
        "📚 Farm History"
    ])

    with tab1:
        st.header("Crop Advisory & What-If Simulator")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Generate Advisory", key="gen_advisory"):
                with st.spinner("Generating comprehensive advisory..."):
                    result = components["orchestrator"].generate_advisory(
                        features=features,
                        farmer_context=None,
                        crop_preference=None
                    )

                    if result["status"] == "success":
                        st.success("✅ Advisory Generated!")

                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        with col_metric1:
                            st.metric("Recommended Crop", result["crop"])
                        with col_metric2:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        with col_metric3:
                            st.metric("Alternatives", ", ".join(result.get("alternative_crops", [])))

                        with st.expander("📄 Full Advisory", expanded=True):
                            st.write(result["advisory"])

                        with st.expander("📋 Timetable"):
                            if result.get("timetable"):
                                for stage in result["timetable"].get("stages", []):
                                    st.write(f"**{stage['stage']}** ({stage['duration']})")
                                    for activity in stage.get("activities", []):
                                        st.write(f"  • {activity}")

                        col_down1, col_down2 = st.columns(2)
                        with col_down1:
                            if st.button("📥 Download PDF Report"):
                                builder = components["pdf_builder"]
                                pdf_buffer = builder.generate_advisory_pdf(
                                    farmer_name=farmer_name,
                                    location=location,
                                    advisory_data=result,
                                    ml_metrics={"predictions": result.get("ml_predictions", [])}
                                )
                                st.download_button(
                                    label="📥 Download PDF",
                                    data=pdf_buffer.getvalue(),
                                    file_name="crop_advisory.pdf",
                                    mime="application/pdf"
                                )
                    else:
                        st.error(f"❌ {result.get('message', 'Advisory generation failed')}")

        with col2:
            st.subheader("What-If Analysis")
            param = st.selectbox(
                "Select Parameter to Vary",
                ["N", "P", "K", "pH", "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent"]
            )

            param_labels = {
                "N": (0, 200, 5),
                "P": (0, 100, 2),
                "K": (0, 100, 2),
                "pH": (0.0, 14.0, 0.1),
                "avg_temp_c": (-10.0, 50.0, 1.0),
                "total_rainfall_mm": (0, 3000, 50),
                "avg_humidity_percent": (0, 100, 5)
            }

            min_val, max_val, step = param_labels.get(param, (0, 100, 1))
            new_value = st.slider(f"New {param} Value", min_val, max_val, features[param], step)

            if st.button("🔄 Run What-If Analysis"):
                with st.spinner("Running what-if analysis..."):
                    result = components["orchestrator"].get_what_if_analysis(
                        base_features=features,
                        parameter=param,
                        new_value=new_value
                    )

                    if result["status"] == "success":
                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            st.metric("Original Crop", result["original_prediction"])
                            st.metric("Original Confidence", f"{result['original_confidence']:.1%}")
                        with col_w2:
                            st.metric("New Crop", result["modified_prediction"])
                            st.metric("New Confidence", f"{result['modified_confidence']:.1%}")

                        st.write("**Modified Predictions:**")
                        pred_df = pd.DataFrame([
                            {"Crop": p["crop"], "Probability": f"{p['probability']:.3f}"}
                            for p in result.get("all_predictions", [])
                        ])
                        st.dataframe(pred_df, use_container_width=True)

    with tab2:
        st.header("Smart Operations & Timetable")

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("📅 Generate Timetable"):
                with st.spinner("Generating cultivation timetable..."):
                    result = components["orchestrator"].generate_advisory(
                        features=features
                    )

                    if result["status"] == "success":
                        st.success("✅ Timetable Generated!")

                        for stage in result["timetable"]["stages"]:
                            with st.expander(f"📍 {stage['stage']} - {stage['duration']}", expanded=True):
                                for i, activity in enumerate(stage["activities"], 1):
                                    st.write(f"{i}. {activity}")

        with col2:
            treatment = st.radio("Treatment", ["Organic", "Chemical"])

        st.divider()

        st.subheader("Companion Crops & Intercropping")
        if st.button("🌱 Get Recommendations"):
            with st.spinner("Fetching companion crop recommendations..."):
                ml_result = components["ml_engine"].predict(features)
                crop = ml_result["top_crop"]

                comp_result = components["orchestrator"].get_companion_crops(crop)

                if comp_result["status"] == "success":
                    st.write(comp_result["recommendations"])

    with tab3:
        st.header("Crop Doctor - Disease Detection")

        st.info("📸 Upload a crop leaf image for disease diagnosis")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            with col2:
                if st.button("🔍 Analyze"):
                    st.info("🚀 Disease detection capability would require trained vision model. "
                           "Currently showing mock results.")

                    st.success("Disease: Healthy Leaf")
                    st.metric("Confidence", "85%")

                    st.subheader("Recommendations:")
                    recommendations = [
                        "Continue regular monitoring",
                        "Maintain proper watering schedule",
                        "Monitor weather for disease-promoting conditions",
                        "Apply preventive fungicide if humid conditions persist"
                    ]
                    for rec in recommendations:
                        st.write(f"• {rec}")

    with tab4:
        st.header("Farm History & Analytics")

        tab_history1, tab_history2 = st.tabs(["📊 Statistics", "💬 Q&A"])

        with tab_history1:
            if st.button("📈 Load Farm Statistics"):
                with st.spinner("Loading farm statistics..."):
                    stats = components["sql_agent"].get_summary_stats()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", stats.get("total_records", 0))
                        st.metric("Unique Crops", stats.get("unique_crops", 0))
                    with col2:
                        st.metric("Avg Yield", f"{stats.get('avg_yield', 0):.2f} t/ha")
                        st.metric("Avg Profit", f"₹{stats.get('avg_profit', 0):.0f}")
                    with col3:
                        st.metric("Avg Nitrogen", f"{stats.get('avg_nitrogen', 0):.2f} kg/ha")
                        st.metric("Unique Seasons", stats.get("unique_seasons", 0))

        with tab_history2:
            st.subheader("Ask Your Farm Questions")

            question = st.text_area(
                "What would you like to know about your farm?",
                placeholder="e.g., What was my rice yield in Kharif? Which crop gave highest profit?"
            )

            if st.button("🔍 Query"):
                if question:
                    with st.spinner("Processing your question..."):
                        result = components["sql_agent"].ask_farm_history(question)

                        if "error" not in result:
                            st.success("✅ Query Result:")
                            st.write(result.get("summary", "No summary available"))

                            if result.get("results"):
                                st.write("**Raw Results:**")
                                st.dataframe(pd.DataFrame(result["results"]))
                        else:
                            st.error(f"Error: {result.get('error', 'Query failed')}")

            st.divider()
            st.subheader("Add New Farm Record")

            col1, col2 = st.columns(2)
            with col1:
                rec_season = st.selectbox("Season", ["Kharif", "Rabi"], key="rec_season")
                rec_crop = st.text_input("Crop Planted", key="rec_crop")
                rec_yield = st.number_input("Yield (tons/ha)", min_value=0.0, step=0.1)

            with col2:
                rec_nitrogen = st.number_input("Nitrogen Used (kg/ha)", min_value=0.0, step=1.0)
                rec_profit = st.number_input("Profit (₹)", min_value=0.0, step=1000.0)

                if st.button("➕ Add Record"):
                    with st.spinner("Adding record..."):
                        result = components["sql_agent"].add_farm_record(
                            season=rec_season,
                            crop_planted=rec_crop,
                            yield_achieved=rec_yield,
                            nitrogen_used=rec_nitrogen,
                            profit=rec_profit
                        )

                        if result["status"] == "success":
                            st.success(f"✅ Record #{result['id']} added successfully!")
                        else:
                            st.error("Failed to add record")

    st.divider()
    st.markdown("""
    ---
    **Crop Advisory System** | Phase 1-5 Complete

    Powered by: LangChain • Gemini API • XGBoost • ChromaDB • Streamlit
    """)

if __name__ == "__main__":
    main()
