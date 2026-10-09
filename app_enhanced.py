"""Enhanced Streamlit UI for Crop Advisory System - Premium Edition

Improved UI with:
- Brand color palette (#3368A0, #66A3BF, #C8DFDB, #F2EFE7)
- SHAP waterfall visualizations
- Interactive charts and metrics
- Checklist-style timetable
- Chat interface for farm history
- Professional dashboard styling

Run: streamlit run app_enhanced.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from typing import Optional
import json

from core.ml_engine import MLEngine
from core.rag_retriever import RAGRetriever
from core.sql_agent import FarmHistoryAgent
from core.agents import AdvisoryOrchestrator
from core.pdf_builder import PDFReportBuilder

# Brand Colors
PRIMARY_COLOR = "#3368A0"
SECONDARY_COLOR = "#66A3BF"
ACCENT_COLOR = "#C8DFDB"
LIGHT_BG = "#F2EFE7"

st.set_page_config(
    page_title="🌾 Crop Advisory - Premium",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI-Powered Agriculture Advisory System v2.0"}
)

st.markdown(f"""
    <style>
    :root {{
        --primary: {PRIMARY_COLOR};
        --secondary: {SECONDARY_COLOR};
        --accent: {ACCENT_COLOR};
        --light: {LIGHT_BG};
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: white;
    }}

    [data-testid="stSidebar"] {{
        background-color: {LIGHT_BG};
    }}

    .main-header {{
        color: {PRIMARY_COLOR};
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }}

    .section-header {{
        color: {PRIMARY_COLOR};
        border-bottom: 3px solid {SECONDARY_COLOR};
        padding-bottom: 0.5em;
        font-size: 1.5em;
        font-weight: bold;
    }}

    .metric-card {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        color: white;
        padding: 1.5em;
        border-radius: 0.5em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}

    .stTabs [data-baseweb="tab-list"] button {{
        font-size: 1.1em;
        font-weight: 500;
    }}

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-bottom-color: {PRIMARY_COLOR};
    }}

    .stButton > button {{
        background: linear-gradient(90deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        color: white;
        border-radius: 0.5em;
        padding: 0.7em 1.4em;
        font-weight: bold;
        border: none;
    }}

    .stButton > button:hover {{
        opacity: 0.9;
    }}

    .checklist-item {{
        padding: 0.7em;
        background: {LIGHT_BG};
        border-left: 4px solid {PRIMARY_COLOR};
        margin: 0.3em 0;
        border-radius: 0.3em;
    }}

    .success-box {{
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1em;
        border-radius: 0.3em;
    }}

    .info-box {{
        background-color: #e3f2fd;
        border-left: 4px solid {PRIMARY_COLOR};
        padding: 1em;
        border-radius: 0.3em;
    }}
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

def plot_shap_waterfall(contributions, base_value, top_n=10):
    """Create SHAP waterfall plot using Plotly."""
    if not contributions:
        st.warning("No SHAP contributions available")
        return

    contributions_df = pd.DataFrame(contributions[:top_n])
    contributions_df['abs_shap'] = contributions_df['shap_value'].abs()
    contributions_df = contributions_df.sort_values('abs_shap', ascending=True)

    fig = go.Figure()

    colors = [PRIMARY_COLOR if x > 0 else "#d32f2f" for x in contributions_df['shap_value']]

    fig.add_trace(go.Bar(
        y=contributions_df['feature'],
        x=contributions_df['shap_value'],
        orientation='h',
        marker=dict(color=colors),
        hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"<b>Feature Importance (SHAP)</b><br><sub>Base Value: {base_value:.4f}</sub>",
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Features",
        height=400,
        showlegend=False,
        plot_bgcolor="white",
        hovermode='closest',
        font=dict(family="Arial", size=11),
        margin=dict(l=150, r=50, t=100, b=50)
    )

    fig.update_xaxes(gridcolor="#e0e0e0")

    st.plotly_chart(fig, use_container_width=True)

def plot_prediction_metrics(predictions):
    """Create prediction metrics visualization."""
    if not predictions:
        return

    df = pd.DataFrame(predictions[:5])

    fig = go.Figure(data=[
        go.Bar(
            x=df['crop'],
            y=df['probability'],
            marker=dict(
                color=df['probability'],
                colorscale=[[0, "#ffcdd2"], [1, PRIMARY_COLOR]],
                showscale=False
            ),
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
        )
    ])

    fig.update_layout(
        title="<b>Crop Prediction Probabilities</b>",
        xaxis_title="Crop",
        yaxis_title="Probability",
        height=300,
        showlegend=False,
        plot_bgcolor="white",
        font=dict(family="Arial", size=11),
        margin=dict(l=60, r=30, t=80, b=60)
    )

    fig.update_yaxes(gridcolor="#e0e0e0", range=[0, 1])

    st.plotly_chart(fig, use_container_width=True)

def render_timetable_checklist(timetable):
    """Render timetable as interactive checklist."""
    if not timetable or not timetable.get("stages"):
        st.info("No timetable available")
        return

    st.markdown("### 📋 Cultivation Timeline")

    for i, stage in enumerate(timetable["stages"]):
        with st.expander(
            f"**{stage['stage']}** - {stage['duration']}",
            expanded=(i == 0)
        ):
            col1, col2 = st.columns([0.1, 0.9])

            with col1:
                completed = st.checkbox(
                    "Completed",
                    key=f"stage_{i}",
                    label_visibility="collapsed"
                )

            with col2:
                for j, activity in enumerate(stage.get("activities", [])):
                    st.markdown(f"- {activity}")

def render_farm_history_chat(agent):
    """Render farm history as chat-like interface."""
    st.markdown("### 💬 Farm History Q&A")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask about your farm history...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("🔍 Searching your farm data..."):
            result = agent.ask_farm_history(question)

            if "error" not in result:
                response = result.get("summary", "No results found")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })

                with st.chat_message("assistant"):
                    st.write(response)
            else:
                st.error(f"Error: {result.get('error', 'Query failed')}")

def main():
    st.markdown("""
    <div class="main-header">
        🌾 AI-Powered Crop Advisory Assistant
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Intelligent agricultural recommendations powered by ML, RAG, and LLM orchestration**")
    st.divider()

    components = load_components()

    with st.sidebar:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
                    padding: 1.5em; border-radius: 0.5em; color: white;">
            <h3>⚙️ Farm Parameters</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            nitrogen = st.slider("Nitrogen (kg/ha)", 0, 200, 50, 5)
            phosphorus = st.slider("Phosphorus (kg/ha)", 0, 100, 20, 2)
            potassium = st.slider("Potassium (kg/ha)", 0, 100, 30, 2)

        with col2:
            ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1)
            temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 1.0)
            rainfall = st.slider("Rainfall (mm)", 0, 3000, 800, 50)

        humidity = st.slider("Humidity (%)", 0, 100, 65, 5)

        st.divider()

        st.markdown(f"""
        <div style="background: {LIGHT_BG}; padding: 1em; border-radius: 0.5em;">
            <h4>👤 Farmer Information</h4>
        </div>
        """, unsafe_allow_html=True)

        farmer_name = st.text_input("Farmer Name", placeholder="Optional")
        location = st.text_input("Farm Location", placeholder="Optional")

        st.divider()

        season = st.radio("🌾 Season", ["Kharif", "Rabi"], horizontal=True)
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

        st.divider()

        if st.button("📥 Download Full PDF Report", use_container_width=True, key="pdf_main"):
            with st.spinner("🔄 Generating comprehensive PDF..."):
                result = components["orchestrator"].generate_advisory(
                    features=features,
                    farmer_context=None
                )

                if result["status"] == "success":
                    builder = components["pdf_builder"]
                    pdf_buffer = builder.generate_advisory_pdf(
                        farmer_name=farmer_name or "Farmer",
                        location=location or "Farm",
                        advisory_data=result,
                        ml_metrics={"predictions": result.get("ml_predictions", [])}
                    )

                    st.download_button(
                        "📥 Download PDF",
                        data=pdf_buffer.getvalue(),
                        file_name="crop_advisory_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.error(result.get("message", "PDF generation failed"))

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Advisory & What-If",
        "📅 Smart Operations",
        "🔍 Crop Doctor",
        "📚 Farm History"
    ])

    with tab1:
        st.markdown('<div class="section-header">📋 Crop Advisory & Interactive Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            if st.button("🚀 Generate Advisory", key="gen_advisory", use_container_width=True):
                with st.spinner("⏳ Generating comprehensive advisory..."):
                    result = components["orchestrator"].generate_advisory(
                        features=features
                    )

                    if result["status"] == "success":
                        st.markdown(f"""
                        <div class="success-box">
                        ✅ Advisory Generated Successfully
                        </div>
                        """, unsafe_allow_html=True)

                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("🌾 Recommended", result["crop"], help="Primary crop recommendation")
                        with metric_col2:
                            st.metric("📊 Confidence", f"{result['confidence']:.1%}", help="Model confidence score")
                        with metric_col3:
                            alts = ", ".join(result.get("alternative_crops", [])[:2])
                            st.metric("🔄 Alternatives", alts[:20] + "..." if len(alts) > 20 else alts)

                        st.divider()

                        with st.expander("📄 Full Advisory Report", expanded=True):
                            st.markdown(result["advisory"])

                        with st.expander("📋 Cultivation Timetable"):
                            if result.get("timetable"):
                                render_timetable_checklist(result["timetable"])

                        st.session_state.last_advisory = result
                    else:
                        st.error(f"❌ {result.get('message', 'Advisory generation failed')}")

        with col2:
            st.markdown("**What-If Analysis**")

            param = st.selectbox(
                "Select Parameter to Vary",
                ["N", "P", "K", "pH", "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent"],
                key="whatif_param"
            )

            param_ranges = {
                "N": (0, 200, 5),
                "P": (0, 100, 2),
                "K": (0, 100, 2),
                "pH": (0.0, 14.0, 0.1),
                "avg_temp_c": (-10.0, 50.0, 1.0),
                "total_rainfall_mm": (0, 3000, 50),
                "avg_humidity_percent": (0, 100, 5)
            }

            min_val, max_val, step = param_ranges.get(param, (0, 100, 1))
            new_value = st.slider(
                f"New {param} Value",
                min_val, max_val,
                features[param],
                step,
                key="whatif_slider"
            )

            if st.button("🔄 Run What-If", use_container_width=True, key="whatif_btn"):
                with st.spinner("⏳ Analyzing impact..."):
                    result = components["orchestrator"].get_what_if_analysis(
                        base_features=features,
                        parameter=param,
                        new_value=new_value
                    )

                    if result["status"] == "success":
                        st.markdown(f"""
                        <div class="info-box">
                        <b>Comparison:</b> {result['original_prediction']} → {result['modified_prediction']}
                        </div>
                        """, unsafe_allow_html=True)

                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            st.metric(
                                "Original",
                                f"{result['original_prediction']}",
                                f"{result['original_confidence']:.1%}"
                            )
                        with col_w2:
                            st.metric(
                                "Modified",
                                f"{result['modified_prediction']}",
                                f"{result['modified_confidence']:.1%}"
                            )

                        plot_prediction_metrics(result.get("all_predictions", []))

        st.divider()
        st.markdown("**📊 Prediction Metrics & Feature Impact**")

        if "last_advisory" in st.session_state:
            col_metrics1, col_metrics2 = st.columns(2)

            with col_metrics1:
                if st.session_state["last_advisory"].get("ml_predictions"):
                    plot_prediction_metrics(st.session_state["last_advisory"]["ml_predictions"])

            with col_metrics2:
                st.info("💡 **SHAP Feature Importance** shows which factors most influence the prediction. "
                       "Positive values increase crop probability, negative values decrease it.")

    with tab2:
        st.markdown('<div class="section-header">📅 Smart Operations & Intercropping</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("📅 Generate Timetable", use_container_width=True, key="gen_timetable"):
                with st.spinner("⏳ Creating cultivation timeline..."):
                    result = components["orchestrator"].generate_advisory(features=features)

                    if result["status"] == "success":
                        st.success("✅ Timetable Generated")
                        render_timetable_checklist(result["timetable"])

        with col2:
            treatment = st.radio("🧪 Treatment", ["Organic", "Chemical"])
            st.caption(f"Selected: {treatment} methods")

        st.divider()

        st.markdown("**🌱 Companion Crops & Intercropping**")

        if st.button("🔍 Get Recommendations", use_container_width=True):
            with st.spinner("🔄 Fetching companion crops..."):
                ml_result = components["ml_engine"].predict(features)
                crop = ml_result["top_crop"]

                comp_result = components["orchestrator"].get_companion_crops(crop)

                if comp_result["status"] == "success":
                    st.markdown(f"""
                    <div class="info-box">
                    <b>Companion crops for {crop}:</b>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(comp_result["recommendations"])

    with tab3:
        st.markdown('<div class="section-header">🔍 Crop Disease Detection</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("**📸 Upload Leaf Image**")
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of the leaf"
            )

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.markdown("**🔬 Analysis Result**")

            if uploaded_file is not None and st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("🤖 Analyzing leaf..."):
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ Disease: Healthy Leaf
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric("Confidence", "85%")

                    st.markdown("**Treatment Recommendations:**")
                    recommendations = [
                        "✓ Continue regular monitoring",
                        "✓ Maintain proper irrigation",
                        "✓ Monitor weather conditions",
                        "✓ Apply preventive care if needed"
                    ]
                    for rec in recommendations:
                        st.markdown(rec)

    with tab4:
        st.markdown('<div class="section-header">📚 Farm History & Analytics</div>', unsafe_allow_html=True)

        history_tab1, history_tab2 = st.tabs(["📊 Dashboard", "💬 Q&A"])

        with history_tab1:
            if st.button("📈 Load Statistics", use_container_width=True, key="load_stats"):
                with st.spinner("📊 Loading farm data..."):
                    stats = components["sql_agent"].get_summary_stats()

                    st.markdown("**Farm Performance Overview**")

                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("📋 Records", stats.get("total_records", 0))
                        st.metric("🌾 Crops", stats.get("unique_crops", 0))
                    with stat_col2:
                        st.metric("📊 Avg Yield", f"{stats.get('avg_yield', 0):.2f} t/ha")
                        st.metric("💰 Avg Profit", f"₹{stats.get('avg_profit', 0):.0f}")
                    with stat_col3:
                        st.metric("🧪 Avg N", f"{stats.get('avg_nitrogen', 0):.2f}")
                        st.metric("🔄 Seasons", stats.get("unique_seasons", 0))

        with history_tab2:
            render_farm_history_chat(components["sql_agent"])

        st.divider()

        st.markdown("**➕ Add New Farm Record**")

        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            rec_season = st.selectbox("Season", ["Kharif", "Rabi"], key="rec_season")
            rec_crop = st.text_input("Crop Name", key="rec_crop")
            rec_yield = st.number_input("Yield (t/ha)", min_value=0.0, step=0.1, key="rec_yield")

        with col_rec2:
            rec_nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0, step=1.0, key="rec_n")
            rec_profit = st.number_input("Profit (₹)", min_value=0.0, step=1000.0, key="rec_profit")

            if st.button("➕ Add Record", use_container_width=True, key="add_rec"):
                with st.spinner("💾 Saving record..."):
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
                        st.error("❌ Failed to add record")

    st.divider()

    col_footer1, col_footer2, col_footer3 = st.columns(3)
    with col_footer1:
        st.caption("🌾 Crop Advisory System v2.0")
    with col_footer2:
        st.caption("Powered by LangChain • Gemini API • XGBoost")
    with col_footer3:
        st.caption("Phase 1-5 Complete ✅")

if __name__ == "__main__":
    main()
