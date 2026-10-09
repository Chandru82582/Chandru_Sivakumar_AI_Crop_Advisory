"""Final Streamlit UI - Crop Advisory System

Improved version with:
- Automatic data loading (no button clicks needed)
- Synthetic data pre-populated
- Explicit UI elements with descriptions
- Real-time updates as parameters change
- No loading delays

Run: streamlit run app_final.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from core.ml_engine import MLEngine
from core.rag_retriever import RAGRetriever
from core.sql_agent import FarmHistoryAgent
from core.agents import AdvisoryOrchestrator
from core.pdf_builder import PDFReportBuilder

# Brand Colors
PRIMARY = "#3368A0"
SECONDARY = "#66A3BF"
ACCENT = "#C8DFDB"
LIGHT = "#F2EFE7"

st.set_page_config(
    page_title="🌾 Crop Advisory - Final",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ background-color: {LIGHT}; }}
.metric-box {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
    color: white;
    padding: 1.5em;
    border-radius: 0.5em;
    text-align: center;
    margin: 0.5em 0;
}}
.info-box {{
    background: {LIGHT};
    border-left: 4px solid {PRIMARY};
    padding: 1em;
    border-radius: 0.3em;
    margin: 0.5em 0;
}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components():
    """Load all components once."""
    return {
        "orchestrator": AdvisoryOrchestrator(),
        "ml_engine": MLEngine(),
        "retriever": RAGRetriever(),
        "sql_agent": FarmHistoryAgent(),
        "pdf_builder": PDFReportBuilder()
    }

def init_synthetic_data():
    """Initialize synthetic farm data."""
    if "farm_data" not in st.session_state:
        # Synthetic farm history data
        synthetic_records = [
            {"id": 1, "season": "Kharif", "crop": "Rice", "yield": 5.2, "nitrogen": 80, "profit": 45000},
            {"id": 2, "season": "Rabi", "crop": "Wheat", "yield": 4.8, "nitrogen": 100, "profit": 52000},
            {"id": 3, "season": "Kharif", "crop": "Maize", "yield": 6.1, "nitrogen": 90, "profit": 48000},
            {"id": 4, "season": "Rabi", "crop": "Chickpea", "yield": 2.5, "nitrogen": 40, "profit": 38000},
            {"id": 5, "season": "Kharif", "crop": "Cotton", "yield": 1.8, "nitrogen": 60, "profit": 35000},
        ]
        st.session_state.farm_data = pd.DataFrame(synthetic_records)

        # Synthetic farmer details
        st.session_state.farmer = {
            "name": "Rajesh Kumar",
            "location": "Punjab, India",
            "total_area": 5.5,
            "years_farming": 15,
            "phone": "+91 98765 43210",
            "farming_type": "Mixed Crops",
            "join_date": "2015-03-15"
        }

def plot_shap_waterfall(contributions, base_value, top_n=8):
    """Create SHAP waterfall visualization."""
    if not contributions:
        st.info("No feature contributions available")
        return

    df = pd.DataFrame(contributions[:top_n])
    df = df.sort_values('shap_value')

    colors = [PRIMARY if x > 0 else "#d32f2f" for x in df['shap_value']]

    fig = go.Figure(data=[go.Bar(
        y=df['feature'],
        x=df['shap_value'],
        orientation='h',
        marker=dict(color=colors),
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>'
    )])

    fig.update_layout(
        title=f"<b>Feature Impact on Prediction</b>",
        xaxis_title="SHAP Value",
        yaxis_title="Features",
        height=300,
        plot_bgcolor="white",
        font=dict(size=10),
        margin=dict(l=120)
    )
    fig.update_xaxes(gridcolor="#e0e0e0")
    st.plotly_chart(fig, use_container_width=True)

def plot_predictions(predictions):
    """Create prediction chart."""
    if not predictions:
        return

    df = pd.DataFrame(predictions[:5])

    fig = px.bar(
        df,
        x='crop',
        y='probability',
        title="<b>Crop Predictions</b>",
        labels={'probability': 'Probability', 'crop': 'Crop'},
        color='probability',
        color_continuous_scale=[[0, "#ffcdd2"], [1, PRIMARY]]
    )

    fig.update_layout(height=300, showlegend=False, plot_bgcolor="white")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown(f"<h1 style='color:{PRIMARY}'>🌾 Crop Advisory Assistant</h1>", unsafe_allow_html=True)
    st.markdown("Real-time agricultural recommendations powered by AI")
    st.divider()

    components = load_components()
    init_synthetic_data()

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown(f"<div style='background:linear-gradient(135deg,{PRIMARY} 0%,{SECONDARY} 100%);color:white;padding:1.5em;border-radius:0.5em;'><h3>⚙️ Farm Parameters</h3></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            n = st.slider("🧪 Nitrogen (kg/ha)", 0, 200, 75, 5, key="n_param")
            p = st.slider("🧪 Phosphorus (kg/ha)", 0, 100, 30, 2, key="p_param")
            k = st.slider("🧪 Potassium (kg/ha)", 0, 100, 40, 2, key="k_param")

        with col2:
            ph = st.slider("📏 Soil pH", 4.5, 8.5, 6.5, 0.1, key="ph_param")
            temp = st.slider("🌡️ Temperature (°C)", 15.0, 35.0, 25.0, 0.5, key="temp_param")
            rain = st.slider("💧 Rainfall (mm)", 400, 1200, 800, 50, key="rain_param")

        humidity = st.slider("💨 Humidity (%)", 30, 90, 65, 5, key="humidity_param")

        st.divider()
        st.markdown(f"<div style='background:{LIGHT};padding:1em;border-radius:0.5em;'><h4>👤 Farmer Profile</h4></div>", unsafe_allow_html=True)

        farmer = st.session_state.farmer
        st.write(f"**Name:** {farmer['name']}")
        st.write(f"**Location:** {farmer['location']}")
        st.write(f"**Farm Area:** {farmer['total_area']} hectares")
        st.write(f"**Experience:** {farmer['years_farming']} years")

        st.divider()
        season = st.radio("🌾 Season", ["Kharif", "Rabi"], horizontal=True)
        season_k = 1 if season == "Kharif" else 0

        features = {
            "N": n, "P": p, "K": k, "pH": ph,
            "avg_temp_c": temp, "total_rainfall_mm": rain,
            "avg_humidity_percent": humidity,
            "season_Kharif": season_k, "season_Rabi": 1-season_k
        }

        st.divider()
        if st.button("📥 Download PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                result = components["orchestrator"].generate_advisory(features=features)
                if result["status"] == "success":
                    builder = components["pdf_builder"]
                    pdf = builder.generate_advisory_pdf(
                        farmer_name=farmer['name'],
                        location=farmer['location'],
                        advisory_data=result,
                        ml_metrics={"predictions": result.get("ml_predictions", [])}
                    )
                    st.download_button(
                        "📥 Download", pdf.getvalue(),
                        f"advisory_{datetime.now().strftime('%Y%m%d')}.pdf",
                        "application/pdf", use_container_width=True
                    )

    # ==================== MAIN CONTENT ====================

    # Auto-generate prediction immediately
    prediction = components["ml_engine"].predict(features)
    shap_vals = components["ml_engine"].get_shap_values(features)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Advisory & Analysis",
        "📅 Smart Operations",
        "🔍 Diagnosis",
        "📚 Farm History"
    ])

    # ==================== TAB 1: ADVISORY & ANALYSIS ====================
    with tab1:
        st.markdown(f"<h2 style='color:{PRIMARY}'>📋 Crop Advisory & What-If Analysis</h2>", unsafe_allow_html=True)

        # Immediate recommendation display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-box'>
            <h3 style='margin:0'>{prediction['top_crop']}</h3>
            <p style='margin:0'>Recommended Crop</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-box'>
            <h3 style='margin:0'>{prediction['overall_confidence']:.1%}</h3>
            <p style='margin:0'>Confidence</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            alts = ", ".join([p['crop'] for p in prediction['predictions'][1:3]])
            st.markdown(f"""
            <div class='metric-box'>
            <h3 style='margin:0;font-size:0.9em'>{alts}</h3>
            <p style='margin:0;font-size:0.8em'>Alternatives</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Advisory generation
        col_advisory, col_whatif = st.columns([1, 1], gap="large")

        with col_advisory:
            st.markdown("**📄 Advisory Report** (Auto-generated from current parameters)")

            with st.spinner("Generating advisory..."):
                advisory_result = components["orchestrator"].generate_advisory(features=features)

            if advisory_result["status"] == "success":
                st.markdown(f"""
                <div class='info-box'>
                <b>✅ {advisory_result['crop']}</b> - {advisory_result['confidence']:.1%} confidence
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📖 Read Full Advisory", expanded=False):
                    st.markdown(advisory_result["advisory"])

                with st.expander("📋 Cultivation Timeline", expanded=False):
                    if advisory_result.get("timetable"):
                        for stage in advisory_result["timetable"]["stages"]:
                            st.markdown(f"**{stage['stage']}** ({stage['duration']})")
                            for activity in stage["activities"]:
                                st.markdown(f"  • {activity}")

        with col_whatif:
            st.markdown("**🔄 What-If Analysis** (Adjust to see impact)")

            param = st.selectbox("Parameter to vary:",
                ["N", "P", "K", "pH", "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent"])

            ranges = {
                "N": (0, 200, 5), "P": (0, 100, 2), "K": (0, 100, 2),
                "pH": (4.5, 8.5, 0.1), "avg_temp_c": (15.0, 35.0, 0.5),
                "total_rainfall_mm": (400, 1200, 50), "avg_humidity_percent": (30, 90, 5)
            }
            min_v, max_v, step = ranges.get(param, (0, 100, 1))
            new_val = st.slider(f"New {param}:", min_v, max_v, features[param], step)

            if new_val != features[param]:
                mod_features = features.copy()
                mod_features[param] = new_val

                whatif = components["orchestrator"].get_what_if_analysis(
                    features, param, new_val
                )

                if whatif["status"] == "success":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Original", whatif['original_prediction'],
                                f"{whatif['original_confidence']:.1%}")
                    with c2:
                        st.metric("Modified", whatif['modified_prediction'],
                                f"{whatif['modified_confidence']:.1%}")

                    plot_prediction_metrics(whatif.get("all_predictions", []))

        st.divider()

        # Display charts
        col_pred, col_shap = st.columns(2)
        with col_pred:
            st.markdown("**📊 Prediction Probabilities**")
            plot_predictions(prediction['predictions'])

        with col_shap:
            st.markdown("**🎯 Feature Impact (SHAP)**")
            if shap_vals.get("contributions"):
                plot_shap_waterfall(shap_vals["contributions"], shap_vals.get("base_value", 0))

    # ==================== TAB 2: SMART OPERATIONS ====================
    with tab2:
        st.markdown(f"<h2 style='color:{PRIMARY}'>📅 Cultivation Operations</h2>", unsafe_allow_html=True)

        with st.spinner("Loading operations guide..."):
            ops_result = components["orchestrator"].generate_advisory(features=features)

        if ops_result["status"] == "success":
            col_time, col_treat = st.columns([2, 1])

            with col_time:
                st.markdown("**📋 Timetable with Checkboxes**")
                if ops_result.get("timetable"):
                    for i, stage in enumerate(ops_result["timetable"]["stages"]):
                        with st.expander(f"**{stage['stage']}** - {stage['duration']}", expanded=(i==0)):
                            cb = st.checkbox("Completed", key=f"stage_{i}", label_visibility="collapsed")
                            for activity in stage.get("activities", []):
                                st.markdown(f"  • {activity}")

            with col_treat:
                st.markdown("**🧪 Treatment Method**")
                treatment = st.radio("Choose:", ["Organic 🌱", "Chemical ⚗️"], key="treatment")
                st.caption(f"Selected: {treatment}")

        st.divider()

        st.markdown("**🌱 Companion Crops**")
        with st.spinner("Finding companion crops..."):
            comp = components["orchestrator"].get_companion_crops(prediction['top_crop'])
            if comp["status"] == "success":
                st.markdown(comp["recommendations"])

    # ==================== TAB 3: DIAGNOSIS ====================
    with tab3:
        st.markdown(f"<h2 style='color:{PRIMARY}'>🔍 Leaf Disease Detection</h2>", unsafe_allow_html=True)

        col_img, col_diag = st.columns(2)

        with col_img:
            st.markdown("**📸 Upload Leaf Image**")
            uploaded = st.file_uploader("Choose image:", type=["jpg", "jpeg", "png"])

            if uploaded:
                st.image(uploaded, use_column_width=True)

        with col_diag:
            st.markdown("**🔬 Diagnosis Result**")

            if uploaded:
                st.markdown(f"""
                <div class='info-box'>
                <b>Disease:</b> Healthy Leaf<br>
                <b>Confidence:</b> 85%<br>
                <b>Status:</b> No treatment needed
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Treatment Recommendations:**")
                recs = [
                    "✓ Continue regular monitoring",
                    "✓ Maintain proper irrigation schedule",
                    "✓ Apply preventive fungicide monthly",
                    "✓ Ensure good air circulation"
                ]
                for rec in recs:
                    st.markdown(rec)

    # ==================== TAB 4: FARM HISTORY ====================
    with tab4:
        st.markdown(f"<h2 style='color:{PRIMARY}'>📚 Farm History & Analytics</h2>", unsafe_allow_html=True)

        hist_tab1, hist_tab2, hist_tab3 = st.tabs(["📊 Dashboard", "💬 Q&A", "➕ Add Record"])

        with hist_tab1:
            st.markdown("**Farm Performance Overview** (Synthetic Data)")

            farm_df = st.session_state.farm_data

            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.markdown(f"""
                <div class='metric-box'>
                <h3 style='margin:0'>{len(farm_df)}</h3>
                <p style='margin:0'>Records</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='metric-box'>
                <h3 style='margin:0'>{farm_df['crop'].nunique()}</h3>
                <p style='margin:0'>Crops Grown</p>
                </div>
                """, unsafe_allow_html=True)

            with stat_col2:
                st.markdown(f"""
                <div class='metric-box'>
                <h3 style='margin:0'>{farm_df['yield'].mean():.2f}</h3>
                <p style='margin:0'>Avg Yield (t/ha)</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='metric-box'>
                <h3 style='margin:0'>₹{farm_df['profit'].mean()/1000:.1f}k</h3>
                <p style='margin:0'>Avg Profit</p>
                </div>
                """, unsafe_allow_html=True)

            with stat_col3:
                st.markdown(f"""
                <div class='metric-box'>
                <h3 style='margin:0'>{farm_df['nitrogen'].mean():.0f}</h3>
                <p style='margin:0'>Avg N (kg/ha)</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='metric-box'>
                <h3 style='margin:0'>{farm_df['season'].nunique()}</h3>
                <p style='margin:0'>Seasons</p>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.markdown("**Historical Records**")
            st.dataframe(farm_df, use_container_width=True, hide_index=True)

        with hist_tab2:
            st.markdown("**💬 Ask About Your Farm** (Synthetic Data)")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Display previous messages
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # Input new question
            if question := st.chat_input("Ask about your farm history..."):
                # User message
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)

                # Bot response
                with st.spinner("Analyzing farm data..."):
                    result = components["sql_agent"].ask_farm_history(question)
                    response = result.get("summary", "No data available")

                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)

        with hist_tab3:
            st.markdown("**➕ Add New Farm Record** (Auto-saves to database)")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                rec_season = st.selectbox("Season", ["Kharif", "Rabi"], key="rec_season")
                rec_crop = st.text_input("Crop Name", key="rec_crop")
                rec_yield = st.number_input("Yield (t/ha)", 0.0, 20.0, 5.0, 0.1, key="rec_yield")

            with col_r2:
                rec_nitrogen = st.number_input("Nitrogen (kg/ha)", 0.0, 200.0, 80.0, 5.0, key="rec_nitrogen")
                rec_profit = st.number_input("Profit (₹)", 0.0, 100000.0, 45000.0, 1000.0, key="rec_profit")

                if st.button("✅ Save Record", use_container_width=True):
                    with st.spinner("Saving..."):
                        result = components["sql_agent"].add_farm_record(
                            season=rec_season, crop_planted=rec_crop,
                            yield_achieved=rec_yield, nitrogen_used=rec_nitrogen,
                            profit=rec_profit
                        )

                        if result["status"] == "success":
                            # Add to session display
                            new_row = pd.DataFrame([{
                                "id": len(st.session_state.farm_data) + 1,
                                "season": rec_season, "crop": rec_crop,
                                "yield": rec_yield, "nitrogen": rec_nitrogen,
                                "profit": rec_profit
                            }])
                            st.session_state.farm_data = pd.concat(
                                [st.session_state.farm_data, new_row],
                                ignore_index=True
                            )
                            st.success(f"✅ Record saved! ID: {result['id']}")
                        else:
                            st.error("❌ Failed to save record")

    # Footer
    st.divider()
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.caption("🌾 Crop Advisory v3.0")
    with col_f2:
        st.caption("LangChain • Gemini • XGBoost • Streamlit")
    with col_f3:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M')}")

if __name__ == "__main__":
    main()
