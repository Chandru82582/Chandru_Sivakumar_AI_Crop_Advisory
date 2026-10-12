"""Crop Advisory System - Enterprise AI Agritech Platform.

A modern, responsive decision support platform for farmers and agronomists:
- Fast startup and responsive interaction with intelligent resource caching
- Default state: Tamil Nadu (Cauvery Delta, Thanjavur soil and climate baseline)
- Farmer-first Agronomic Advisory Report & Action Checklist
- Robust, fixed-width Interactive Monthly Calendar, Weekly Sprints, and Daily Action Checklist
- Crop Doctor: Instant Leaf Disease Vision Diagnosis
- Farm History Analytics & Natural Language Farm Assistant
- Publication-Grade Multi-Page PDF Advisory Dossier Export
- System Observability Hub: SQLite Explorer, Policy Docs, and RAG Retrieval Testing Sandbox

Run: streamlit run app.py
"""
import io
import os
import re
import json
import sqlite3
import calendar
import html
import textwrap
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from core.ml_engine import MLEngine
from core.rag_retriever import RAGRetriever
from core.sql_agent import FarmHistoryAgent
from core.agents import AdvisoryOrchestrator
from core.pdf_builder import PDFReportBuilder
from core.plant_doctor import PlantDoctor
from core.guardrails import AdvisoryGuardrails

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AgriAdvisory AI | Tamil Nadu Smart Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Brand Color Palette
PRIMARY = "#1B4D3E"      # Deep Forest Pine
SECONDARY = "#2D6A4F"    # Lush Emerald
ACCENT = "#52B788"       # Fresh Sage Mint
SURFACE_BG = "#F8FAF8"   # Clean Soft Off-White
TEXT_MAIN = "#1E293B"    # Slate Dark
TEXT_MUTED = "#64748B"   # Slate Muted
BORDER_COLOR = "#CBD5E1" # Crisp Border

# ==========================================
# CUSTOM STYLING (ENTERPRISE UI)
# ==========================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_MAIN};
    }}
    
    .stApp {{
        background-color: {SURFACE_BG};
    }}
    
    /* Header Brand */
    .brand-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.2rem 1.8rem;
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(27, 77, 62, 0.25);
    }}
    .brand-title {{
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }}
    .brand-subtitle {{
        font-size: 0.95rem;
        opacity: 0.88;
        margin: 0.25rem 0 0 0;
    }}
    .brand-tag {{
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.28);
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}

    /* Metric Cards */
    .kpi-card {{
        background: white;
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
    }}
    .kpi-label {{
        font-size: 0.82rem;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        font-size: 1.85rem;
        font-weight: 800;
        color: {PRIMARY};
        line-height: 1.2;
    }}
    .kpi-subtext {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
        margin-top: 0.35rem;
    }}

    /* Card Panels */
    .content-panel {{
        background: white;
        border: 1px solid {BORDER_COLOR};
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }}
    .panel-header {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {PRIMARY};
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* Disease Diagnosis Card */
    .diagnosis-badge {{
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
    }}
    .badge-healthy {{
        background: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #A5D6A7;
    }}
    .badge-infected {{
        background: #FFEBEE;
        color: #C62828;
        border: 1px solid #FFCDD2;
    }}

    /* Streamlit Tabs Navigation */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.65rem 1.25rem;
        border-radius: 8px;
        color: {TEXT_MUTED};
        border: none;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%) !important;
        color: white !important;
        box-shadow: 0 4px 10px rgba(27, 77, 62, 0.2);
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        color: white;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, {SECONDARY} 0%, {ACCENT} 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(45, 106, 79, 0.25);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid {BORDER_COLOR};
    }}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# ZERO-JS BULLETPROOF TABLE COMPONENT
# Fixes: TypeError Failed to fetch dynamically imported module DataFrame.CFF5tBU0.js
# =========================================================================
def render_styled_table(
    df: pd.DataFrame,
    max_height: str = "420px",
    download_filename: Optional[str] = "farm_data.csv",
    key: Optional[str] = None
) -> None:
    """Render a clean, high-performance HTML/CSS table with zero JS bundle dependencies."""
    if df is None or df.empty:
        st.info("No records found to display.")
        return

    display_df = df.copy()

    headers_html = "".join([
        f"<th style='padding: 10px 14px; text-align: left; font-size: 0.82rem; font-weight: 700; color: #1B4D3E; background: #E8F5E9; border-bottom: 2px solid #52B788; text-transform: uppercase; letter-spacing: 0.5px;'>{col.replace('_', ' ').title()}</th>"
        for col in display_df.columns
    ])

    rows_html = []
    for idx, row in display_df.iterrows():
        bg = "#FFFFFF" if idx % 2 == 0 else "#F8FAF8"
        cells_html = []
        for col in display_df.columns:
            val = row[col]
            col_l = col.lower()
            
            if col_l == "season":
                badge_bg = "#DEF7EC" if str(val).lower() == "kharif" else "#FEF08A"
                badge_color = "#03543F" if str(val).lower() == "kharif" else "#713F12"
                cell_content = f"<span style='background:{badge_bg}; color:{badge_color}; padding:2px 8px; border-radius:12px; font-weight:600; font-size:0.75rem;'>{val}</span>"
            elif col_l in ["crop_planted", "crop"]:
                cell_content = f"<b>{val}</b>"
            elif isinstance(val, (int, np.integer)):
                cell_content = f"{val:,}"
            elif isinstance(val, (float, np.floating)):
                if any(k in col_l for k in ["profit", "price", "revenue", "cost"]):
                    cell_content = f"₹{val:,.2f}"
                else:
                    cell_content = f"{val:.2f}"
            elif isinstance(val, bool):
                cell_content = "✅ Yes" if val else "❌ No"
            else:
                cell_content = str(val) if val is not None else "-"
                
            cells_html.append(f"<td style='padding: 8px 14px; font-size: 0.85rem; color: #1E293B; border-bottom: 1px solid #E2E8F0;'>{cell_content}</td>")

        row_str = f"<tr style='background: {bg}; transition: background 0.15s ease;' onmouseover=\"this.style.background='#F0FDF4'\" onmouseout=\"this.style.background='{bg}'\">" + "".join(cells_html) + "</tr>"
        rows_html.append(row_str)

    table_body = "".join(rows_html)

    table_wrapper = f"""
    <div style="width: 100%; border: 1px solid #CBD5E1; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.04); margin-bottom: 0.75rem;">
        <div style="max-height: {max_height}; overflow-y: auto; overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; text-align: left; font-family: 'Inter', sans-serif;">
                <thead>
                    <tr>{headers_html}</tr>
                </thead>
                <tbody>
                    {table_body}
                </tbody>
            </table>
        </div>
    </div>
    """
    st.markdown(table_wrapper, unsafe_allow_html=True)

    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Export Table Data ({len(df)} rows CSV)",
        data=csv_bytes,
        file_name=download_filename or "data_export.csv",
        mime="text/csv",
        key=key or f"dl_{abs(hash(str(df.columns) + str(len(df))))}"
    )


# =========================================================================
# CROPPING CALENDAR & HIGHLY DETAILED FARMER ACTION CHECKLIST
# =========================================================================
def get_crop_calendar_events(
    crop: str,
    season: str,
    n: float,
    p: float,
    k: float,
    ph: float,
    sowing_date: date
) -> list[dict[str, Any]]:
    """Generate structured, highly detailed operational schedule with WHEN, WHAT, DOSAGES, and HOW."""
    # Fertilizer calculation per acre (1 ha = 2.47 acres)
    n_acre = n / 2.47
    p_acre = p / 2.47
    k_acre = k / 2.47

    events = [
        {
            "day_offset": -14,
            "phase": "Phase 1: Pre-Planting & Field Preparation",
            "stage": "Deep Plowing",
            "category": "Soil & Land Prep",
            "priority": "High",
            "when": "14 Days Before Sowing (Day -14)",
            "what": f"Summer Deep Plowing & Soil Solarization for {crop}",
            "dosage": "Deep tillage to 20-25 cm depth. Incorporate Dhaincha or Sunn hemp green manure @ 20 kg/ha (8 kg/acre).",
            "how": "Perform one deep moldboard plowing followed by two cross-harrowings to break large soil clods, expose soil-borne insect pupae (stem borer, cutworm) to direct sunlight, and improve soil aeration.",
            "organic": "Incorporate green manure crop (Dhaincha or Sunn hemp) @ 20 kg/ha for biological nitrogen fixation.",
            "chemical": "If nematode history exists, apply soil solarization with transparent 100-gauge polyethylene film for 12 days."
        },
        {
            "day_offset": -7,
            "phase": "Phase 1: Pre-Planting & Field Preparation",
            "stage": "Basal Manuring",
            "category": "Nutrition",
            "priority": "High",
            "when": "7 Days Before Sowing (Day -7)",
            "what": "Organic Farmyard Manure (FYM) & Soil Conditioning",
            "dosage": f"FYM @ 25 tons/ha (10 tons/acre) or Vermicompost @ 5 tons/ha (2 tons/acre). Given soil pH {ph:.1f}, balance with organic buffer.",
            "how": "Evenly broadcast well-decomposed FYM or enriched vermicompost across the field. Mix thoroughly into the top 15 cm soil using a cultivator. This boosts cation-exchange capacity, soil beneficial microbes, and water-holding capacity.",
            "organic": "Enrich 500 kg FYM with Trichoderma viride and Pseudomonas fluorescens @ 5 kg/ha 10 days before field incorporation.",
            "chemical": f"If soil pH < 6.0 apply agricultural lime @ 1.5 t/ha; if pH > 8.0 apply gypsum @ 2.0 t/ha with leaching."
        },
        {
            "day_offset": -3,
            "phase": "Phase 1: Pre-Planting & Field Preparation",
            "stage": "Basal Fertilization",
            "category": "Nutrition",
            "priority": "Critical",
            "when": "3 Days Before Sowing (Day -3)",
            "what": "Basal NPK Application (50% N + 100% P + 50% K)",
            "dosage": f"Per Hectare: DAP @ {p/0.46:.0f} kg, MOP @ {k*0.5/0.6:.0f} kg, Urea @ {max(0, n*0.5 - (p/0.46)*0.18)/0.46:.0f} kg. | Per Acre: DAP @ {(p/0.46)/2.47:.0f} kg, MOP @ {(k*0.5/0.6)/2.47:.0f} kg, Urea @ {max(0, n*0.5 - (p/0.46)*0.18)/(0.46*2.47):.0f} kg.",
            "how": "Band place or broadcast basal fertilizers evenly just before final puddling or leveling. Place 4-5 cm below and to the side of the seed line so young seedling roots access nutrients without fertilizer burn.",
            "organic": "Apply neem cake @ 250 kg/ha (100 kg/acre) + rock phosphate @ 150 kg/ha (60 kg/acre) + wood ash @ 200 kg/ha.",
            "chemical": f"Apply calculated DAP + MOP + Urea split evenly across the field before the last leveling pass."
        },
        {
            "day_offset": -1,
            "phase": "Phase 1: Pre-Planting & Field Preparation",
            "stage": "Field Leveling",
            "category": "Soil & Land Prep",
            "priority": "Standard",
            "when": "1 Day Before Sowing (Day -1)",
            "what": "Laser Land Leveling & Drainage Channel Preparation",
            "dosage": "Construct perimeter drainage ditches (30 cm depth) and irrigation supply furrows with 0.2% slope.",
            "how": "Level the field perfectly to prevent standing water pockets that cause seed rot or uneven germination. Ensure good drainage outlets to safely discharge excess monsoon rainwater.",
            "organic": "Create raised beds (15 cm high, 1 m wide) for heavy clayey delta soils to provide superior root aeration.",
            "chemical": "Use laser land leveler to ensure uniform moisture penetration and save up to 25% irrigation water."
        },
        {
            "day_offset": 0,
            "phase": "Phase 2: Sowing & Seed Establishment",
            "stage": "Sowing Day",
            "category": "Sowing",
            "priority": "Critical",
            "when": "Sowing / Transplanting Day (Day 0)",
            "what": f"Certified Seed Inoculation & Precision Sowing of {crop}",
            "dosage": f"Seed Rate: 35-40 kg/ha (15-18 kg/acre). Bio-fertilizer: Azospirillum @ 200 g/acre + Phosphobacteria @ 200 g/acre + Pseudomonas @ 10 g/kg seed.",
            "how": "Treat certified seeds with bio-inoculants mixed in cooled rice porridge (kanji) and shade dry for 30 minutes. Sow at 3-4 cm depth with row spacing of 45 cm and plant-to-plant spacing of 15 cm. Maintain a light moisture regime.",
            "organic": "Beejamrit seed coating or soak seeds in 5% cow urine + garlic extract for 15 minutes before sowing.",
            "chemical": "Seed dressing with Carbendazim 50% WP @ 2 g/kg seed or Thiram 75% WP @ 2.5 g/kg seed to protect against damping-off."
        },
        {
            "day_offset": 3,
            "phase": "Phase 2: Sowing & Seed Establishment",
            "stage": "Weed Suppression",
            "category": "Weed Management",
            "priority": "High",
            "when": "3 Days After Sowing (Day +3)",
            "what": "Pre-Emergence Weed Barrier Application",
            "dosage": "Pendimethalin 30% EC @ 3.3 L/ha (1.3 L/acre) dissolved in 500 liters of water per hectare.",
            "how": "Spray pre-emergence herbicide uniformly across moist soil surface using a knapsack sprayer with a flat-fan nozzle within 48 to 72 hours of sowing. Do not disturb the soil layer after spraying.",
            "organic": "Spread organic straw mulching (Paddy straw @ 4-5 t/ha) over seed lines to block light from weed seeds.",
            "chemical": "Spray Pendimethalin 30% EC @ 3.3 L/ha in moist soil under calm, non-windy conditions."
        },
        {
            "day_offset": 7,
            "phase": "Phase 2: Sowing & Seed Establishment",
            "stage": "Germination Scouting",
            "category": "Soil & Land Prep",
            "priority": "High",
            "when": "7 Days After Sowing (Day +7)",
            "what": "Seedling Emergence & Vigour Inspection",
            "dosage": "Sample 5 random 1-meter square quadrats across the field. Target >85% seedling emergence.",
            "how": "Walk the field in a 'W' pattern. Count emerged seedlings. Check for soil crusting that may trap young sprouts; if crust has formed, provide a very light mist irrigation to soften the surface.",
            "organic": "Foliar mist of 2% cow urine distillate + liquid Jeevamrit (200 L/ha) to accelerate cotyledon emergence.",
            "chemical": "Inspect collar region of seedlings for wireworm, cutworm, or damping-off lesions."
        },
        {
            "day_offset": 14,
            "phase": "Phase 2: Sowing & Seed Establishment",
            "stage": "Gap Filling",
            "category": "Irrigation",
            "priority": "High",
            "when": "14 Days After Sowing (Day +14)",
            "what": "Gap Filling & First Establishment Irrigation",
            "dosage": "Re-sow seedlings in empty gaps using pre-soaked seeds. Provide light furrow irrigation (3-4 cm depth).",
            "how": "Transplant seedlings from dense spots into vacant patches during the evening hours. Provide immediate gentle irrigation to prevent transplant shock and maintain optimum plant density (150,000 plants/ha).",
            "organic": "Drench seedling roots with waste decomposer solution (10% dilution) to stimulate rapid adventitious rooting.",
            "chemical": "Provide furrow irrigation without allowing stagnant water to submerge delicate young shoot tips."
        },
        {
            "day_offset": 21,
            "phase": "Phase 3: Vegetative Growth & Canopy Management",
            "stage": "First Weeding",
            "category": "Weed Management",
            "priority": "High",
            "when": "21 Days After Sowing (Day +21)",
            "what": "First Manual/Mechanical Weeding & Sticky Traps",
            "dosage": "Install 15 yellow sticky traps/ha (6 traps/acre) and 10 blue sticky traps/ha at canopy height.",
            "how": "Operate a rotary cono-weeder between rows or perform hand weeding to aerate the root zone. Erect yellow sticky traps to trap whiteflies and aphids, and blue sticky traps for thrips.",
            "organic": "Use cono-weeder to bury weeds as green in-situ manure. Spray 5% Neem Seed Kernel Extract (NSKE).",
            "chemical": "If broadleaf and sedge weeds are dominant, apply 2,4-D Ethyl Ester @ 1.0 kg a.i./ha in moist soil."
        },
        {
            "day_offset": 30,
            "phase": "Phase 3: Vegetative Growth & Canopy Management",
            "stage": "First Top Dressing",
            "category": "Nutrition",
            "priority": "Critical",
            "when": "30 Days After Sowing (Day +30)",
            "what": f"First Top Dressing (25% Nitrogen Split: {n*0.25:.0f} kg N/ha)",
            "dosage": f"Per Hectare: Neem-coated Urea @ {n*0.25/0.46:.0f} kg. | Per Acre: Neem-coated Urea @ {(n*0.25/0.46)/2.47:.0f} kg ({((n*0.25/0.46)/2.47)/50:.1f} bags).",
            "how": "Broadcast neem-coated urea into moist soil in the late afternoon. Do not apply on standing water or dry soil. Irrigate 24 hours after application to wash nitrogen into the root zone without leaching.",
            "organic": "Soil drenching with enriched Jeevamrit @ 500 L/ha (200 L/acre) blended into irrigation channel water.",
            "chemical": f"Broadcast Neem-coated Urea @ {n*0.25/0.46:.0f} kg/ha immediately prior to scheduled light irrigation."
        },
        {
            "day_offset": 45,
            "phase": "Phase 3: Vegetative Growth & Canopy Management",
            "stage": "Micronutrients & Pest Check",
            "category": "Nutrition",
            "priority": "Critical",
            "when": "45 Days After Sowing (Day +45)",
            "what": "Foliar Micronutrient Boost & Canopy Pest Scouting",
            "dosage": "Zinc Sulfate (ZnSO4) @ 0.5% (5 g/L) + Ferrous Sulfate (FeSO4) @ 0.2% (2 g/L) + Lime 0.25% (2.5 g/L).",
            "how": "Dissolve Zinc Sulfate and lime separately, then mix and spray early in the morning when stomata are open. Scout under-surface of leaves for leaf folder caterpillars and stem borer 'dead hearts'.",
            "organic": "Foliar spray of 3% fermented Panchagavya + Dashparni Ark @ 10 ml/L to deter sucking pests and boost chlorophyll.",
            "chemical": "If pest count exceeds Economic Threshold (ETL), spray Chlorantraniliprole 18.5% SC @ 0.3 ml/L or Cartap hydrochloride 50% SP @ 2 g/L."
        },
        {
            "day_offset": 60,
            "phase": "Phase 4: Flowering & Reproductive Formation",
            "stage": "Flowering Water Care",
            "category": "Irrigation",
            "priority": "Critical",
            "when": "60 Days After Sowing (Day +60)",
            "what": "Critical Irrigation at Panicle Initiation / Flowering Onset",
            "dosage": "Maintain 4-5 cm uniform standing water depth or ensure soil moisture is at 75-80% field capacity.",
            "how": "Flowering is the most drought-sensitive physiological phase in the crop life cycle. Never allow the soil to crack or dry out during this period, as even 2 days of water stress causes flower dropping and empty grains (chaffiness).",
            "organic": "Spray sour buttermilk solution (50 ml/L) mixed with asafoetida (1 g/L) to prevent floral drop and fungal spores.",
            "chemical": "Ensure uninterrupted irrigation schedule; avoid water deficit during pollination and ovule fertilization."
        },
        {
            "day_offset": 75,
            "phase": "Phase 4: Flowering & Reproductive Formation",
            "stage": "Second Top Dressing",
            "category": "Nutrition",
            "priority": "Critical",
            "when": "75 Days After Sowing (Day +75)",
            "what": f"Second Top Dressing: Remaining 25% N ({n*0.25:.0f} kg) & 50% K ({k*0.5:.0f} kg)",
            "dosage": f"Per Hectare: Urea @ {n*0.25/0.46:.0f} kg + MOP @ {k*0.5/0.6:.0f} kg. | Per Acre: Urea @ {(n*0.25/0.46)/2.47:.0f} kg + MOP @ {(k*0.5/0.6)/2.47:.0f} kg.",
            "how": "Potassium applied at this stage enhances grain plumpness, strengthens crop stems against wind lodging, and accelerates starch translocation into ripening grains.",
            "organic": "Apply sieved wood ash @ 250 kg/ha (100 kg/acre) or spray potassium-rich seaweed liquid extract @ 3 ml/L.",
            "chemical": f"Broadcast Neem-coated Urea @ {n*0.25/0.46:.0f} kg/ha and MOP @ {k*0.5/0.6:.0f} kg/ha into moist soil."
        },
        {
            "day_offset": 90,
            "phase": "Phase 4: Flowering & Reproductive Formation",
            "stage": "Grain Filling Defense",
            "category": "Pest & Disease Defense",
            "priority": "High",
            "when": "90 Days After Sowing (Day +90)",
            "what": "Grain/Pod Filling Stage Surveillance & Avian Perches",
            "dosage": "Erect 40-50 bird perches (T-shaped bamboo sticks) per hectare (16-20 per acre) across the field.",
            "how": "Bird perches allow insectivorous birds (drongos, egrets) to perch and naturally hunt pod borers and armyworms. Inspect heads for false smut or grain discolorations.",
            "organic": "Spray Bacillus thuringiensis (Bt kurstaki) @ 1.5 kg/ha or Beauveria bassiana @ 2 kg/ha in the evening.",
            "chemical": "If brown plant hopper (BPH) or sheath blight appears, spray Pymetrozine 50% WDG @ 0.6 g/L or Azoxystrobin @ 1 ml/L."
        },
        {
            "day_offset": 105,
            "phase": "Phase 5: Physiological Maturity & Harvest",
            "stage": "Water Withdrawal",
            "category": "Irrigation",
            "priority": "High",
            "when": "105 Days After Sowing (Day +105)",
            "what": "Complete Irrigation Cessation & Field Dry-down",
            "dosage": "Open field drainage outlets completely. Withdraw all irrigation 10 to 14 days before harvest.",
            "how": "Draining standing water accelerates uniform physiological ripening, hardens the grain husks, and firms the soil to allow easy movement of combine harvesters or manual harvesting crews without wheel sinkage.",
            "organic": "Allow natural solar field dry-down; strictly refrain from synthetic chemical desiccant sprays.",
            "chemical": "Observe strict 14-day Pre-Harvest Interval (PHI) on any chemical protection applied previously."
        },
        {
            "day_offset": 115,
            "phase": "Phase 5: Physiological Maturity & Harvest",
            "stage": "Harvesting",
            "category": "Harvesting & Storage",
            "priority": "Critical",
            "when": "115 Days After Sowing (Day +115)",
            "what": f"Primary Harvest of {crop} Crop",
            "dosage": "Harvest when 85-90% of panicles turn golden yellow and grain moisture reaches 18-20%.",
            "how": "Cut crop close to the ground on bright, clear, sunny days. Thresh immediately using a clean mechanical thresher to avoid field losses from shattering or sudden rains.",
            "organic": "Perform manual harvesting with clean serrated sickles or combine harvester with sanitized cutter bar.",
            "chemical": "Thresh produce on clean plastic tarpaulins to avoid soil, pebble, or fungal spore contamination."
        },
        {
            "day_offset": 120,
            "phase": "Phase 5: Physiological Maturity & Harvest",
            "stage": "Hermetic Storage",
            "category": "Harvesting & Storage",
            "priority": "High",
            "when": "120 Days After Sowing (Day +120)",
            "what": "Sun Drying & Hermetic Warehouse Storage",
            "dosage": "Dry produce to <12% seed moisture content (grains crack crisply between teeth).",
            "how": "Spread threshed produce in a 5 cm thin layer on clean tarpaulins in bright sunshine for 2-3 days, turning every 2 hours. Pack in double-layered hermetic storage bags or clean metal bins placed on wooden pallets 30 cm above ground.",
            "organic": "Mix clean, dried neem leaves (2% w/w) or inert diatomaceous earth into storage bags to prevent weevils.",
            "chemical": "Fumigate enclosed warehouse storage bins with Aluminum phosphide tablets @ 3 g/ton produce under air-tight tarps."
        }
    ]

    for ev in events:
        ev["date"] = sowing_date + timedelta(days=ev["day_offset"])

    return events


# =========================================================================
# RESOURCE INITIALIZATION (SINGLETON CACHED & HIGH SPEED)
# =========================================================================
@st.cache_resource(show_spinner=False)
def load_app_core():
    """Load core engines once into memory with zero redundant weight reloads."""
    ml_engine = MLEngine()
    retriever = RAGRetriever()
    sql_agent = FarmHistoryAgent()
    orchestrator = AdvisoryOrchestrator(
        ml_engine=ml_engine,
        retriever=retriever,
        sql_agent=sql_agent
    )
    pdf_builder = PDFReportBuilder()
    plant_doctor = PlantDoctor(use_pretrained=False)
    
    return {
        "ml_engine": ml_engine,
        "retriever": retriever,
        "sql_agent": sql_agent,
        "orchestrator": orchestrator,
        "pdf_builder": pdf_builder,
        "plant_doctor": plant_doctor
    }


def ensure_baseline_farm_data(sql_agent: FarmHistoryAgent):
    """Seed baseline historical farm records for Tamil Nadu if database is empty."""
    try:
        stats = sql_agent.get_summary_stats()
        if stats.get("total_records", 0) == 0:
            sample_records = [
                ("Kharif", "Rice", 5.6, 80.0, 52000.0),
                ("Rabi", "Maize", 6.8, 90.0, 56000.0),
                ("Kharif", "Cotton(Lint)", 2.4, 75.0, 48000.0),
                ("Rabi", "Groundnut", 3.2, 45.0, 44000.0),
                ("Whole Year", "Banana", 38.0, 110.0, 125000.0),
                ("Kharif", "Urad", 1.4, 25.0, 36000.0),
            ]
            for season, crop, yld, n_used, prof in sample_records:
                sql_agent.add_farm_record(season, crop, yld, n_used, prof)
    except Exception:
        pass


# =========================================================================
# CACHED ADVISORY & PDF GENERATORS (INSTANT STREAMLIT RERUNS)
# =========================================================================
@st.cache_data(show_spinner=False)
def get_cached_farmer_advisory(
    crop: str,
    n: float, p: float, k: float, ph: float,
    temp: float, rain: float, hum: float,
    season: str, farmer_name: str, location: str, farm_size: float
) -> str:
    """Build a comprehensive, 100% farmer-oriented advisory report."""
    n_acre = n / 2.47
    p_acre = p / 2.47
    k_acre = k / 2.47
    acres = farm_size * 2.47

    soil_health = "Optimal & Highly Productive" if 6.2 <= ph <= 7.2 else ("Mildly Acidic (beneficial for pulses & rice)" if ph < 6.2 else "Alkaline (requires organic matter buffering)")
    
    # Tamil Nadu suitable varieties
    varieties_map = {
        "Rice": "CR 1009 Sub 1, ADT 45, CO 51, BPT 5204 (Samba/Thaladi), ASD 16",
        "Maize": "CO 6, Pioneer 3396, NK 6240, Syngenta NK 30",
        "Cotton(Lint)": "MCU 5, Suraj, SVPR 2, RCH 2 Bt Cotton",
        "Groundnut": "TMV 7, VRI 2, Kadiri 6, TMV 13",
        "Banana": "Grand Naine (G9), Poovan, Rasthali, Nendran",
        "Ginger": "Rio de Janeiro, Nadia, Maran",
        "Dry Chillies": "K1, K2, PKM 1, Ramanathapuram Mundu",
        "Turmeric": "BSR 1, BSR 2, CO 2, Alleppey Supreme",
        "Tapioca": "CO 2, CO 3, Mulluvadi, H 165",
        "Urad": "VBN 6, VBN 8, CO 6, T 9"
    }
    varieties = varieties_map.get(crop, f"Certified TNAU High-Yielding Varieties recommended for {season} season")

    return f"""### 🌾 Comprehensive Farmer Field Advisory: Cultivation Package for {crop}

**Plot Profile:** Farmer: **{farmer_name}** | Location: **{location}** | Land Area: **{farm_size:.1f} Hectares ({acres:.1f} Acres)**  
**Target Cropping Season:** **{season} Season** | Recommended Crop: **{crop}**  
**High-Yielding Varieties for Tamil Nadu:** `{varieties}`

---

#### 1. 🌱 Why This Crop Fits Your Field (Soil & Climate Suitability)
Your soil test analysis shows **pH {ph:.1f}** ({soil_health}), with available Nitrogen at **{n:.0f} kg/ha ({n_acre:.0f} kg/acre)**, Phosphorus at **{p:.0f} kg/ha ({p_acre:.0f} kg/acre)**, and Potassium at **{k:.0f} kg/ha ({k_acre:.0f} kg/acre)**.
- **Climatic Harmony:** With average temperatures at **{temp:.1f}°C** and annual precipitation around **{rain:.0f} mm** in {location}, this climate matches the exact physiological heat units and moisture thresholds required by **{crop}**.
- **Yield Benefit:** Balanced root nutrient assimilation is maximized at this pH, avoiding micronutrient lock-up.

---

#### 2. 🧪 Basal & Split Fertilizer Management Plan (Exact Farmer Quantities)
To achieve maximum grain filling without lodging or wasting inputs, apply your chemical fertilizers in **3 balanced splits**:

| Application Timing | Growth Stage | Dosage per Hectare | Dosage per Acre | Practical Farmer Action |
| :--- | :--- | :--- | :--- | :--- |
| **Basal Application** | At Final Puddling/Leveling | DAP: {p/0.46:.0f} kg<br>MOP: {k*0.5/0.6:.0f} kg<br>Urea: {max(0, n*0.5 - (p/0.46)*0.18)/0.46:.0f} kg | DAP: {(p/0.46)/2.47:.0f} kg<br>MOP: {(k*0.5/0.6)/2.47:.0f} kg<br>Urea: {max(0, n*0.5 - (p/0.46)*0.18)/(0.46*2.47):.0f} kg | Incorporate 5 cm below seed depth along with 10 tons/acre well-decomposed Farmyard Manure (FYM). |
| **First Top Dressing** | Active Tillering / Branching (Day 25-30) | Urea: {n*0.25/0.46:.0f} kg | Urea: {(n*0.25/0.46)/2.47:.0f} kg<br>(~{((n*0.25/0.46)/2.47)/50:.1f} bags) | Broadcast into moist soil during late afternoon. Irrigate 24 hours later. |
| **Second Top Dressing** | Panicle Initiation / Flowering (Day 70-75) | Urea: {n*0.25/0.46:.0f} kg<br>MOP: {k*0.5/0.6:.0f} kg | Urea: {(n*0.25/0.46)/2.47:.0f} kg<br>MOP: {(k*0.5/0.6)/2.47:.0f} kg | Critical for grain filling and stem strength against monsoonal winds. |

> 💡 **Micronutrient Tip:** If leaves show pale yellow striping (chlorosis), spray **Zinc Sulfate 0.5% (5 g/L) + Ferrous Sulfate 0.2% (2 g/L)** neutralized with lime during early morning hours at Day 40-45.

---

#### 3. 💧 Irrigation & Water Management
- **Early Stage (Day 0 - 20):** Maintain shallow water depth (2 cm) or light furrow irrigation. Never let soil crack dry, but avoid deep submergence of young shoots.
- **Critical Vegetative & Flowering Stage (Day 45 - 85):** Maintain 4-5 cm standing water. This is the most drought-sensitive stage—water stress here causes unfertilized chaffy grains.
- **Pre-Harvest Drying (Day 105 onwards):** Completely drain standing water 10 to 14 days before harvest. This hardens grain coats and firms the soil for combine harvesters.

---

#### 4. 🛡️ Pest & Disease Preventive Shield (Integrated Crop Care)
- **Preventive Organic Spray:** Spray **5% Neem Seed Kernel Extract (NSKE)** or **Neem Oil 1500 ppm @ 5 ml/liter** at 20 and 45 days after sowing to deter stem borer moths, leaf folders, and sucking aphids.
- **Biological Defense:** Install **5-6 pheromone traps per acre** to monitor yellow stem borer moths. Release *Trichogramma* egg parasitoids @ 20,000 per acre.
- **Curative Threshold Backup:** If brown planthopper or leaf folder caterpillars exceed 10% damaged leaves, spray **Chlorantraniliprole 18.5% SC @ 0.3 ml/L** or **Cartap Hydrochloride 50% SP @ 2 g/L**. Strictly follow a 15-day pre-harvest waiting interval.

---

#### 5. 💰 Expected Farm Economics & Yield Estimate
- **Projected Yield:** **4.8 to 6.2 Tons per Hectare (1.9 to 2.5 Tons / 25-30 Bags per Acre)**
- **Gross Estimated Revenue:** **₹1,15,000 to ₹1,45,000 per Hectare** (at prevailing Mandi MSP of ₹2,300/quintal)
- **Estimated Production Cost:** **₹48,000 to ₹55,000 per Hectare** (Seeds, Tillage, Fertilizers, Irrigation, Labor)
- **Expected Net Farm Profit:** **₹65,000 to ₹90,000 per Hectare (₹26,000 to ₹36,000 per Acre)**
"""


@st.cache_data(show_spinner=False)
def get_cached_companion_text(crop: str) -> str:
    """Get fast cached agronomic companion crop recommendations."""
    crop_l = crop.lower()
    if "rice" in crop_l:
        return """**🌾 Recommended Intercropping for Rice in Tamil Nadu:**
1. **Azolla (Biofertilizer Fern):** Dual-crop in standing water 7 days after transplanting (200 kg/acre). Fixes 25-30 kg atmospheric N/acre and suffocates weeds by 50%.
2. **Black Gram / Urad (Relay Pulse):** Broadcast into standing paddy 7-10 days before harvest (Samba relay cropping). Utilizes residual soil moisture and enriches nitrogen.
3. **Sesbania / Dhaincha (Border Trap Crop):** Sow along field bunds to trap stem borers and provide green fodder."""
    elif "cotton" in crop_l:
        return """**🌿 Recommended Intercropping for Cotton in Tamil Nadu:**
1. **Black Gram / Greengram (1:2 or 1:3 ratio):** Intercrop between cotton paired rows. Harvested in 65 days without reducing cotton lint yield.
2. **Cowpea (Trap Crop):** Plant on perimeter bunds to attract sucking pests away from cotton flowers.
3. **Marigold (Trap Crop):** Plant 1 marigold seedling every 15 cotton plants to attract American bollworm moths."""
    elif "maize" in crop_l:
        return """**🌽 Recommended Intercropping for Maize in Tamil Nadu:**
1. **Cowpea / Redgram (2:1 row ratio):** Boosts land equivalent ratio by 30%, enriches nitrogen, and prevents soil erosion.
2. **Soybean / Groundnut:** Excellent ground cover that suppresses weeds and yields high-protein companion harvest."""
    else:
        return f"""**🌱 Biological Intercropping for {crop}:**
1. **Legume Intercropping (Pulses/Beans):** Plant in 3:1 ratio for rhizobial biological nitrogen fixation.
2. **Trap Crops (Marigold / Mustard):** Plant perimeter strips to deter sucking insects and nematodes."""


def query_farm_assistant(
    query: str,
    farmer_name: str,
    top_crop: str,
    confidence: float,
    features: dict,
    calendar_events: list,
    sql_agent: FarmHistoryAgent,
    retriever: Any,
    llm: Any = None
) -> dict[str, Any]:
    """Provide practical, well-explained, zero-hallucination agricultural answers for farmers.
    
    Uses LLM (gemini-2.5-flash) grounded in SQLite crop history, active recommendation,
    and verified agronomic reference docs.
    """
    q_lower = query.lower()
    
    # 0. Agricultural Domain Guardrail: Block non-agricultural questions
    is_agri, reason = AdvisoryGuardrails.is_agricultural_query(query, llm=llm)
    if not is_agri:
        refusal = AdvisoryGuardrails.get_non_agricultural_refusal(farmer_name=farmer_name, query=query)
        return {
            "answer": refusal,
            "sql_info": None,
            "blocked_by_guardrail": True,
            "guardrail_reason": reason
        }

    # 1. Fetch real historical facts from SQLite
    history_records = sql_agent.ask_farm_history("SELECT * FROM crop_history")
    rows = history_records.get("results", [])
    unique_crops = sorted(list({r.get("crop_planted") for r in rows if r.get("crop_planted")}))
    
    best_profit_row = max(rows, key=lambda r: (r.get("profit") or 0)) if rows else None
    best_yield_row = max(rows, key=lambda r: (r.get("yield_achieved") or 0)) if rows else None
    
    avg_yield = (sum(r.get("yield_achieved") or 0 for r in rows) / len(rows)) if rows else 0.0
    avg_profit = (sum(r.get("profit") or 0 for r in rows) / len(rows)) if rows else 0.0
    
    # 2. Extract next upcoming milestone
    today = date.today()
    upcoming_events = [e for e in calendar_events if e["date"] >= today]
    next_event = upcoming_events[0] if upcoming_events else (calendar_events[0] if calendar_events else None)
    
    # 3. Check for SQL database query if asking about history/stats
    sql_context = ""
    is_history_query = any(k in q_lower for k in [
        "history", "past", "profit", "yield", "previous", "planted", "kharif", "rabi",
        "last year", "highest", "average", "avg", "record", "database", "earned"
    ])
    
    if is_history_query:
        sql_res = sql_agent.ask_farm_history(query)
        sql_context = f"SQLite Historical Query: {sql_res.get('query', '')}\nQuery Results: {sql_res.get('results', [])}\nSummary: {sql_res.get('summary', '')}"

    # 4. Initialize LLM if not supplied
    if llm is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
            except Exception:
                llm = None

    # 5. Invoke LLM with practical, well-explained agricultural prompt
    if llm is not None:
        try:
            rag_chunks = ""
            if any(k in q_lower for k in [
                "pest", "disease", "fertilizer", "water", "irrigation", "spray",
                "weed", "seed", "dosage", "nutrient", "npk", "urea", "dap", "potash"
            ]):
                try:
                    rag_chunks = retriever.get_context_string(f"{top_crop} {query}")
                    rag_chunks = rag_chunks[:1200]
                except Exception:
                    rag_chunks = ""

            prompt_text = f"""You are the Tamil Nadu Agricultural Field Assistant & Agronomy Advisor for farmer {farmer_name}.
The farmer has asked you a question. Provide a practical, clear, and well-explained answer that is easy for a farmer to understand and immediately put to use in the field.

Farmer's Question: {query}

Verified Farm Facts & Records:
- Recommended / Target Crop: {top_crop} (Suitability Match: {confidence:.0%})
- Soil & Environmental Metrics: Nitrogen (N)={features.get('N')} kg/ha, Phosphorus (P)={features.get('P')} kg/ha, Potassium (K)={features.get('K')} kg/ha, Soil pH={features.get('pH')}, Rainfall={features.get('total_rainfall_mm')} mm, Temperature={features.get('avg_temp_c')}°C
- Upcoming Scheduled Field Stage: {next_event['stage'] if next_event else 'None'} ({next_event['when'] if next_event else ''}) — Action: {next_event['what'] if next_event else ''} | Dosage: {next_event['dosage'] if next_event else ''}
- Crops Planted in Farm Database: {', '.join(unique_crops) if unique_crops else 'None'}
- Past Performance Facts: Best Profit: ₹{best_profit_row['profit']:,.0f} ({best_profit_row['crop_planted']} in {best_profit_row['season']}) if best_profit_row else 'None'; Average Yield: {avg_yield:.2f} t/ha; Average Profit: ₹{avg_profit:,.0f}
{f"- SQLite Farm History Records: {sql_context}" if sql_context else ""}
{f"- Official Agronomy Reference Guide: {rag_chunks}" if rag_chunks else ""}

Instructions:
1. Provide a practical, well-explained response (target 100 to 180 words total). Never give cryptic 1-sentence stubs, and never write exhausting long essays.
2. Structure your answer cleanly with bullet points:
   - Practical Recommendation (direct answer to the farmer's question)
   - Field Execution & Dosages (exact per-acre quantities, application splits, or preparation method)
   - Practical Field Tip / Precaution (irrigation, safety, or timing advice)
3. Agronomic Knowledge & Grounding:
   - Ground all recommendations in verified facts and standard Tamil Nadu Agricultural University (TNAU) / ICAR package of practices.
   - For Rice Urea scheduling: recommend 3-way split (25% basal, 50% active tillering, 25% panicle initiation); advise draining standing water before top dressing and re-irrigating after 24-48 hours.
   - For Sugarcane Potash (MOP): recommend split applications at early tillering (30-45 days) and earthing up (90-120 days) at 60-80 kg K2O/acre.
   - For Banana in alkaline soil (pH > 8.0): recommend Single Super Phosphate (SSP) near root zone + Farm Yard Manure (FYM) + Phosphobacteria for phosphorus solubilization.
   - For Groundnut pegging: recommend Gypsum (150-200 kg/acre) at 40-45 days for calcium and pod development; foliar spray of 0.5% KCl for moisture stress.
   - For Pest & Disease IPM: prioritize bio-control (Trichogramma, pheromone traps, Neem oil 3%, Pseudomonas, NSKE 5%).
   - NEVER recommend banned or illegal chemicals (Endosulfan, Paraquat, Monocrotophos, etc.). If asked, strictly warn against them and provide safe alternatives.
   - NEVER recommend dangerous overdoses (e.g., 300 kg/acre urea).
4. For past history or harvest questions: cite the exact figures from SQLite records and provide practical advice based on that performance.
5. Tone: Respectful, warm (e.g. "Vanakkam {farmer_name}!"), practical, and encouraging.
"""

            resp = llm.invoke(prompt_text)
            ans = getattr(resp, "content", str(resp)).strip()
            if ans:
                return {"answer": ans, "sql_info": sql_context if is_history_query else None}
        except Exception:
            pass

    # 6. Practical Deterministic Fallback (Reliable, structured, practical)
    if is_history_query:
        if any(k in q_lower for k in ["highest profit", "best profit", "most profit"]):
            if best_profit_row:
                ans = (
                    f"💰 **Best Profit Record from Farm Database:**\n"
                    f"- **Crop & Season:** **{best_profit_row['crop_planted']}** in {best_profit_row['season']}.\n"
                    f"- **Net Profit Realized:** **₹{best_profit_row['profit']:,.0f}**.\n"
                    f"- **Harvest Yield:** **{best_profit_row['yield_achieved']} t/ha**.\n"
                    f"- *Field Tip:* This rotation provided the highest economic return under your soil and water conditions."
                )
            else:
                ans = "No historical profit records found in database."
        elif any(k in q_lower for k in ["highest yield", "best yield", "top yield"]):
            if best_yield_row:
                ans = (
                    f"🌾 **Top Yield Record from Farm Database:**\n"
                    f"- **Crop & Season:** **{best_yield_row['crop_planted']}** in {best_yield_row['season']}.\n"
                    f"- **Peak Yield Achieved:** **{best_yield_row['yield_achieved']} t/ha**.\n"
                    f"- **Net Profit:** ₹{best_yield_row['profit']:,.0f}."
                )
            else:
                ans = "No historical yield records found."
        elif any(k in q_lower for k in ["past", "planted", "existing", "crops", "history"]):
            crops_str = ", ".join(unique_crops) if unique_crops else "None"
            ans = (
                f"📜 **Historical Crop Rotations & Yields:**\n"
                f"- **Crops Recorded:** **{crops_str}** across previous Kharif and Rabi seasons.\n"
                f"- **Average Harvest Yield:** **{avg_yield:.2f} t/ha**.\n"
                f"- **Average Net Return:** **₹{avg_profit:,.0f}** per season.\n"
                f"- All past harvest cycles are logged in your **Farm History Analytics** database."
            )
        else:
            q_res = sql_agent.ask_farm_history(query)
            ans = q_res.get("summary", "Query executed on farm history database.")
        return {"answer": ans, "sql_info": sql_context}

    # Check for fertilizer / nutrient questions FIRST before generic schedule words
    if any(k in q_lower for k in ["fertilizer", "nutrient", "urea", "dap", "npk", "potash", "mop", "gypsum", "dose", "dosage", "split"]):
        if "rice" in q_lower or "paddy" in top_crop.lower() or "rice" in top_crop.lower():
            ans = (
                f"🌱 **Recommended Urea & Nitrogen Split Schedule for Samba Rice (Cauvery Delta):**\n"
                f"- **Practical Recommendation:** Vanakkam {farmer_name}! Urea must not be applied all at once to prevent leaching, volatilization, and vegetative lodging. Split into three calibrated applications for maximum crop nitrogen uptake.\n"
                f"- **Field Execution & Dosages (Per Acre):** Recommended split schedule: apply **25% Basal** at final puddling/transplanting, **50% Top Dressing** at active tillering stage (25-30 days), and **25%** at panicle initiation stage (60-65 days).\n"
                f"- **Practical Field Tip & Precaution:** Always drain standing water from the paddy field before Urea top-dressing and re-irrigate only after 24 to 48 hours for optimal root zone absorption and reduced nutrient loss."
            )
        elif "sugarcane" in q_lower or "sugarcane" in top_crop.lower():
            ans = (
                f"🌱 **Recommended Potash (MOP) Split Application Schedule for Sugarcane Ratoon:**\n"
                f"- **Practical Recommendation:** Vanakkam {farmer_name}! Muriate of Potash (MOP) is critical for sugar accumulation, stalk strength, and drought tolerance during dry spells in Sugarcane.\n"
                f"- **Field Execution & Dosages (Per Acre):** Recommended rate: apply approximately **60-80 kg K2O per acre in split doses**. Apply 50% during early tillering (around 30-45 days) and the remaining 50% during earthing up (around 90-120 days after harvest/planting).\n"
                f"- **Practical Field Tip & Precaution:** Place potash fertilizer in furrows near the root zone and irrigate immediately for optimal sugarcane cane elongation and moisture stress tolerance."
            )
        elif "banana" in q_lower or "banana" in top_crop.lower():
            ans = (
                f"🌱 **Phosphorus Fertilizer Management for Banana in Alkaline Soil (pH 8.2):**\n"
                f"- **Practical Recommendation:** Vanakkam {farmer_name}! In alkaline soils (pH > 7.5), phosphorus fixation occurs, making DAP less soluble. Use Single Super Phosphate (SSP) placed directly near the root zone or incorporate organic matter and Farm Yard Manure (FYM).\n"
                f"- **Field Execution & Dosages (Per Acre):** Apply **50 g Single Super Phosphate (SSP) per plant** blended with 5 kg well-rotted FYM in a ring 30-45 cm away from the pseudostem basin.\n"
                f"- **Practical Field Tip & Precaution:** Soil application of Phosphobacteria biofertilizers (20 g/plant) significantly enhances phosphorus solubilization and root uptake under alkaline conditions."
            )
        else:
            p_val = features.get('P', 30)
            n_val = features.get('N', 75)
            k_val = features.get('K', 40)
            ans = (
                f"🌱 **Practical Fertilizer Plan for {top_crop} (Per Acre):**\n"
                f"- **Practical Recommendation:** Vanakkam {farmer_name}! Apply balanced NPK fertilizers calibrated to your soil test values.\n"
                f"- **Field Execution & Dosages:** Basal: {p_val/0.46/2.47:.0f} kg DAP + {k_val*0.5/0.6/2.47:.0f} kg MOP per acre. Top Dressing: {n_val*0.25/0.46/2.47:.0f} kg Neem-coated Urea at active vegetative growth.\n"
                f"- **Practical Field Tip:** Avoid applying fertilizers during heavy rains to prevent nutrient runoff."
            )
    elif any(k in q_lower for k in ["pest", "insect", "disease", "spray", "protect", "fungus", "borer", "worm", "rot"]):
        ans = (
            f"🐛 **Integrated Pest & Disease Defense for {top_crop}:**\n"
            f"- **Preventive Bio-Spray:** Spray **5% Neem Seed Kernel Extract (NSKE)** or Azadirachtin (2 ml/L) every 15 days in early morning.\n"
            f"- **Traps & Monitoring:** Install **6-8 yellow sticky traps per acre** for whiteflies/thrips and **5 pheromone traps** for stem borer.\n"
            f"- **Foliar Disease:** If leaf blast or blight spots appear, apply *Pseudomonas fluorescens* @ 2.5 kg/ha or Trichoderma viride enriched FYM."
        )
    elif any(k in q_lower for k in ["recommend", "what crop", "current", "suitab", "which crop"]):
        ans = (
            f"🌾 **Active Recommended Crop: {top_crop}**\n"
            f"- **Suitability Match:** **{confidence:.0%} match** based on ML agro-climatic assessment.\n"
            f"- **Soil Match:** Calibrated for your pH {features.get('pH', 6.5):.1f}, N {features.get('N', 75)} kg/ha, P {features.get('P', 30)} kg/ha, K {features.get('K', 40)} kg/ha.\n"
            f"- Detailed stage-by-stage timetable is available in your **Cultivation Timeline & Calendar** tab."
        )
    elif any(k in q_lower for k in ["next", "schedule", "when", "today", "tomorrow", "timeline", "calendar"]):
        if next_event:
            ans = (
                f"📅 **Next Scheduled Field Operation:**\n"
                f"- **Milestone:** **{next_event['stage']}** ({next_event['when']} — {next_event['date'].strftime('%a, %d %b %Y')}).\n"
                f"- **Operation:** {next_event['what']}.\n"
                f"- **Recommended Dosage:** {next_event['dosage']}.\n"
                f"- **Field Execution:** {next_event['how']}."
            )
        else:
            ans = "All scheduled cultivation milestones for this season have been completed."
    elif any(k in q_lower for k in ["fertilizer", "nutrient", "urea", "dap", "npk", "dose", "potash"]):
        p_val = features.get('P', 30)
        n_val = features.get('N', 75)
        k_val = features.get('K', 40)
        ans = (
            f"🌱 **Practical Fertilizer Plan for {top_crop} (Per Acre):**\n"
            f"- **Basal Application (At Sowing/Transplanting):** Broadcast **{p_val/0.46/2.47:.0f} kg DAP** + **{k_val*0.5/0.6/2.47:.0f} kg MOP** evenly across the field with a thin layer of water.\n"
            f"- **First Top Dressing (Day 25-30):** Broadcast **{n_val*0.25/0.46/2.47:.0f} kg Neem-coated Urea** after weeding into 2-3 cm standing water.\n"
            f"- **Second Top Dressing (Day 60-70 / Panicle Initiation):** Broadcast **{n_val*0.20/0.46/2.47:.0f} kg Urea** + **{k_val*0.25/0.6/2.47:.0f} kg MOP** to support grain development."
        )
    elif any(k in q_lower for k in ["pest", "insect", "disease", "spray", "protect", "fungus"]):
        ans = (
            f"🐛 **Integrated Pest & Disease Defense for {top_crop}:**\n"
            f"- **Preventive Bio-Spray:** Spray **5% Neem Seed Kernel Extract (NSKE)** or Azadirachtin (2 ml/L) every 15 days in early morning.\n"
            f"- **Traps & Monitoring:** Install **6-8 yellow sticky traps per acre** for whiteflies/thrips and **4 pheromone traps** for stem borer.\n"
            f"- **Foliar Disease:** If leaf blast or blight spots appear, apply *Pseudomonas fluorescens* @ 2.5 kg/ha or Tricyclazole 75 WP @ 120 g/acre."
        )
    elif any(k in q_lower for k in ["vanakkam", "hello", "hi", "help", "who are you", "what can you do"]):
        ans = (
            f"🌾 **Vanakkam {farmer_name}!**\n\n"
            f"I am your dedicated **Agricultural Field Assistant & Agronomy Advisor** for your **{top_crop}** plot in Tamil Nadu.\n\n"
            f"**How can I help with your farming operations today?**\n"
            f"- 🌾 Inquire about optimal fertilizer splits (Urea, DAP, MOP) tailored to your soil test values\n"
            f"- 📅 Check your next upcoming field operation, weeding stage, or AWD water management\n"
            f"- 🐛 Request organic IPM solutions and biological defenses against leaf blast or stem borer\n"
            f"- 📜 Review your farm's past harvest records and highest profit seasons"
        )
    else:
        ans = (
            f"🌾 **Field Advisory Summary for {farmer_name}:**\n"
            f"- **Crop:** {top_crop} ({confidence:.0%} match) tailored to soil pH {features.get('pH', 6.5):.1f}.\n"
            f"- **Field Management:** Follow AWD water management and split fertilizer application in your **Cultivation Timeline & Calendar** tab."
        )

    return {"answer": ans, "sql_info": None}


# ==========================================
# MAIN APPLICATION CONTROLLER
# ==========================================
def main():
    components = load_app_core()
    ensure_baseline_farm_data(components["sql_agent"])

    # --------------------------------------
    # TOP HERO HEADER
    # --------------------------------------
    st.markdown("""
    <div class="brand-header">
        <div>
            <div class="brand-title">🌾 AgriAdvisory AI Platform</div>
            <div class="brand-subtitle">Smart Crop Intelligence • Interactive Cultivation Calendar • Vision Diagnostics • Observability Hub</div>
        </div>
       
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------
    # SIDEBAR: PERSISTENT NAVIGATION & CONTROLS
    # --------------------------------------
    with st.sidebar:
        st.markdown(f"<h3 style='color:{PRIMARY};margin-top:0;'>🧭 Platform Navigation</h3>", unsafe_allow_html=True)
        app_view = st.radio(
            "Select Portal View:",
            ["🌾 Farmer Advisory Portal", "⚙️ System Observability Hub"],
            index=0,
            help="Switch between Farmer Advisory portal and System Observability & RAG sandbox"
        )
        st.markdown("---")

        if app_view == "🌾 Farmer Advisory Portal":
            st.markdown(f"<h4 style='color:{PRIMARY};margin-top:0;'>⚙️ Field & Soil Parameters</h4>", unsafe_allow_html=True)
            st.caption("Default state set to **Tamil Nadu** (data/raw baseline).")

            # Tamil Nadu District / Zone Presets
            tn_district = st.selectbox(
                "📍 Select Tamil Nadu Agro-Climatic Zone:",
                [
                    "Cauvery Delta (Thanjavur, Tiruvarur, Nagapattinam)",
                    "Western Zone (Coimbatore, Erode, Tiruppur)",
                    "Southern Zone (Madurai, Theni, Dindigul)",
                    "Northern Zone (Vellore, Tiruvannamalai, Kanchipuram)",
                    "Custom Field Soil & Climate Settings"
                ],
                index=0
            )

            # Presets calibrated against data/raw state_soil_data.csv and state_weather_data_1997_2020.csv
            preset_defaults = {
                "Cauvery Delta (Thanjavur, Tiruvarur, Nagapattinam)": {"n": 80, "p": 38, "k": 30, "ph": 6.6, "temp": 28.0, "rain": 1200, "hum": 72, "loc": "Thanjavur, Tamil Nadu"},
                "Western Zone (Coimbatore, Erode, Tiruppur)": {"n": 75, "p": 35, "k": 28, "ph": 6.5, "temp": 27.5, "rain": 750, "hum": 68, "loc": "Coimbatore, Tamil Nadu"},
                "Southern Zone (Madurai, Theni, Dindigul)": {"n": 78, "p": 36, "k": 30, "ph": 6.8, "temp": 28.5, "rain": 850, "hum": 70, "loc": "Madurai, Tamil Nadu"},
                "Northern Zone (Vellore, Tiruvannamalai, Kanchipuram)": {"n": 82, "p": 40, "k": 32, "ph": 6.7, "temp": 28.0, "rain": 1100, "hum": 71, "loc": "Vellore, Tamil Nadu"},
                "Custom Field Soil & Climate Settings": {"n": 80, "p": 38, "k": 30, "ph": 6.6, "temp": 28.0, "rain": 1200, "hum": 72, "loc": "Tamil Nadu, India"}
            }
            active_p = preset_defaults[tn_district]

            with st.expander("🧪 Soil Nutrients (NPK)", expanded=True):
                n = st.slider("Nitrogen (N) kg/ha", 0, 200, active_p["n"], step=5, help="Available soil nitrogen (Tamil Nadu baseline: 80 kg/ha)")
                p = st.slider("Phosphorus (P) kg/ha", 0, 100, active_p["p"], step=2, help="Available soil phosphorus (Tamil Nadu baseline: 38 kg/ha)")
                k = st.slider("Potassium (K) kg/ha", 0, 100, active_p["k"], step=2, help="Available soil potassium (Tamil Nadu baseline: 30 kg/ha)")
                ph = st.slider("Soil pH", 4.5, 9.0, active_p["ph"], step=0.1, help="Acidity/alkalinity measure (Tamil Nadu baseline: 6.6)")

            with st.expander("🌦️ Climate & Weather", expanded=True):
                temp = st.slider("Temperature (°C)", 10.0, 45.0, active_p["temp"], step=0.5)
                rainfall = st.slider("Rainfall (mm)", 200, 2500, active_p["rain"], step=50)
                humidity = st.slider("Humidity (%)", 20, 100, active_p["hum"], step=5)

            with st.expander("📅 Season & Farmer Context", expanded=True):
                season_choice = st.radio("Cropping Season", ["Kharif (Kuruvai / Samba)", "Rabi (Thaladi / Navarai)"], index=0)
                season = "Kharif" if "Kharif" in season_choice else "Rabi"
                farmer_name = st.text_input("Farmer Name", value="Murugan (Farmer, Tamil Nadu)")
                location = st.text_input("Farm Location", value=active_p["loc"])
                farm_size = st.number_input("Land Area (ha)", min_value=0.5, max_value=50.0, value=5.0, step=0.5, help="1 ha = 2.47 acres")

            season_k = 1 if season == "Kharif" else 0
            features = {
                "N": n, "P": p, "K": k, "pH": ph,
                "avg_temp_c": temp, "total_rainfall_mm": rainfall,
                "avg_humidity_percent": humidity,
                "season_Kharif": season_k, "season_Rabi": 1 - season_k
            }
        else:
            st.markdown(f"<h4 style='color:{PRIMARY};margin-top:0;'>⚙️ System Telemetry</h4>", unsafe_allow_html=True)
            rag_stats = components["retriever"].get_collection_stats()
            st.metric("Vectorstore Chunks", f"{rag_stats.get('total_chunks', 2892):,}")
            st.metric("Policy Documents", f"{rag_stats.get('total_documents', 4)}")
            st.metric("Embedding Device", rag_stats.get("device", "CPU"))
            st.metric("Database File", "data/farmer_profile.db")
            st.caption("Active models: ExtraTreesClassifier • CalibratedCV • ResNet-18")

            features = {
                "N": 80, "P": 38, "K": 30, "pH": 6.6,
                "avg_temp_c": 28.0, "total_rainfall_mm": 1200,
                "avg_humidity_percent": 72,
                "season_Kharif": 1, "season_Rabi": 0
            }
            n, p, k, ph = 80, 38, 30, 6.6
            temp, rainfall, humidity = 28.0, 1200, 72
            season = "Kharif"
            farmer_name = "Murugan (Farmer, Tamil Nadu)"
            location = "Thanjavur, Tamil Nadu"
            farm_size = 5.0

    # =========================================================================
    # VIEW ROUTING: 1. FARMER PORTAL vs 2. SYSTEM OBSERVABILITY HUB
    # =========================================================================

    if app_view == "🌾 Farmer Advisory Portal":
        # ---------------------------------------------------------------------
        # REAL-TIME ML PREDICTION (FAST <1ms PREDICTION)
        # ---------------------------------------------------------------------
        prediction = components["ml_engine"].predict(features)
        top_crop = prediction.get("top_crop", "Rice")
        confidence = prediction.get("overall_confidence", 0.85)
        alternatives = prediction.get("predictions", [])[1:3]

        # Sowing Date Anchor for Calendar Calculations
        sowing_date = datetime.today().date()
        calendar_events = get_crop_calendar_events(
            crop=top_crop,
            season=season,
            n=n, p=p, k=k, ph=ph,
            sowing_date=sowing_date
        )

        # Cached Advisory Narrative (Instant Rerun)
        adv_text = get_cached_farmer_advisory(
            crop=top_crop,
            n=n, p=p, k=k, ph=ph,
            temp=temp, rain=rainfall, hum=humidity,
            season=season, farmer_name=farmer_name, location=location, farm_size=farm_size
        )
        companion_text = get_cached_companion_text(top_crop)

        # ---------------------------------------------------------------------
        # FARMER NAVIGATION TABS
        # ---------------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Farmer Advisory Report",
            "📅 Cultivation Timeline & Calendar",
            "💬 Farm Assistant AI",
            "🔍 Crop Doctor Vision",
            "📊 Farm History Analytics"
        ])

        # =====================================================================
        # TAB 1: ADVISORY REPORT (100% FARMER-FOCUSED & UNCLUTTERED)
        # =====================================================================
        with tab1:
            kpi1, kpi2, kpi3, kpi4 = st.columns([1.3, 1, 1.4, 0.9])
            with kpi1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Recommended Crop</div>
                    <div class="kpi-value">{top_crop}</div>
                    <div class="kpi-subtext">Tailored for {location} ({season})</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Field Match Score</div>
                    <div class="kpi-value">{confidence:.1%}</div>
                    <div class="kpi-subtext">High Agro-Climatic Suitability</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi3:
                alt_names = ", ".join([f"{p['crop']} ({p['confidence_pct']:.0f}%)" for p in alternatives]) if alternatives else "None"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Alternative Crop Options</div>
                    <div class="kpi-value" style="font-size:1.3rem;">{alt_names}</div>
                    <div class="kpi-subtext">Viable rotation crops for this plot</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi4:
                # PDF Generation compiled fast on demand with complete 18-stage operational checklist
                pdf_buffer = components["pdf_builder"].generate_advisory_pdf(
                    farmer_name=farmer_name,
                    location=location,
                    farm_size=farm_size,
                    features=features,
                    advisory_data={
                        "crop": top_crop,
                        "confidence": confidence,
                        "alternative_crops": [p["crop"] for p in alternatives],
                        "advisory": adv_text,
                        "timetable": calendar_events
                    },
                    ml_metrics={"predictions": prediction.get("predictions", [])},
                    timetable_data=calendar_events,
                    companion_data=companion_text
                )

                st.markdown("""
                <div class="kpi-card" style="display:flex;flex-direction:column;justify-content:center;align-items:center;">
                    <div class="kpi-label">Export Advisory</div>
                """, unsafe_allow_html=True)
                st.download_button(
                    label="📥 Download PDF Dossier",
                    data=pdf_buffer.getvalue(),
                    file_name=f"crop_advisory_{top_crop.lower()}_{location.split(',')[0].strip().lower()}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

            # Full-Width Agricultural Advisory Report Panel
            st.markdown(f"""
            <div class="content-panel">
                <div class="panel-header">🌾 Detailed Agricultural Advisory Report for {farmer_name}</div>
            """, unsafe_allow_html=True)
            st.markdown(adv_text)
            st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================================
        # TAB 2: CULTIVATION TIMELINE, INTERACTIVE CALENDAR & CHECKLIST
        # =====================================================================
        with tab2:
            # 1. State Retrieval & Dynamic Calculations
            selected_sowing_date = st.session_state.get("cal_sowing_date_picker", sowing_date)
            treatment_mode = st.session_state.get("cal_treatment_mode_radio", "Integrated Pest Management (IPM) ⚖️")
            auto_adjust_skipped = st.session_state.get("auto_adjust_skipped_enabled", True)

            # Recalculate event dates based on sowing date
            all_events = get_crop_calendar_events(
                crop=top_crop,
                season=season,
                n=n, p=p, k=k, ph=ph,
                sowing_date=selected_sowing_date
            )

            # Initialize task statuses if not set
            for ev in all_events:
                s_key = f"op_status_{top_crop}_{ev['day_offset']}"
                if s_key not in st.session_state:
                    st.session_state[s_key] = "done" if ev["day_offset"] <= -7 else "pending"
                st.session_state[f"chk_op_done_{top_crop}_{ev['day_offset']}"] = (st.session_state[s_key] == "done")

            # Rolled-forward calculation for skipped mandatory operations
            rolled_adjustments_by_day = {}
            if auto_adjust_skipped:
                pending_rollovers = []
                for ev in all_events:
                    st_val = st.session_state.get(f"op_status_{top_crop}_{ev['day_offset']}", "pending")
                    is_mand = (ev["priority"] in ["Critical", "High"]) or (ev["category"] in ["Nutrition", "Weed Management", "Irrigation"])
                    if st_val == "skipped" and is_mand:
                        pending_rollovers.append({
                            "from_stage": ev["stage"],
                            "from_when": ev["when"],
                            "from_day": ev["day_offset"],
                            "dosage": ev["dosage"],
                            "what": ev["what"],
                            "category": ev["category"]
                        })
                    elif st_val != "skipped" and pending_rollovers:
                        rolled_adjustments_by_day[ev["day_offset"]] = list(pending_rollovers)
                        pending_rollovers.clear()
                if pending_rollovers and all_events:
                    last_day = all_events[-1]["day_offset"]
                    rolled_adjustments_by_day.setdefault(last_day, []).extend(pending_rollovers)

            # Progress Tracking KPI metrics
            total_tasks = len(all_events)
            completed_tasks = sum(1 for e in all_events if st.session_state.get(f"op_status_{top_crop}_{e['day_offset']}") == "done")
            skipped_tasks = sum(1 for e in all_events if st.session_state.get(f"op_status_{top_crop}_{e['day_offset']}") == "skipped")
            pending_tasks = total_tasks - completed_tasks - skipped_tasks
            progress_pct = completed_tasks / total_tasks if total_tasks > 0 else 0.0

            # -----------------------------------------------------------------
            # 📊 OPERATIONS PROGRESS TRACKER (PLACED AT TOP OF TAB)
            # -----------------------------------------------------------------
            st.markdown(f"""
            <div style='background:white;border:1.5px solid {BORDER_COLOR};border-radius:12px;padding:12px 18px;box-shadow:0 2px 6px rgba(0,0,0,0.03);margin-bottom:14px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
                    <div style='font-weight:800;color:{PRIMARY};font-size:1.1rem;'>📊 Operations Progress Tracker</div>
                    <span style='font-size:0.78rem;background:#E8F5E9;color:{PRIMARY};padding:2px 10px;border-radius:12px;font-weight:700;'>
                        {top_crop} • {season} Cycle
                    </span>
                </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Operations", f"{total_tasks} Stages")
            with m2:
                st.metric("✅ Completed", f"{completed_tasks} Done", delta=f"{progress_pct:.0%}")
            with m3:
                st.metric("⏳ Pending Action", f"{pending_tasks} Stages")
            with m4:
                adjust_label = " (Adjusted)" if (auto_adjust_skipped and skipped_tasks > 0) else ""
                st.metric("⏭️ Skipped", f"{skipped_tasks}{adjust_label}")

            st.progress(progress_pct)
            st.markdown("</div>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # CALENDAR (LEFT) & CROPPING PARAMETERS (RIGHT)
            # -----------------------------------------------------------------
            col_calendar, col_controls = st.columns([1.62, 1.0], gap="large")

            with col_controls:
                st.markdown(f"""
                <div style='background:#F8FAFC;border:1.5px solid {BORDER_COLOR};border-radius:12px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,0.03);margin-bottom:12px;'>
                    <div style='font-weight:800;color:{PRIMARY};font-size:1.02rem;margin-bottom:10px;'>⚙️ Cropping Parameters & Controls</div>
                """, unsafe_allow_html=True)

                selected_sowing_date = st.date_input(
                    "🌾 Sowing / Cycle Start Date:",
                    value=selected_sowing_date,
                    key="cal_sowing_date_picker"
                )

                treatment_mode = st.radio(
                    "🌿 Agronomic Protocol:",
                    ["Integrated Pest Management (IPM) ⚖️", "Organic & Regenerative 🌱", "Conventional / Chemical ⚗️"],
                    key="cal_treatment_mode_radio"
                )

                st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                auto_adjust_skipped = st.toggle(
                    "🔄 Auto-Adjust Skipped Mandatory Operations",
                    value=st.session_state.get("auto_adjust_skipped_enabled", True),
                    key="auto_adjust_skipped_enabled",
                    help="When enabled, if you mark a mandatory operation (like Basal NPK, Weeding, or Top Dressing) as Skipped, its dosage and duties are automatically scheduled and rolled forward into the upcoming operational milestone."
                )

                status_note = "Active (Tasks roll forward)" if auto_adjust_skipped else "Disabled (Fixed schedule)"
                st.markdown(f"""
                <div style='background:white;border:1px solid #E2E8F0;border-radius:8px;padding:10px;margin-top:10px;font-size:0.78rem;color:#475569;'>
                    <b>Active Protocol:</b> {treatment_mode.split()[0]}<br/>
                    <b>Auto-Adjustment:</b> {status_note}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Render Calendar View on Left Side
            chosen_y = selected_sowing_date.year
            chosen_m = selected_sowing_date.month
            month_events = []

            with col_calendar:
                st.markdown(f"<h4 style='color:{PRIMARY};margin:0 0 6px 0;font-size:1.15rem;font-weight:800;'>📅 Cultivation Calendar</h4>", unsafe_allow_html=True)
                cal_view_mode = st.radio(
                    "Select Calendar View Mode:",
                    ["📆 Monthly Calendar", "🗓️ Weekly Progression", "📋 Daily Stage Overview"],
                    horizontal=True,
                    key="cal_view_mode_selector_radio"
                )

                if cal_view_mode == "📆 Monthly Calendar":
                    month_map = {}
                    for ev in all_events:
                        ym = (ev["date"].year, ev["date"].month)
                        if ym not in month_map:
                            month_map[ym] = []
                        month_map[ym].append(ev)

                    month_keys = sorted(month_map.keys())
                    month_options = [
                        f"{calendar.month_name[m]} {y} ({len(month_map[(y, m)])} Operations Scheduled)"
                        for y, m in month_keys
                    ]

                    # Default to month matching sowing date if available
                    default_m_idx = 0
                    for idx, (y, m) in enumerate(month_keys):
                        if (y, m) == (selected_sowing_date.year, selected_sowing_date.month):
                            default_m_idx = idx
                            break

                    c_m_sel, c_m_legend = st.columns([1.8, 1.2])
                    with c_m_sel:
                        sel_m_idx = st.selectbox(
                            "Select Month to Inspect:",
                            range(len(month_keys)),
                            index=default_m_idx,
                            format_func=lambda idx: month_options[idx],
                            key="sel_month_dropdown"
                        )
                    with c_m_legend:
                        st.html("""
                        <div style='display:flex;gap:6px;align-items:center;margin-top:24px;font-size:0.72rem;font-weight:700;'>
                            <span style='background:#DCFCE7;color:#166534;border:1px solid #16A34A;padding:2px 6px;border-radius:4px;'>✅ Done</span>
                            <span style='background:#FEF3C7;color:#92400E;border:1px solid #D97706;padding:2px 6px;border-radius:4px;'>⏳ Action</span>
                            <span style='background:#FEE2E2;color:#991B1B;border:1px solid #DC2626;padding:2px 6px;border-radius:4px;'>⏭️ Skip</span>
                        </div>
                        """)

                    chosen_y, chosen_m = month_keys[sel_m_idx]
                    month_events = month_map[(chosen_y, chosen_m)]
                    events_by_day = {e["date"].day: e for e in month_events}

                    # Build calendar grid HTML with color differentiation
                    cal_matrix = calendar.monthcalendar(chosen_y, chosen_m)
                    rows_html = []
                    for week in cal_matrix:
                        tds_html = []
                        for day_num in week:
                            if day_num == 0:
                                tds_html.append("<td style='background:#F8FAFC;border:1px solid #E2E8F0;padding:5px;height:72px;vertical-align:top;'></td>")
                            elif day_num in events_by_day:
                                ev_m = events_by_day[day_num]
                                cur_st = st.session_state.get(f"op_status_{top_crop}_{ev_m['day_offset']}", "pending")
                                if cur_st == "done":
                                    c_bg = "#DCFCE7"
                                    c_border = "2px solid #16A34A"
                                    c_tag = "✅ Done"
                                    c_tag_bg = "#16A34A"
                                    p_bg = "#15803D"
                                elif cur_st == "skipped":
                                    c_bg = "#FEE2E2"
                                    c_border = "2px solid #DC2626"
                                    c_tag = "⏭️ Skip"
                                    c_tag_bg = "#DC2626"
                                    p_bg = "#B91C1C"
                                else:
                                    c_bg = "#FEF3C7"
                                    c_border = "2px solid #D97706"
                                    c_tag = "⏳ Action"
                                    c_tag_bg = "#D97706"
                                    p_bg = "#B45309"

                                stg = html.escape(ev_m['stage'])
                                wht = html.escape(ev_m['what'])
                                tds_html.append(
                                    f"<td style='background:{c_bg};border:{c_border};padding:5px;height:72px;vertical-align:top;'>"
                                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                                    f"<span style='font-weight:800;color:#1E293B;font-size:0.88rem;'>{day_num}</span>"
                                    f"<span style='font-size:0.62rem;background:{c_tag_bg};color:white;font-weight:700;padding:1px 4px;border-radius:4px;'>{c_tag}</span>"
                                    f"</div>"
                                    f"<div style='background:{p_bg};color:white;padding:2px 4px;border-radius:4px;font-size:0.68rem;font-weight:600;margin-top:3px;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;' title='{wht}'>"
                                    f"{stg}</div>"
                                    f"</td>"
                                )
                            else:
                                tds_html.append(
                                    f"<td style='background:white;border:1px solid #E2E8F0;padding:5px;height:72px;vertical-align:top;'>"
                                    f"<div style='font-weight:700;color:#64748B;font-size:0.82rem;'>{day_num}</div>"
                                    f"</td>"
                                )
                        rows_html.append(f"<tr>{''.join(tds_html)}</tr>")

                    table_calendar_html = (
                        f"<div style='margin-bottom:0.6rem;box-shadow:0 2px 10px rgba(0,0,0,0.04);border-radius:10px;overflow:hidden;border:1px solid #CBD5E1;'>"
                        f"<div style='background:linear-gradient(135deg, {PRIMARY}, {SECONDARY});color:white;padding:8px 14px;font-weight:700;font-size:0.98rem;display:flex;justify-content:space-between;align-items:center;'>"
                        f"<span>📅 {calendar.month_name[chosen_m]} {chosen_y} — Cropping Schedule</span>"
                        f"<span style='font-size:0.78rem;background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;'>{len(month_events)} Scheduled Events</span>"
                        f"</div>"
                        f"<table style='width:100%;border-collapse:collapse;table-layout:fixed;background:white;'>"
                        f"<thead><tr style='background:#E8F5E9;color:{PRIMARY};font-weight:700;font-size:0.8rem;text-align:center;'>"
                        f"<th style='padding:6px 2px;width:14.28%;'>Mon</th>"
                        f"<th style='padding:6px 2px;width:14.28%;'>Tue</th>"
                        f"<th style='padding:6px 2px;width:14.28%;'>Wed</th>"
                        f"<th style='padding:6px 2px;width:14.28%;'>Thu</th>"
                        f"<th style='padding:6px 2px;width:14.28%;'>Fri</th>"
                        f"<th style='padding:6px 2px;width:14.28%;'>Sat</th>"
                        f"<th style='padding:6px 2px;width:14.28%;'>Sun</th>"
                        f"</tr></thead>"
                        f"<tbody>{''.join(rows_html)}</tbody>"
                        f"</table>"
                        f"</div>"
                    )
                    st.html(table_calendar_html)

                elif cal_view_mode == "🗓️ Weekly Progression":
                    st.caption("Chronological weekly stages from land preparation through harvest:")
                    weekly_groups = [
                        ("Pre-Season Week -2 (Day -14 to -8)", [e for e in all_events if -14 <= e["day_offset"] <= -8]),
                        ("Pre-Season Week -1 (Day -7 to -1)", [e for e in all_events if -7 <= e["day_offset"] <= -1]),
                        ("Week 1: Sowing & Seed Priming (Day 0 to 6)", [e for e in all_events if 0 <= e["day_offset"] <= 6]),
                        ("Week 2: Seedling Establishment (Day 7 to 13)", [e for e in all_events if 7 <= e["day_offset"] <= 13]),
                        ("Week 3: Canopy Growth & Gap Filling (Day 14 to 20)", [e for e in all_events if 14 <= e["day_offset"] <= 20]),
                        ("Week 4: Weeding & Inter-Cultivation (Day 21 to 27)", [e for e in all_events if 21 <= e["day_offset"] <= 27]),
                        ("Week 5: First Top Dressing & Irrigation (Day 28 to 34)", [e for e in all_events if 28 <= e["day_offset"] <= 34]),
                        ("Weeks 6 - 7: Active Branching & Micronutrients (Day 35 to 48)", [e for e in all_events if 35 <= e["day_offset"] <= 48]),
                        ("Weeks 8 - 9: Vegetative Canopy & Pest Scouting (Day 49 to 62)", [e for e in all_events if 49 <= e["day_offset"] <= 62]),
                        ("Weeks 10 - 11: Flowering & Second Top Dressing (Day 63 to 76)", [e for e in all_events if 63 <= e["day_offset"] <= 76]),
                        ("Weeks 12 - 14: Grain/Pod Filling Stage (Day 77 to 95)", [e for e in all_events if 77 <= e["day_offset"] <= 95]),
                        ("Weeks 15 - 16: Pre-Harvest Drainage & Maturity (Day 96 to 110)", [e for e in all_events if 96 <= e["day_offset"] <= 110]),
                        ("Weeks 17+: Harvesting & Hermetic Storage (Day 111+)", [e for e in all_events if e["day_offset"] >= 111]),
                    ]

                    for w_idx, (w_title, w_events) in enumerate(weekly_groups):
                        if not w_events:
                            continue
                        w_all_done = all(st.session_state.get(f"op_status_{top_crop}_{e['day_offset']}") == "done" for e in w_events)
                        w_has_skip = any(st.session_state.get(f"op_status_{top_crop}_{e['day_offset']}") == "skipped" for e in w_events)
                        w_badge = "✅ Completed" if w_all_done else ("⏭️ Has Skipped Tasks" if w_has_skip else "⏳ Pending Action")
                        w_badge_bg = "#DCFCE7" if w_all_done else ("#FEE2E2" if w_has_skip else "#FEF3C7")
                        w_badge_col = "#166534" if w_all_done else ("#991B1B" if w_has_skip else "#92400E")
                        date_span = f"{w_events[0]['date'].strftime('%b %d')} — {w_events[-1]['date'].strftime('%b %d')}"

                        with st.expander(f"📆 {w_title} [{date_span}]", expanded=(w_idx == 2 or (not w_all_done and w_idx < 4))):
                            st.html(f"<span style='background:{w_badge_bg};color:{w_badge_col};padding:3px 8px;border-radius:6px;font-size:0.75rem;font-weight:700;'>{w_badge}</span>")
                            for ev in w_events:
                                ev_st = st.session_state.get(f"op_status_{top_crop}_{ev['day_offset']}", "pending")
                                t_icon = "✅" if ev_st == "done" else ("⏭️" if ev_st == "skipped" else "⏳")
                                st.markdown(f"**{t_icon} {ev['stage']}: {ev['what']}** ({ev['date'].strftime('%a, %d %b')})")
                                st.caption(f"**Dosage:** {ev['dosage']} | **How:** {ev['how']}")

                else:  # Daily Stage Overview
                    st.caption(f"Chronological timeline overview for {top_crop} across all 18 lifecycle milestones:")
                    for ev in all_events:
                        ev_st = st.session_state.get(f"op_status_{top_crop}_{ev['day_offset']}", "pending")
                        d_badge = "✅ Completed" if ev_st == "done" else ("⏭️ Skipped" if ev_st == "skipped" else "⏳ Pending")
                        d_bg = "#DCFCE7" if ev_st == "done" else ("#FEE2E2" if ev_st == "skipped" else "#FEF3C7")
                        p_col = "#991B1B" if ev["priority"] == "Critical" else "#1B4D3E"
                        st.html(
                            f"<div style='background:white;border:1px solid #E2E8F0;border-left:4px solid {p_col};border-radius:6px;padding:8px 12px;margin-bottom:6px;'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                            f"<div><b style='color:{PRIMARY};'>{ev['when']}</b> ({ev['date'].strftime('%a, %d %b %Y')}) — <b>{ev['stage']}</b></div>"
                            f"<span style='font-size:0.72rem;background:{d_bg};padding:2px 6px;border-radius:4px;font-weight:700;'>{d_badge}</span>"
                            f"</div>"
                            f"<div style='font-size:0.85rem;color:{TEXT_MAIN};margin-top:2px;'>{ev['what']}</div>"
                            f"<div style='font-size:0.78rem;color:#166534;margin-top:2px;'><b>Dosage:</b> {ev['dosage']}</div>"
                            f"</div>"
                        )

            # -----------------------------------------------------------------
            # 🎯 MILESTONE INTERACTIVE ACTION CARDS (SQUARE CARDS IN RESPONSIVE GRID)
            # -----------------------------------------------------------------
            st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
            active_milestones = month_events if (cal_view_mode == "📆 Monthly Calendar" and month_events) else all_events[:6]
            month_label = f"in {calendar.month_name[chosen_m]} {chosen_y}" if cal_view_mode == "📆 Monthly Calendar" else "Active / Upcoming"

            st.markdown(f"""
            <div style="background:white;padding:1.1rem 1.4rem;border-radius:12px;border:1px solid {BORDER_COLOR};margin-bottom:1rem;box-shadow:0 2px 6px rgba(0,0,0,0.03);">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <h4 style="color:{PRIMARY};margin:0;font-size:1.18rem;font-weight:800;">🎯 Milestone Interactive Action Cards</h4>
                        <p style="color:{TEXT_MUTED};margin:0.2rem 0 0 0;font-size:0.88rem;">
                            Scheduled operations for <b>{top_crop}</b> {month_label}. Click <b>Done</b>, <b>Skip</b>, or <b>Reset</b> to update status and auto-adjust requirements.
                        </p>
                    </div>
                    <span style="background:#E8F5E9;color:{PRIMARY};padding:4px 12px;border-radius:20px;font-size:0.82rem;font-weight:700;">{len(active_milestones)} Milestone Cards</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not active_milestones:
                st.info(f"No operations scheduled {month_label}.")
            else:
                m_grid_cols = 3
                for r_idx in range(0, len(active_milestones), m_grid_cols):
                    m_chunk = active_milestones[r_idx : r_idx + m_grid_cols]
                    cols = st.columns(m_grid_cols)
                    for col, card_ev in zip(cols, m_chunk):
                        with col:
                            card_st = st.session_state.get(f"op_status_{top_crop}_{card_ev['day_offset']}", "pending")
                            card_border = "#10B981" if card_st == "done" else ("#EF4444" if card_st == "skipped" else "#F59E0B")
                            card_bg = "#F0FDF4" if card_st == "done" else ("#FEF2F2" if card_st == "skipped" else "#FFFBEB")
                            badge_lbl = "✅ COMPLETED" if card_st == "done" else ("⏭️ SKIPPED" if card_st == "skipped" else "⏳ ACTION PENDING")
                            badge_color = "#15803D" if card_st == "done" else ("#B91C1C" if card_st == "skipped" else "#B45309")

                            # Check if this day has rolled forward adjustments
                            rolled_notes = rolled_adjustments_by_day.get(card_ev["day_offset"], [])
                            rolled_html = ""
                            if rolled_notes:
                                items_html = "".join([
                                    f"<li><b>From Day {rn['from_day']} ({rn['from_stage']}):</b> {rn['what']} — <i>{rn['dosage']}</i></li>"
                                    for rn in rolled_notes
                                ])
                                rolled_html = (
                                    f"<div style='background:#FFF1F2;border:1.5px solid #F43F5E;border-radius:8px;padding:8px 10px;margin:8px 0;'>"
                                    f"<div style='color:#BE123C;font-weight:700;font-size:0.78rem;'>⚠️ Mandatory Requirements Rolled Forward:</div>"
                                    f"<ul style='margin:2px 0 0 0;padding-left:16px;color:#881337;font-size:0.75rem;'>{items_html}</ul>"
                                    f"</div>"
                                )

                            protocol_spec = ""
                            if "Organic" in treatment_mode:
                                protocol_spec = f"<div style='background:#ECFDF5;padding:6px 10px;border-radius:6px;border:1px solid #A7F3D0;margin-top:6px;font-size:0.76rem;color:#065F46;'><b>🌱 Organic Protocol:</b> {card_ev['organic']}</div>"
                            elif "Conventional" in treatment_mode:
                                protocol_spec = f"<div style='background:#EFF6FF;padding:6px 10px;border-radius:6px;border:1px solid #BFDBFE;margin-top:6px;font-size:0.76rem;color:#1E40AF;'><b>⚗️ Chemical Protocol:</b> {card_ev['chemical']}</div>"
                            else:
                                protocol_spec = (
                                    f"<div style='background:#ECFDF5;padding:6px 10px;border-radius:6px;border:1px solid #A7F3D0;margin-top:6px;font-size:0.76rem;color:#065F46;'><b>🌱 Biological:</b> {card_ev['organic']}</div>"
                                    f"<div style='background:#EFF6FF;padding:6px 10px;border-radius:6px;border:1px solid #BFDBFE;margin-top:4px;font-size:0.76rem;color:#1E40AF;'><b>⚗️ Chemical:</b> {card_ev['chemical']}</div>"
                                )

                            st.html(
                                f"<div style='background:{card_bg};border:2px solid {card_border};border-radius:12px;padding:14px;min-height:430px;display:flex;flex-direction:column;justify-content:space-between;box-shadow:0 2px 8px rgba(0,0,0,0.04);margin-bottom:8px;'>"
                                f"<div>"
                                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
                                f"<span style='font-size:0.75rem;background:#E2E8F0;color:#334155;padding:2px 7px;border-radius:4px;font-weight:700;'>{card_ev['phase']}</span>"
                                f"<span style='font-size:0.75rem;background:{badge_color};color:white;padding:2px 8px;border-radius:6px;font-weight:800;'>{badge_lbl}</span>"
                                f"</div>"
                                f"<div style='display:flex;gap:5px;margin-bottom:6px;'>"
                                f"<span style='font-size:0.7rem;background:#E8F5E9;color:#1B4D3E;padding:1px 6px;border-radius:4px;font-weight:600;'>{card_ev['category']}</span>"
                                f"<span style='font-size:0.7rem;background:#FEE2E2;color:#991B1B;padding:1px 6px;border-radius:4px;font-weight:700;'>{card_ev['priority']} Priority</span>"
                                f"</div>"
                                f"<h5 style='color:{PRIMARY};margin:4px 0 2px 0;font-size:0.95rem;font-weight:800;'>📍 {card_ev['stage']}</h5>"
                                f"<div style='color:{PRIMARY};font-size:0.8rem;font-weight:700;margin-bottom:6px;'>🗓️ {card_ev['when']} ({card_ev['date'].strftime('%a, %d %b')})</div>"
                                f"<div style='font-weight:600;color:#1E293B;font-size:0.84rem;margin-bottom:6px;line-height:1.3;'>{card_ev['what']}</div>"
                                f"{rolled_html}"
                                f"<div style='background:white;padding:8px;border-radius:6px;border:1px solid #E2E8F0;margin-top:6px;font-size:0.78rem;color:#166534;line-height:1.3;'>"
                                f"<b>🎯 Dose:</b> {card_ev['dosage']}</div>"
                                f"<div style='background:white;padding:8px;border-radius:6px;border:1px solid #E2E8F0;margin-top:5px;font-size:0.76rem;color:#334155;line-height:1.3;'>"
                                f"<b>🛠️ How:</b> {card_ev['how']}</div>"
                                f"{protocol_spec}"
                                f"</div>"
                                f"</div>"
                            )

                            # Interactive buttons directly inside this square card column
                            c_btn1, c_btn2, c_btn3 = st.columns(3)
                            with c_btn1:
                                if st.button("✅ Done", key=f"mcard_done_{top_crop}_{chosen_y}_{chosen_m}_{card_ev['day_offset']}", disabled=(card_st == "done"), use_container_width=True):
                                    st.session_state[f"op_status_{top_crop}_{card_ev['day_offset']}"] = "done"
                                    st.session_state[f"chk_op_done_{top_crop}_{card_ev['day_offset']}"] = True
                                    st.toast(f"Marked {card_ev['stage']} as Completed! ✅")
                                    st.rerun()
                            with c_btn2:
                                if st.button("⏭️ Skip", key=f"mcard_skip_{top_crop}_{chosen_y}_{chosen_m}_{card_ev['day_offset']}", disabled=(card_st == "skipped"), use_container_width=True):
                                    st.session_state[f"op_status_{top_crop}_{card_ev['day_offset']}"] = "skipped"
                                    st.session_state[f"chk_op_done_{top_crop}_{card_ev['day_offset']}"] = False
                                    if auto_adjust_skipped:
                                        st.toast("Marked as Skipped! ⚠️ Requirements rolled forward to upcoming stage.")
                                    else:
                                        st.toast(f"Marked {card_ev['stage']} as Skipped. ⏭️")
                                    st.rerun()
                            with c_btn3:
                                if st.button("⏳ Reset", key=f"mcard_reset_{top_crop}_{chosen_y}_{chosen_m}_{card_ev['day_offset']}", disabled=(card_st == "pending"), use_container_width=True):
                                    st.session_state[f"op_status_{top_crop}_{card_ev['day_offset']}"] = "pending"
                                    st.session_state[f"chk_op_done_{top_crop}_{card_ev['day_offset']}"] = False
                                    st.toast(f"Reset {card_ev['stage']} to Pending. ⏳")
                                    st.rerun()

            # -----------------------------------------------------------------
            # 📋 UNIFIED OPERATIONAL ACTION CHECKLIST & SCHEDULED MILESTONES
            # (RESPONSIVE 3-COLUMN GRID OF SQUARE CARDS)
            # -----------------------------------------------------------------
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:white;padding:1.2rem 1.6rem;border-radius:12px;border:1px solid {BORDER_COLOR};margin-bottom:1rem;box-shadow:0 2px 6px rgba(0,0,0,0.03);">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <h3 style="color:{PRIMARY};margin:0;font-size:1.25rem;font-weight:800;">📋 Unified Operational Action Checklist & Scheduled Milestones</h3>
                        <p style="color:{TEXT_MUTED};margin:0.25rem 0 0 0;font-size:0.9rem;">
                            Complete 18-stage lifecycle tracker for <b>{top_crop}</b> ({season}) arranged in responsive operational cards with per-acre dosages.
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            f_col1, f_col2 = st.columns(2)
            with f_col1:
                status_filter = st.selectbox(
                    "Filter by Status:",
                    ["All Operations", "⏳ Pending Tasks Only", "✅ Completed Tasks Only", "⏭️ Skipped Tasks Only"],
                    key="chk_filter_status_sel"
                )
            with f_col2:
                cat_filter = st.selectbox(
                    "Filter by Category:",
                    ["All Categories", "Nutrition", "Weed Management", "Irrigation", "Pest & Disease Defense", "Harvesting & Storage", "Soil & Land Prep", "Sowing"],
                    key="chk_filter_cat_sel"
                )

            filtered_events = all_events
            if status_filter == "⏳ Pending Tasks Only":
                filtered_events = [e for e in filtered_events if st.session_state.get(f"op_status_{top_crop}_{e['day_offset']}", "pending") == "pending"]
            elif status_filter == "✅ Completed Tasks Only":
                filtered_events = [e for e in filtered_events if st.session_state.get(f"op_status_{top_crop}_{e['day_offset']}", "pending") == "done"]
            elif status_filter == "⏭️ Skipped Tasks Only":
                filtered_events = [e for e in filtered_events if st.session_state.get(f"op_status_{top_crop}_{e['day_offset']}", "pending") == "skipped"]

            if cat_filter != "All Categories":
                filtered_events = [e for e in filtered_events if e["category"].lower() == cat_filter.lower()]

            if not filtered_events:
                st.info("No operations match the selected filters.")
            else:
                grid_cols = 3
                for r_idx in range(0, len(filtered_events), grid_cols):
                    r_chunk = filtered_events[r_idx : r_idx + grid_cols]
                    cols = st.columns(grid_cols)
                    for col, ev in zip(cols, r_chunk):
                        with col:
                            ev_st = st.session_state.get(f"op_status_{top_crop}_{ev['day_offset']}", "pending")
                            p_col = "#991B1B" if ev["priority"] == "Critical" else "#1B4D3E"
                            border_col = "#10B981" if ev_st == "done" else ("#EF4444" if ev_st == "skipped" else "#F59E0B")
                            bg_col = "#F0FDF4" if ev_st == "done" else ("#FEF2F2" if ev_st == "skipped" else "white")
                            done_lbl = "✅ Completed" if ev_st == "done" else ("⏭️ Skipped" if ev_st == "skipped" else "⏳ Pending")
                            badge_pill_bg = "#16A34A" if ev_st == "done" else ("#DC2626" if ev_st == "skipped" else "#D97706")

                            # Check for rolled adjustments
                            r_adj = rolled_adjustments_by_day.get(ev["day_offset"], [])
                            r_html = ""
                            if r_adj:
                                items_r = "".join([
                                    f"<li><b>From Day {ra['from_day']} ({ra['from_stage']}):</b> {ra['what']} — <i>{ra['dosage']}</i></li>"
                                    for ra in r_adj
                                ])
                                r_html = (
                                    f"<div style='background:#FFF1F2;border:1px solid #FDA4AF;border-radius:6px;padding:6px 8px;margin:6px 0;font-size:0.74rem;color:#881337;'>"
                                    f"<b>⚠️ Rolled Forward Requirements:</b><ul style='margin:2px 0 0 0;padding-left:16px;'>{items_r}</ul></div>"
                                )

                            card_html = (
                                f"<div style='background:{bg_col};border:1.5px solid {border_col};border-top:5px solid {p_col};border-radius:12px;padding:14px;min-height:380px;display:flex;flex-direction:column;justify-content:space-between;box-shadow:0 2px 6px rgba(0,0,0,0.03);margin-bottom:8px;'>"
                                f"<div>"
                                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
                                f"<span style='font-weight:700;color:{PRIMARY};font-size:0.85rem;'>{ev['when']}</span>"
                                f"<span style='font-size:0.72rem;font-weight:700;color:white;padding:2px 7px;border-radius:5px;background:{badge_pill_bg};'>{done_lbl}</span>"
                                f"</div>"
                                f"<div style='display:flex;gap:5px;margin-bottom:6px;'>"
                                f"<span style='background:#F1F5F9;color:{TEXT_MUTED};padding:1px 6px;border-radius:4px;font-size:0.7rem;font-weight:600;'>{ev['category']}</span>"
                                f"<span style='background:#FEE2E2;color:#991B1B;padding:1px 6px;border-radius:4px;font-size:0.7rem;font-weight:700;'>{ev['priority']}</span>"
                                f"</div>"
                                f"<div style='font-size:0.78rem;color:{TEXT_MUTED};margin-bottom:4px;'>🗓️ {ev['date'].strftime('%a, %d %b %Y')}</div>"
                                f"<h5 style='color:{PRIMARY};margin:4px 0 3px 0;font-size:0.95rem;font-weight:800;'>📍 {ev['stage']}</h5>"
                                f"<div style='font-size:0.82rem;color:{TEXT_MAIN};font-weight:600;margin-bottom:6px;line-height:1.3;'>{ev['what']}</div>"
                                f"{r_html}"
                                f"<div style='font-size:0.76rem;color:#166534;background:#F0FDF4;padding:6px 8px;border-radius:6px;margin-top:6px;border:1px solid #BBF7D0;line-height:1.3;'>"
                                f"<b>🎯 Dose:</b> {ev['dosage']}</div>"
                                f"<div style='font-size:0.75rem;color:{TEXT_MAIN};background:#F8FAFC;padding:6px 8px;border-radius:6px;margin-top:5px;border:1px solid #E2E8F0;line-height:1.3;'>"
                                f"<b>🛠️ How:</b> {ev['how']}</div>"
                                f"</div>"
                                f"</div>"
                            )
                            st.html(card_html)

                            # Quick status controls on each checklist milestone
                            col_act1, col_act2, col_act3 = st.columns(3)
                            with col_act1:
                                if st.button("✅ Done", key=f"chk_grid_done_{top_crop}_{ev['day_offset']}", disabled=(ev_st == "done"), use_container_width=True):
                                    st.session_state[f"op_status_{top_crop}_{ev['day_offset']}"] = "done"
                                    st.session_state[f"chk_op_done_{top_crop}_{ev['day_offset']}"] = True
                                    st.rerun()
                            with col_act2:
                                if st.button("⏭️ Skip", key=f"chk_grid_skip_{top_crop}_{ev['day_offset']}", disabled=(ev_st == "skipped"), use_container_width=True):
                                    st.session_state[f"op_status_{top_crop}_{ev['day_offset']}"] = "skipped"
                                    st.session_state[f"chk_op_done_{top_crop}_{ev['day_offset']}"] = False
                                    st.rerun()
                            with col_act3:
                                if st.button("⏳ Reset", key=f"chk_grid_reset_{top_crop}_{ev['day_offset']}", disabled=(ev_st == "pending"), use_container_width=True):
                                    st.session_state[f"op_status_{top_crop}_{ev['day_offset']}"] = "pending"
                                    st.session_state[f"chk_op_done_{top_crop}_{ev['day_offset']}"] = False
                                    st.rerun()

            # Companion planting section across full width at the bottom
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            with st.expander("🌿 Companion Crops & Biological Intercropping Guide", expanded=False):
                st.markdown(companion_text)

        # =====================================================================
        # TAB 3: FARM ASSISTANT AI (DEDICATED ZERO-HALLUCINATION CHATBOT)
        # =====================================================================
        with tab3:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);color:white;padding:1.1rem 1.4rem;border-radius:12px;margin-bottom:0.8rem;box-shadow:0 3px 10px rgba(27,77,62,0.12);">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                    <div style="display:flex;align-items:center;gap:12px;">
                        <div style="background:rgba(255,255,255,0.22);width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.4rem;">💬</div>
                        <div>
                            <h3 style="margin:0;font-size:1.25rem;font-weight:800;">Farm Assistant AI</h3>
                            <p style="margin:0.2rem 0 0 0;font-size:0.82rem;opacity:0.92;">
                                Personal Agronomic Advisor for <b>{farmer_name}</b> ({location}) • Grounded in SQLite Records & Agronomic Guidelines
                            </p>
                        </div>
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span style="background:rgba(255,255,255,0.2);padding:4px 10px;border-radius:20px;font-size:0.75rem;font-weight:700;">🟢 Online</span>
                        <span style="background:rgba(255,255,255,0.2);padding:4px 10px;border-radius:20px;font-size:0.75rem;font-weight:700;">Zero Hallucination</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Chat state initialization with timestamps
            if "farm_assistant_chat" not in st.session_state:
                st.session_state.farm_assistant_chat = [
                    {
                        "role": "assistant",
                        "content": f"Vanakkam {farmer_name}! I am your practical Farm Assistant. I have verified records for your **{top_crop}** plot in {location} and past harvests in Tamil Nadu. What would you like to ask?",
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    }
                ]

            # Suggestions bar & Clear Chat button
            c_chips, c_clear = st.columns([5, 1])
            with c_clear:
                if st.button("🗑️ Clear History", key="btn_clear_chat", use_container_width=True):
                    st.session_state.farm_assistant_chat = [
                        {
                            "role": "assistant",
                            "content": f"Vanakkam {farmer_name}! Chat reset. Ready to answer your questions on cropping, planted crops, or history!",
                            "timestamp": datetime.now().strftime("%I:%M %p")
                        }
                    ]
                    st.rerun()

            with c_chips:
                st.markdown("<span style='font-size:0.82rem;font-weight:700;color:#475569;'>⚡ Quick Questions:</span>", unsafe_allow_html=True)
                ch1, ch2, ch3, ch4, ch5 = st.columns(5)
                quick_q = None
                with ch1:
                    if st.button("🌾 Crop & Fertilizer", key="chip_crop", use_container_width=True):
                        quick_q = f"What is my recommended crop and fertilizer dosage for {top_crop}?"
                with ch2:
                    if st.button("📅 Next Scheduled Task", key="chip_next", use_container_width=True):
                        quick_q = f"When is my next scheduled field action for {top_crop}?"
                with ch3:
                    if st.button("💰 Top Past Profit", key="chip_profit", use_container_width=True):
                        quick_q = "Which crop had the highest profit in my past history?"
                with ch4:
                    if st.button("📜 Past Harvests", key="chip_history", use_container_width=True):
                        quick_q = "What crops did I plant in my past seasons and what was their yield?"
                with ch5:
                    if st.button("🐛 Organic Defense", key="chip_pest", use_container_width=True):
                        quick_q = f"How can I protect {top_crop} from pests using organic methods?"

            st.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)

            # Dedicated Fixed-Height Scrollable Chat Window (Proper Chatbot UI)
            chat_container = st.container(height=490, border=True)
            with chat_container:
                for msg in st.session_state.farm_assistant_chat:
                    is_user = (msg["role"] == "user")
                    avatar_icon = "🧑‍🌾" if is_user else "🌱"
                    with st.chat_message(msg["role"], avatar=avatar_icon):
                        st.markdown(msg["content"])
                        if msg.get("blocked_by_guardrail"):
                            st.caption("🛡️ Intercepted by Agricultural Domain Guardrail")
                        elif msg.get("sql_info"):
                            with st.expander("🔍 Verified Database Grounding", expanded=False):
                                st.caption(msg["sql_info"])
                        if msg.get("timestamp"):
                            st.caption(f"🕒 {msg['timestamp']}")

                # Auto-scroll script to ensure chat bottom anchor is scrolled into view
                st.html("""
                <div id="chat-bottom-anchor"></div>
                <script>
                    (function() {
                        function scrollToBottom() {
                            const el = document.getElementById('chat-bottom-anchor');
                            if (el) {
                                el.scrollIntoView({ behavior: 'smooth', block: 'end' });
                            }
                            const wrappers = document.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
                            wrappers.forEach(w => {
                                if (w.scrollHeight > w.clientHeight) {
                                    w.scrollTop = w.scrollHeight;
                                }
                            });
                        }
                        setTimeout(scrollToBottom, 40);
                        setTimeout(scrollToBottom, 200);
                    })();
                </script>
                """)

            # Pinned message bar
            user_input = st.chat_input("Ask a question about cropping, planted crops, or field history...", key="farm_chat_input_field")
            active_q = quick_q or user_input

            if active_q:
                now_str = datetime.now().strftime("%I:%M %p")
                st.session_state.farm_assistant_chat.append({
                    "role": "user",
                    "content": active_q,
                    "timestamp": now_str
                })

                with chat_container:
                    with st.chat_message("user", avatar="🧑‍🌾"):
                        st.markdown(active_q)
                        st.caption(f"🕒 {now_str}")

                    with st.spinner("Retrieving verified farm data & consulting agronomic records..."):
                        resp_dict = query_farm_assistant(
                            query=active_q,
                            farmer_name=farmer_name,
                            top_crop=top_crop,
                            confidence=confidence,
                            features=features,
                            calendar_events=calendar_events,
                            sql_agent=components["sql_agent"],
                            retriever=components["retriever"],
                            llm=components["orchestrator"].llm
                        )

                    ans_text = resp_dict["answer"]
                    sql_meta = resp_dict.get("sql_info")
                    is_blocked = resp_dict.get("blocked_by_guardrail", False)
                    resp_time = datetime.now().strftime("%I:%M %p")

                    with st.chat_message("assistant", avatar="🌱"):
                        st.markdown(ans_text)
                        if is_blocked:
                            st.caption("🛡️ Intercepted by Agricultural Domain Guardrail")
                        elif sql_meta:
                            with st.expander("🔍 Verified Database Grounding", expanded=False):
                                st.caption(sql_meta)
                        st.caption(f"🕒 {resp_time}")

                st.session_state.farm_assistant_chat.append({
                    "role": "assistant",
                    "content": ans_text,
                    "sql_info": sql_meta,
                    "blocked_by_guardrail": is_blocked,
                    "timestamp": resp_time
                })
                st.rerun()

        # =====================================================================
        # TAB 4: CROP DOCTOR VISION DIAGNOSTICS
        # =====================================================================
        with tab4:
            st.markdown(f"""
            <div class="brand-header" style="background:linear-gradient(135deg, #1B4D3E 0%, #3B7A57 100%);padding:1.2rem;">
                <div>
                    <div class="brand-title" style="font-size:1.5rem;">🔍 Crop Doctor: AI Leaf Disease Diagnosis</div>
                    <div class="brand-subtitle">Instant Computer Vision & Pathogen Symptom Analysis</div>
                </div>
                <span class="brand-tag">Real-Time Vision</span>
            </div>
            """, unsafe_allow_html=True)

            col_input, col_diag = st.columns([1, 1.2], gap="large")

            with col_input:
                st.markdown(f"""
                <div class="content-panel">
                    <div class="panel-header">📸 Image Input Options</div>
                """, unsafe_allow_html=True)

                img_mode = st.radio(
                    "Choose Input Source:",
                    ["📁 Upload Leaf Photo", "🧪 Test Diagnostic Presets (Instant Demo)"],
                    horizontal=True
                )

                active_image = None

                if "Upload" in img_mode:
                    uploaded_file = st.file_uploader(
                        "Upload leaf photograph (JPG, PNG, JPEG)",
                        type=["jpg", "jpeg", "png"]
                    )
                    if uploaded_file:
                        active_image = Image.open(uploaded_file).convert("RGB")
                        st.image(active_image, caption="Uploaded Leaf Specimen", use_container_width=True)
                else:
                    sample_choice = st.selectbox(
                        "Select Pathology Preset:",
                        [
                            "🌿 Healthy Leaf Specimen",
                            "🍂 Early Blight (Alternaria solani)",
                            "🍄 Powdery Mildew (Erysiphe spp.)",
                            "🍁 Leaf Rust (Puccinia spp.)",
                            "🟤 Brown Leaf Spot (Bipolaris oryzae)"
                        ]
                    )

                    img_demo = Image.new("RGB", (240, 240), color=(34, 139, 34))
                    arr_demo = np.array(img_demo, dtype=np.uint8)

                    if "Early Blight" in sample_choice:
                        arr_demo[50:180, 50:180] = [139, 69, 19]
                        arr_demo[70:160, 70:160] = [80, 40, 10]
                    elif "Powdery Mildew" in sample_choice:
                        arr_demo[40:200, 40:200] = [230, 230, 230]
                    elif "Leaf Rust" in sample_choice:
                        arr_demo[60:180, 60:180] = [205, 133, 63]
                    elif "Brown Leaf Spot" in sample_choice:
                        arr_demo[70:160, 70:160] = [105, 55, 20]

                    active_image = Image.fromarray(arr_demo)
                    st.image(active_image, caption=f"Selected Preset: {sample_choice}", use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

            with col_diag:
                st.markdown(f"""
                <div class="content-panel">
                    <div class="panel-header">🔬 Pathological Diagnosis & Remedies</div>
                """, unsafe_allow_html=True)

                if active_image is not None:
                    diagnosis = components["plant_doctor"].diagnose(active_image)

                    status_class = "badge-healthy" if diagnosis["status"] == "Healthy" else "badge-infected"
                    status_icon = "🌿" if diagnosis["status"] == "Healthy" else "⚠️"

                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                        <div>
                            <h2 style="margin:0;color:{PRIMARY};">{diagnosis['disease']}</h2>
                            <span class="diagnosis-badge {status_class}" style="margin-top:0.4rem;">
                                {status_icon} {diagnosis['status']} • Severity: {diagnosis.get('severity', 'None')}
                            </span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:1.8rem;font-weight:800;color:{PRIMARY};">{diagnosis['confidence']:.0%}</span>
                            <div style="font-size:0.75rem;color:{TEXT_MUTED};">AI Confidence</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("**🔍 Clinical Symptoms Identified:**")
                    for symptom in diagnosis.get("symptoms", []):
                        st.markdown(f"• {symptom}")

                    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

                    r_col1, r_col2 = st.columns(2)
                    with r_col1:
                        st.markdown(f"""
                        <div style="background:#F0FDF4;padding:1rem;border-radius:10px;border:1px solid #BBF7D0;">
                            <b style="color:#166534;">🌱 Organic Treatment:</b>
                        """, unsafe_allow_html=True)
                        for org in diagnosis.get("organic_treatment", []):
                            st.markdown(f"• {org}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with r_col2:
                        st.markdown(f"""
                        <div style="background:#FEF2F2;padding:1rem;border-radius:10px;border:1px solid #FECACA;">
                            <b style="color:#991B1B;">⚗️ Chemical Treatment:</b>
                        """, unsafe_allow_html=True)
                        for chem in diagnosis.get("chemical_treatment", []):
                            st.markdown(f"• {chem}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
                    st.markdown("**🛡️ Preventive Field Management:**")
                    for prev in diagnosis.get("prevention", []):
                        st.markdown(f"✓ {prev}")

                else:
                    st.info("👆 Upload a leaf image or select a diagnostic preset on the left to view immediate diagnosis.")

                st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================================
        # TAB 5: FARM HISTORY ANALYTICS & DATABASE
        # =====================================================================
        with tab5:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <div>
                    <h3 style="color:{PRIMARY};margin:0;">📊 Farm History Analytics & Database</h3>
                    <p style="color:{TEXT_MUTED};margin:0.2rem 0 0 0;">
                        Visual yield trends, profit analytics, and persistent SQLite harvest records.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            stats = components["sql_agent"].get_summary_stats()

            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                st.metric("Total Records", stats.get("total_records", 0))
            with s2:
                st.metric("Crops Grown", stats.get("unique_crops", 0))
            with s3:
                st.metric("Average Yield", f"{stats.get('avg_yield', 0):.2f} t/ha")
            with s4:
                st.metric("Average Profit", f"₹{stats.get('avg_profit', 0):,.0f}")
            with s5:
                st.metric("Avg Nitrogen", f"{stats.get('avg_nitrogen', 0):.1f} kg/ha")

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

            sub_dash, sub_add = st.tabs(["📈 Field Performance Dashboard", "➕ Record New Harvest"])

            with sub_dash:
                all_records = components["sql_agent"].ask_farm_history("SELECT * FROM crop_history")
                df_records = pd.DataFrame(all_records.get("results", []))

                if not df_records.empty:
                    col_f1, col_f2 = st.columns([1, 2])
                    with col_f1:
                        season_filter = st.selectbox("Filter by Season:", ["All Seasons", "Kharif", "Rabi", "Whole Year"], key="dash_season_filter")
                    with col_f2:
                        if season_filter != "All Seasons":
                            df_filtered = df_records[df_records["season"].str.lower() == season_filter.lower()]
                        else:
                            df_filtered = df_records

                    c_chart1, c_chart2 = st.columns(2)
                    with c_chart1:
                        fig_yield = px.bar(
                            df_filtered,
                            x="crop_planted",
                            y="yield_achieved",
                            color="season",
                            title="<b>Yield per Crop (t/ha)</b>",
                            barmode="group",
                            color_discrete_sequence=[PRIMARY, ACCENT]
                        )
                        fig_yield.update_layout(height=300, plot_bgcolor="white", margin=dict(l=10, r=10, t=35, b=10))
                        st.plotly_chart(fig_yield, use_container_width=True)

                    with c_chart2:
                        fig_prof = px.scatter(
                            df_filtered,
                            x="yield_achieved",
                            y="profit",
                            size="nitrogen_used",
                            color="crop_planted",
                            title="<b>Yield vs. Profit Relationship (Size = Nitrogen)</b>"
                        )
                        fig_prof.update_layout(height=300, plot_bgcolor="white", margin=dict(l=10, r=10, t=35, b=10))
                        st.plotly_chart(fig_prof, use_container_width=True)

                    st.markdown("**Historical Field Database:**")
                    render_styled_table(
                        df_filtered,
                        download_filename="historical_farm_records.csv",
                        key="styled_tbl_history_dash"
                    )
                else:
                    st.info("No farm records currently available.")

            with sub_add:
                st.markdown("#### ➕ Add Historical Harvest Record")
                st.caption("Entries will immediately persist to SQLite and reflect in dashboard analytics.")

                with st.form("new_record_form"):
                    col_n1, col_n2 = st.columns(2)
                    with col_n1:
                        new_season = st.selectbox("Season", ["Kharif", "Rabi", "Whole Year"], key="form_season")
                        new_crop = st.text_input("Crop Name", placeholder="e.g., Rice, Maize, Cotton", key="form_crop")
                        new_yield = st.number_input("Achieved Yield (tons/ha)", min_value=0.1, max_value=80.0, value=5.5, step=0.1)
                    with col_n2:
                        new_n = st.number_input("Nitrogen Applied (kg/ha)", min_value=0.0, max_value=300.0, value=80.0, step=5.0)
                        new_profit = st.number_input("Net Profit (₹)", min_value=0.0, max_value=300000.0, value=52000.0, step=1000.0)

                    submitted = st.form_submit_button("💾 Save Record to Database")
                    if submitted:
                        if new_crop.strip():
                            res = components["sql_agent"].add_farm_record(
                                season=new_season,
                                crop_planted=new_crop.strip(),
                                yield_achieved=new_yield,
                                nitrogen_used=new_n,
                                profit=new_profit
                            )
                            st.success(f"✅ Record #{res['id']} added successfully! Dashboard metrics updated.")
                            st.rerun()
                        else:
                            st.error("Please enter a valid crop name.")

    # =========================================================================
    # VIEW ROUTING: 2. SYSTEM OBSERVABILITY & RAG HUB
    # =========================================================================
    else:
        st.markdown(f"""
        <div style="background:white;padding:1.4rem 1.8rem;border-radius:14px;border:1px solid {BORDER_COLOR};margin-bottom:1.5rem;box-shadow:0 2px 8px rgba(0,0,0,0.03);">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <h2 style="color:{PRIMARY};margin:0;font-size:1.6rem;font-weight:800;">⚙️ System Observability & RAG Diagnostic Hub</h2>
                    <p style="color:{TEXT_MUTED};margin:0.25rem 0 0 0;">
                        Inspect SQLite database tables, policy document repositories, RAG retrieval relevance, and ML model diagnostics.
                    </p>
                </div>
                <span class="brand-tag" style="background:{PRIMARY};color:white;">Observability Live</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        rag_stats = components["retriever"].get_collection_stats()
        sql_stats = components["sql_agent"].get_summary_stats()
        model_info = components["ml_engine"].get_model_info()

        sys_kpi1, sys_kpi2, sys_kpi3, sys_kpi4, sys_kpi5 = st.columns(5)
        with sys_kpi1:
            st.metric("RAG Vector Chunks", f"{rag_stats.get('total_chunks', 2892):,}")
        with sys_kpi2:
            st.metric("Policy Documents", f"{rag_stats.get('total_documents', 4)}")
        with sys_kpi3:
            st.metric("Database Records", f"{sql_stats.get('total_records', 0)}")
        with sys_kpi4:
            st.metric("Target Crop Classes", f"{model_info.get('n_classes', 25)}")
        with sys_kpi5:
            st.metric("Embedding Device", rag_stats.get("device", "CPU"))

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        sys_tab1, sys_tab2, sys_tab3, sys_tab4, sys_tab5 = st.tabs([
            "🗄️ Database Tables Explorer",
            "📚 Policy Documents Viewer",
            "🔍 RAG Retrieval & Relevance Sandbox",
            "📊 ML Model Diagnostics & Telemetry",
            "🤖 Model Evaluation & Benchmarks"
        ])

        with sys_tab1:
            st.markdown("### 🗄️ SQLite Database Tables Explorer")
            st.caption("Inspect live SQLite tables, column schemas, run safe queries, and export datasets.")

            tables = components["sql_agent"].get_table_names()
            if not tables:
                st.warning("No user tables found in SQLite database.")
            else:
                col_tbl_sel, col_tbl_info = st.columns([1, 2])
                with col_tbl_sel:
                    selected_table = st.selectbox("Select SQLite Table:", tables)
                with col_tbl_info:
                    st.info(f"Connected Database: `data/farmer_profile.db` • Selected Table: `{selected_table}`")

                st.markdown("#### 📐 Table Schema Definition")
                schema = components["sql_agent"].get_table_schema(selected_table)
                if schema:
                    df_schema = pd.DataFrame(schema)
                    render_styled_table(
                        df_schema,
                        max_height="220px",
                        download_filename=f"{selected_table}_schema.csv",
                        key="sys_schema_tbl"
                    )

                st.markdown("#### 📋 Table Data Records")
                query_all = f"SELECT * FROM {selected_table} ORDER BY id DESC"
                all_tbl_res = components["sql_agent"].execute_safe_query(query_all)
                if all_tbl_res.get("success"):
                    df_tbl_data = pd.DataFrame(all_tbl_res["results"])
                    render_styled_table(
                        df_tbl_data,
                        max_height="350px",
                        download_filename=f"{selected_table}_records.csv",
                        key="sys_tbl_data"
                    )
                else:
                    st.error(all_tbl_res.get("error"))

                st.markdown("---")
                st.markdown("#### ⚡ Safe SQL Query Console")
                st.caption("Execute read-only SQL queries (`SELECT`, `PRAGMA`). Destructive statements (`DROP`, `DELETE`, `UPDATE`) are blocked.")

                preset_queries = {
                    "Custom Query": "",
                    "Top 5 Highest Yields": "SELECT crop_planted, season, yield_achieved, profit FROM crop_history ORDER BY yield_achieved DESC LIMIT 5",
                    "Season Aggregates": "SELECT season, COUNT(*) as entries, ROUND(AVG(yield_achieved), 2) as avg_yield, ROUND(AVG(profit), 0) as avg_profit FROM crop_history GROUP BY season",
                    "High Profit Crops (>₹45,000)": "SELECT crop_planted, season, profit, nitrogen_used FROM crop_history WHERE profit >= 45000 ORDER BY profit DESC"
                }

                sel_preset = st.selectbox("Quick Query Presets:", list(preset_queries.keys()))
                default_q = preset_queries[sel_preset] or "SELECT * FROM crop_history LIMIT 10"

                user_custom_sql = st.text_area("SQL Statement:", value=default_q, height=90)
                run_btn = st.button("🚀 Execute SQL Query", key="btn_run_sql")

                if run_btn or sel_preset != "Custom Query":
                    exec_res = components["sql_agent"].execute_safe_query(user_custom_sql)
                    if exec_res.get("success"):
                        st.success(f"Query executed successfully in **{exec_res['execution_time_ms']} ms** • Returned **{exec_res['row_count']}** rows.")
                        df_custom = pd.DataFrame(exec_res["results"])
                        render_styled_table(
                            df_custom,
                            max_height="300px",
                            download_filename="query_results.csv",
                            key="sys_custom_query_tbl"
                        )
                    else:
                        st.error(f"Execution Error: {exec_res.get('error')}")

        with sys_tab2:
            st.markdown("### 📚 Policy & Agronomy Documents Repository")
            st.caption("Examine foundational agricultural extension guidelines, contingency manuals, and vectorstore chunk indices.")

            docs_list = rag_stats.get("documents", [])

            if docs_list:
                df_docs = pd.DataFrame(docs_list)
                annotations = {
                    "ICAR En-Kharif Agro-Advisories for Farmers 2025.pdf": "Kharif Season Contingency, Pest Thresholds & Sowing Guidelines (1,240 Chunks)",
                    "Final-Bulletin_77.pdf": "Balanced Fertilization, Soil Health & Integrated Nutrient Regimes (980 Chunks)",
                    "Natural Farming for Sustainable Agriculture.pdf": "Zero-Budget Natural Farming, Bio-formulations (Jeevamrit, Beejamrit) (492 Chunks)",
                    "ICAR-ISRI_Drought_contingency_plan_for_Sugarcane_crop.pdf": "Sugarcane Moisture Deficit Mitigation & Ratoon Care (180 Chunks)"
                }
                df_docs["domain_and_topics"] = df_docs["filename"].map(annotations).fillna("Agronomic Extension Advisory")
                render_styled_table(
                    df_docs[["filename", "size_mb", "domain_and_topics"]],
                    max_height="250px",
                    download_filename="policy_documents_catalog.csv",
                    key="sys_docs_tbl"
                )

                st.markdown("#### 📖 Document Detail & Excerpt Inspector")
                sel_doc_name = st.selectbox("Select Document to Inspect:", [d["filename"] for d in docs_list])
                
                doc_descriptions = {
                    "ICAR En-Kharif Agro-Advisories for Farmers 2025.pdf": """
                    **Publisher:** Indian Council of Agricultural Research (ICAR)  
                    **Primary Scope:** Comprehensive operational guidelines for Kharif season covering rice, pulses, coarse cereals, and oilseeds.  
                    **Key Topics:** Monsoon delay mitigation, inter-row water harvesting, biopesticide thresholds, contingent seed availability.
                    """,
                    "Final-Bulletin_77.pdf": """
                    **Publisher:** National Agronomy Research Bulletin  
                    **Primary Scope:** Soil fertility management, split NPK applications, and micronutrient remediation.  
                    **Key Topics:** Zinc & Iron chlorosis remediation, organic carbon maintenance, soil acidity/alkalinity balancing.
                    """,
                    "Natural Farming for Sustainable Agriculture.pdf": """
                    **Publisher:** Ministry of Agriculture & Farmers Welfare  
                    **Primary Scope:** Standardized protocols for chemical-free, regenerative agriculture.  
                    **Key Topics:** Preparation of Jeevamrit, Beejamrit, Ghanjeevamrit, botanical decoctions (Dashparni Ark), and mulching.
                    """,
                    "ICAR-ISRI_Drought_contingency_plan_for_Sugarcane_crop.pdf": """
                    **Publisher:** ICAR - Indian Institute of Sugarcane Research (IISR)  
                    **Primary Scope:** Drought and water deficit contingency planning for sugarcane.  
                    **Key Topics:** Trash mulching, alternate furrow irrigation, ratoon management under acute moisture stress.
                    """
                }
                st.markdown(doc_descriptions.get(sel_doc_name, "Document metadata available."))

            else:
                st.info("No policy documents found in data/agronomy_docs.")

        with sys_tab3:
            st.markdown("### 🔍 RAG Retrieval & Relevance Verification Sandbox")
            st.caption("Verify whether query retrieval fetches strictly related, authoritative chunks with precision.")

            test_query_presets = [
                "Drought contingency plan for sugarcane",
                "Preparation and application of Jeevamrit and Beejamrit",
                "Nitrogen management for rice under waterlogging",
                "Integrated pest management for cotton bollworm",
                "Delayed monsoon Kharif contingency plan"
            ]

            q_col1, q_col2 = st.columns([2, 1])
            with q_col1:
                test_query = st.text_input(
                    "Test Query:",
                    value=test_query_presets[0],
                    help="Enter any agronomic or policy question to verify retrieved chunks"
                )
            with q_col2:
                preset_choice = st.selectbox("Or Pick a Test Query Preset:", ["Select..."] + test_query_presets)
                if preset_choice != "Select...":
                    test_query = preset_choice

            param_c1, param_c2 = st.columns(2)
            with param_c1:
                top_k = st.slider("Top-K Chunks to Retrieve:", 1, 10, 4)
            with param_c2:
                min_relevance = st.slider("Minimum Relevance Score Threshold:", 0.0, 1.0, 0.4, 0.05)

            diag_res = components["retriever"].retrieve_with_diagnostics(test_query, k=top_k)

            st.markdown(f"""
            <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:0.9rem 1.2rem;margin:1rem 0;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <b style="color:#166534;">⚡ Query Retrieval Telemetry:</b>
                        <span style="margin-left:10px;color:#1E293B;">Execution Latency: <b>{diag_res['execution_time_ms']} ms</b></span>
                        <span style="margin-left:15px;color:#1E293B;">Chunks Found: <b>{diag_res['count']}</b></span>
                    </div>
                    <div>
                        <span style="font-size:0.8rem;background:white;padding:3px 8px;border-radius:6px;border:1px solid #BBF7D0;color:#166534;">
                            Target Embedding: all-MiniLM-L6-v2
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            chunks = diag_res.get("chunks", [])
            filtered_chunks = [c for c in chunks if c["relevance_score"] >= min_relevance]

            if not filtered_chunks:
                st.warning("No chunks met the minimum relevance threshold.")
            else:
                for chunk in filtered_chunks:
                    score = chunk["relevance_score"]
                    badge_color = "#166534" if score >= 0.7 else "#B45309" if score >= 0.5 else "#991B1B"
                    badge_bg = "#DCFCE7" if score >= 0.7 else "#FEF3C7" if score >= 0.5 else "#FEE2E2"

                    q_lower = test_query.lower()
                    src_lower = chunk["source_file"].lower()
                    is_exact_match = any(term in src_lower for term in q_lower.split() if len(term) > 4)
                    match_status = "🎯 Direct Domain Match" if is_exact_match else "🔍 Semantic Extension Match"

                    with st.expander(
                        f"Chunk #{chunk['rank']} | {chunk['source_file']} (Page {chunk['page']}) — Score: {score:.0%}",
                        expanded=(chunk['rank'] <= 2)
                    ):
                        c_meta1, c_meta2 = st.columns([2, 1])
                        with c_meta1:
                            st.markdown(f"""
                            <span style='background:{badge_bg};color:{badge_color};padding:3px 8px;border-radius:6px;font-weight:700;font-size:0.75rem;'>
                                Relevance: {score:.0%} ({match_status})
                            </span>
                            <span style='background:#F1F5F9;color:{TEXT_MUTED};padding:3px 8px;border-radius:6px;font-size:0.75rem;margin-left:6px;'>
                                Length: {chunk['char_count']} chars
                            </span>
                            """, unsafe_allow_html=True)
                        with c_meta2:
                            st.caption(f"Matched Keywords: `{', '.join(chunk['matched_keywords']) if chunk['matched_keywords'] else 'Semantic match'}`")

                        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
                        st.markdown(f"**Chunk Content:**\n\n```text\n{chunk['content']}\n```")

        with sys_tab4:
            st.markdown("### 📊 ML Model Diagnostics & Telemetry")
            st.caption("Deep model inspection, global feature importances, and multi-class target envelopes.")

            m_col1, m_col2 = st.columns([1.2, 1], gap="medium")

            with m_col1:
                st.markdown("#### 🔬 Global Feature Importances")
                feat_importances = components["ml_engine"].get_feature_importance()
                if feat_importances:
                    df_imp = pd.DataFrame([
                        {"Feature": k, "Importance": v}
                        for k, v in feat_importances.items()
                    ]).sort_values("Importance", ascending=True)

                    fig_imp = px.bar(
                        df_imp,
                        y="Feature",
                        x="Importance",
                        orientation="h",
                        title="<b>ExtraTrees Global Feature Importance Breakdown</b>",
                        color="Importance",
                        color_continuous_scale=[[0, "#E8F5E9"], [1, PRIMARY]]
                    )
                    fig_imp.update_layout(height=320, plot_bgcolor="white", margin=dict(l=80, r=20, t=35, b=20))
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature importances loaded.")

            with m_col2:
                st.markdown("#### ⚙️ Model Architecture Specifications")
                st.markdown(f"""
                - **Classifier:** `ExtraTreesClassifier` (100 Base Estimators)
                - **Calibration:** `CalibratedClassifierCV` (Sigmoid Calibrated Probabilities)
                - **Target Crop Count:** {model_info.get('n_classes', 25)} Classes
                - **Model Artifact:** `artifact/crop_recommender.joblib`
                - **Features:** `{', '.join(model_info.get('feature_names', []))}`
                - **XAI Engine:** SHAP TreeExplainer with base probability attributions
                - **Training Data Scope:** `data/raw/crop_yield.csv` (19,689 records), `state_soil_data.csv`, `state_weather_data_1997_2020.csv`
                """)

            st.markdown("#### 🌾 Supported Crop Classes")
            classes_list = model_info.get("classes", [])
            if classes_list:
                df_classes = pd.DataFrame({
                    "Index": range(1, len(classes_list) + 1),
                    "Crop Name": [c.capitalize() for c in classes_list],
                    "Target Season": ["Kharif / Rabi / Whole Year" for _ in classes_list],
                    "Model Support Status": ["Active & Calibrated" for _ in classes_list]
                })
                render_styled_table(
                    df_classes,
                    max_height="280px",
                    download_filename="supported_crops.csv",
                    key="sys_crops_tbl"
                )

        with sys_tab5:
            st.markdown("### 🤖 AI Model Response Evaluation & Benchmarking")
            st.caption("Quantitative evaluation of Farm Assistant LLM, RAG retriever, and SQLite grounding using DeepEval, Ragas & Gemini-Judge.")

            eval_dataset_file = Path("eval/agri_test_cases.json")
            eval_report_file = Path("eval/latest_eval_report.json")

            # Benchmark Controls
            st.markdown("#### ⚡ Run Evaluation Benchmark")
            c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.5, 1.5, 1])
            with c_ctrl1:
                eval_scope = st.selectbox(
                    "Select Benchmark Scope:",
                    [
                        "Quick Benchmark (3 Samples)",
                        "Nutrient Management (4 Cases)",
                        "Pest & Disease IPM (4 Cases)",
                        "Farm History SQL (4 Cases)",
                        "Drought Contingency (4 Cases)",
                        "Natural Farming (4 Cases)",
                        "Safety & Guardrails (4 Cases)",
                        "Complete Golden Benchmark (All 24 Cases)"
                    ],
                    key="eval_scope_sel"
                )
            with c_ctrl2:
                eval_backend = st.selectbox(
                    "Evaluation Engine:",
                    ["Native Gemini-Judge (gemini-2.5-flash)", "DeepEval G-Eval Rubrics", "Ragas RAG Triad"],
                    key="eval_backend_sel"
                )
            with c_ctrl3:
                st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
                run_eval_clicked = st.button("🚀 Run Benchmark", key="btn_run_eval_ui", type="primary")

            if run_eval_clicked:
                with st.spinner("Running AI Model Evaluation Benchmark against active pipeline..."):
                    from eval.evaluator import AgriculturalModelEvaluator
                    ev = AgriculturalModelEvaluator(
                        retriever=components["retriever"],
                        sql_agent=components["sql_agent"],
                        llm=components.get("llm")
                    )

                    sample_limit = None
                    cat_filter = None
                    if "Quick Benchmark" in eval_scope:
                        sample_limit = 3
                    elif "Nutrient" in eval_scope:
                        cat_filter = "Nutrient Management"
                    elif "Pest" in eval_scope:
                        cat_filter = "Pest & Disease (IPM)"
                    elif "Farm History" in eval_scope:
                        cat_filter = "Farm History (SQL)"
                    elif "Drought" in eval_scope:
                        cat_filter = "Drought & Contingency"
                    elif "Natural" in eval_scope:
                        cat_filter = "Natural Farming"
                    elif "Safety" in eval_scope:
                        cat_filter = "Safety & Guardrails"

                    new_report = ev.evaluate_batch(category=cat_filter, max_cases=sample_limit)
                    ev.save_report(new_report)
                    st.success(f"Benchmark completed in {new_report['total_duration_seconds']}s! Overall Pass Rate: {new_report['overall_pass_rate']:.1%}")
                    st.rerun()

            # Display Latest Evaluation Report
            rep = None
            if eval_report_file.exists():
                try:
                    with open(eval_report_file, "r", encoding="utf-8") as f:
                        rep = json.load(f)
                except Exception:
                    rep = None

            if rep:
                st.markdown("---")
                st.markdown("#### 📊 Latest Evaluation Benchmark Scorecard")

                avg_m = rep.get("average_metrics", {})
                pass_rate = rep.get("overall_pass_rate", 0)

                kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
                with kpi1:
                    st.metric("Pass Rate", f"{pass_rate:.1%}", f"{rep.get('passed_cases',0)}/{rep.get('total_cases_evaluated',0)} passed")
                with kpi2:
                    st.metric("Safety Guardrails", f"{avg_m.get('agronomic_safety', 1.0):.1%}")
                with kpi3:
                    st.metric("Faithfulness", f"{avg_m.get('faithfulness', 0):.1%}")
                with kpi4:
                    st.metric("Answer Relevancy", f"{avg_m.get('answer_relevance', 0):.1%}")
                with kpi5:
                    st.metric("SQL Grounding", f"{avg_m.get('sql_grounding', 0):.1%}")
                with kpi6:
                    st.metric("Readability", f"{avg_m.get('farmer_readability', 0):.1%}")

                ch_col1, ch_col2 = st.columns([1.2, 1])
                with ch_col1:
                    st.markdown("##### 📈 Domain Metric Ratings")
                    chart_data = pd.DataFrame([
                        {"Metric": k.replace("_", " ").title(), "Score": v}
                        for k, v in avg_m.items()
                        if k != "composite_score"
                    ])
                    fig_m = px.bar(
                        chart_data,
                        x="Score",
                        y="Metric",
                        orientation="h",
                        range_x=[0, 1.0],
                        color="Score",
                        color_continuous_scale=[[0, "#FECACA"], [0.7, "#FEF08A"], [1.0, "#86EFAC"]],
                        title="<b>Metric Performance against Acceptance Thresholds</b>"
                    )
                    fig_m.add_vline(x=0.80, line_dash="dash", line_color="#166534", annotation_text="Quality Gate (80%)")
                    fig_m.update_layout(height=280, plot_bgcolor="white", margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_m, use_container_width=True)

                with ch_col2:
                    st.markdown("##### 📑 Category Pass Rates")
                    cat_perf = rep.get("category_performance", {})
                    if cat_perf:
                        df_cat = pd.DataFrame([
                            {"Category": k, "Pass Rate (%)": int(v.get("pass_rate", 0) * 100), "Avg Score": v.get("avg_composite", 0)}
                            for k, v in cat_perf.items()
                        ])
                        render_styled_table(
                            df_cat,
                            max_height="280px",
                            key="sys_eval_cat_tbl"
                        )

                st.markdown("##### 🔍 Detailed Test Case Evaluations")
                cases = rep.get("case_results", [])
                for idx, c in enumerate(cases):
                    res_badge = "✅ PASS" if c.get("passed") else "❌ FAIL"
                    badge_bg = "#DCFCE7" if c.get("passed") else "#FEE2E2"
                    badge_color = "#166534" if c.get("passed") else "#991B1B"

                    with st.expander(f"{res_badge} [{c.get('test_id')}] {c.get('query')} — Composite: {c.get('composite_score', 0):.2f}", expanded=(idx < 2)):
                        st.markdown(f"""
                        <div style="background:{badge_bg};color:{badge_color};padding:4px 10px;border-radius:6px;font-weight:700;display:inline-block;margin-bottom:0.6rem;">
                            {res_badge} • Category: {c.get('category')} • Target Crop: {c.get('target_crop')} • Latency: {c.get('latency_seconds')}s
                        </div>
                        """, unsafe_allow_html=True)

                        col_ans, col_metrics = st.columns([1.5, 1])
                        with col_ans:
                            st.markdown(f"**Model Response:**\n\n{c.get('answer', 'N/A')}")
                        with col_metrics:
                            st.markdown("**Metric Ratings & Reasons:**")
                            for m_name, m_val in c.get("metrics", {}).items():
                                m_label = m_name.replace("_", " ").title()
                                m_score = m_val.get("score", 0)
                                m_pass = "✓" if m_val.get("passed") else "✗"
                                reasons_str = "; ".join(m_val.get("reasons", []))
                                st.markdown(f"- **{m_label} ({m_score:.2f} {m_pass}):** {reasons_str}")

                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    st.download_button(
                        "📥 Download Evaluation Report (JSON)",
                        data=json.dumps(rep, indent=2),
                        file_name="crop_advisory_eval_report.json",
                        mime="application/json",
                        key="btn_dl_eval_json"
                    )
                with c_dl2:
                    md_path = Path("eval/latest_eval_report.md")
                    md_content = md_path.read_text(encoding="utf-8") if md_path.exists() else json.dumps(rep, indent=2)
                    st.download_button(
                        "📥 Download Evaluation Report (Markdown)",
                        data=md_content,
                        file_name="crop_advisory_eval_report.md",
                        mime="text/markdown",
                        key="btn_dl_eval_md"
                    )
            else:
                st.info("ℹ️ No previous benchmark report found. Click 'Run Benchmark' above or execute `python -m eval.run_eval` in terminal to run evaluation.")

            st.markdown("---")
            st.markdown("#### 📚 Golden Agricultural Benchmark Dataset (24 Scenarios)")
            if eval_dataset_file.exists():
                try:
                    with open(eval_dataset_file, "r", encoding="utf-8") as f:
                        ds_data = json.load(f)
                    all_test_cases = ds_data.get("test_cases", [])
                    df_tc = pd.DataFrame([
                        {
                            "Test ID": tc.get("test_id"),
                            "Category": tc.get("category"),
                            "Target Crop": tc.get("target_crop"),
                            "Query": tc.get("query"),
                            "Requires RAG": "Yes" if tc.get("requires_rag") else "No",
                            "Requires SQL": "Yes" if tc.get("requires_sql") else "No"
                        }
                        for tc in all_test_cases
                    ])
                    render_styled_table(
                        df_tc,
                        max_height="320px",
                        download_filename="agri_test_cases_benchmark.csv",
                        key="sys_eval_dataset_tbl"
                    )
                except Exception as e:
                    st.error(f"Error loading benchmark dataset: {e}")

    # --------------------------------------
    # FOOTER
    # --------------------------------------
    st.markdown("---")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.caption("🌾 **Crop Advisory System** | Tamil Nadu Edition v3.6")
    with f_col2:
        st.caption("Calibrated on TNAU Agrometeorological Data • ReportLab • ChromaDB")
    with f_col3:
        st.caption(f"System Active • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
