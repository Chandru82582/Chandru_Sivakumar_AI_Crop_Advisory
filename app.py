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
        <div>
            <span class="brand-tag">Tamil Nadu Edition • v3.6</span>
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
            st.caption("Default state set to **Tamil Nadu** (dataAnalysis/data baseline).")

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

            # Presets calibrated against dataAnalysis/data state_soil_data.csv and state_weather_data_1997_2020.csv
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Farmer Advisory Report",
            "📅 Cultivation Timeline & Calendar",
            "🔍 Crop Doctor Vision",
            "📊 Farm Analytics & Chat"
        ])

        # =====================================================================
        # TAB 1: ADVISORY REPORT & ACTION SUMMARY (100% FARMER-FOCUSED)
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
                # PDF Generation compiled fast on demand
                timetable_for_pdf = [
                    {
                        "phase": e["phase"],
                        "duration": f"Day {e['day_offset']:+d} ({e['date'].strftime('%b %d')})",
                        "tasks": [f"{e['what']}: {e['dosage']}"],
                        "organic": e["organic"],
                        "chemical": e["chemical"]
                    }
                    for e in calendar_events[:6]
                ]

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
                        "timetable": timetable_for_pdf
                    },
                    ml_metrics={"predictions": prediction.get("predictions", [])},
                    timetable_data=timetable_for_pdf,
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

            col_adv_main, col_adv_side = st.columns([1.5, 1], gap="medium")

            with col_adv_main:
                st.markdown(f"""
                <div class="content-panel">
                    <div class="panel-header">🌾 Detailed Agricultural Advisory Report for {farmer_name}</div>
                """, unsafe_allow_html=True)
                st.markdown(adv_text)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_adv_side:
                # Farmer Action Checklist Snapshot
                st.markdown(f"""
                <div class="content-panel">
                    <div class="panel-header">📋 Farmer's Immediate Action Checklist</div>
                    <p style="color:{TEXT_MUTED};font-size:0.88rem;">Key operations for upcoming field stages. Track progress in the Cultivation Timeline tab.</p>
                """, unsafe_allow_html=True)

                for ev in calendar_events[:5]:
                    t_key = f"snap_task_{top_crop}_{ev['day_offset']}"
                    if t_key not in st.session_state:
                        st.session_state[t_key] = (ev["day_offset"] <= -7)
                    is_t_done = st.session_state.get(t_key, False)
                    done_icon = "✅" if is_t_done else "⏳"
                    
                    st.markdown(f"""
                    <div style="background:#F8FAF8;border:1px solid #E2E8F0;border-radius:8px;padding:0.75rem;margin-bottom:0.6rem;">
                        <div style="display:flex;justify-content:space-between;">
                            <span style="font-weight:700;color:{PRIMARY};font-size:0.85rem;">{done_icon} {ev['when']}</span>
                            <span style="font-size:0.72rem;background:#E8F5E9;color:#1B4D3E;padding:2px 6px;border-radius:4px;font-weight:600;">{ev['stage']}</span>
                        </div>
                        <div style="font-weight:600;font-size:0.85rem;color:{TEXT_MAIN};margin-top:2px;">{ev['what']}</div>
                        <div style="font-size:0.78rem;color:{TEXT_MUTED};margin-top:2px;">{ev['dosage'][:85]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:#E8F5E9;padding:0.8rem;border-radius:8px;font-size:0.85rem;color:#1B4D3E;margin-top:1rem;">
                    👉 <b>Open the '📅 Cultivation Timeline' tab</b> to view the full interactive Monthly Calendar and Daily Checklist!
                </div>
                </div>
                """, unsafe_allow_html=True)

                # What-If Simulator (Farmer Sensitivity)
                st.markdown(f"""
                <div class="content-panel">
                    <div class="panel-header">🔄 Field What-If Sensitivity Test</div>
                    <p style="color:{TEXT_MUTED};font-size:0.85rem;">Adjust factors to see how recommendation shifts if weather or nutrients change.</p>
                """, unsafe_allow_html=True)
                sim_param = st.selectbox(
                    "Simulate Factor:",
                    ["total_rainfall_mm", "avg_temp_c", "N", "P", "K", "pH"],
                    format_func=lambda x: {
                        "total_rainfall_mm": "Rainfall (mm)",
                        "avg_temp_c": "Temperature (°C)",
                        "N": "Nitrogen (N) kg/ha",
                        "P": "Phosphorus (P) kg/ha",
                        "K": "Potassium (K) kg/ha",
                        "pH": "Soil pH"
                    }.get(x, x),
                    key="sim_param_sel"
                )
                p_ranges = {"total_rainfall_mm": (200, 2500, 50), "avg_temp_c": (15.0, 42.0, 0.5), "N": (0, 200, 5), "P": (0, 100, 2), "K": (0, 100, 2), "pH": (4.5, 9.0, 0.1)}
                p_min, p_max, p_step = p_ranges[sim_param]
                sim_val = st.slider(f"Simulate New Value", p_min, p_max, features[sim_param], p_step, key="sim_val_slider")
                whatif_res = components["orchestrator"].get_what_if_analysis(features, sim_param, sim_val)
                st.caption(f"Baseline Recommendation: **{whatif_res['original_prediction']}** ({whatif_res['original_confidence']:.1%})  \nSimulated Recommendation: **{whatif_res['modified_prediction']}** ({whatif_res['modified_confidence']:.1%})")
                st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================================
        # TAB 2: CULTIVATION TIMELINE, INTERACTIVE CALENDAR & CHECKLIST
        # =====================================================================
        with tab2:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <div>
                    <h3 style="color:{PRIMARY};margin:0;">📅 Cultivation Operations & Interactive Calendar</h3>
                    <p style="color:{TEXT_MUTED};margin:0.2rem 0 0 0;">
                        Operational cultivation calendar for <b>{top_crop}</b> in <b>{location}</b> • Detailed farmer checklist.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top Controls
            ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.4, 1.4])
            with ctrl1:
                selected_sowing_date = st.date_input(
                    "🌾 Sowing / Cycle Start Date:",
                    value=sowing_date,
                    key="cal_sowing_date_picker"
                )
            with ctrl2:
                calendar_view_mode = st.radio(
                    "📅 Calendar View Mode:",
                    ["📆 Monthly Calendar View", "🗓️ Weekly Stage Progression", "📋 Daily Farmer Action Checklist"],
                    horizontal=True,
                    key="cal_view_mode_radio"
                )
            with ctrl3:
                treatment_mode = st.radio(
                    "🌿 Agronomic Protocol:",
                    ["Integrated Pest Management (IPM) ⚖️", "Organic & Regenerative 🌱", "Conventional / Chemical ⚗️"],
                    horizontal=True,
                    key="cal_treatment_mode_radio"
                )

            # Recalculate event dates based on sowing date
            all_events = get_crop_calendar_events(
                crop=top_crop,
                season=season,
                n=n, p=p, k=k, ph=ph,
                sowing_date=selected_sowing_date
            )

            # Progress Tracking
            total_tasks = len(all_events)
            completed_tasks = 0
            for ev in all_events:
                t_key = f"cal_task_{top_crop}_{ev['day_offset']}"
                if t_key not in st.session_state:
                    st.session_state[t_key] = (ev["day_offset"] <= -7)
                if st.session_state.get(t_key, False):
                    completed_tasks += 1

            progress_pct = completed_tasks / total_tasks if total_tasks > 0 else 0.0

            st.markdown(f"""
            <div style="background:white;padding:1.1rem 1.4rem;border-radius:12px;border:1px solid {BORDER_COLOR};margin:0.5rem 0 1.25rem 0;box-shadow:0 1px 4px rgba(0,0,0,0.03);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                    <span style="font-weight:700;color:{PRIMARY};font-size:0.95rem;">🌾 Plot Operational Readiness</span>
                    <span style="font-weight:800;color:{SECONDARY};font-size:1.05rem;">{completed_tasks} of {total_tasks} Operations Completed ({progress_pct:.0%})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(progress_pct)

            col_calendar_main, col_companion = st.columns([1.6, 1], gap="medium")

            with col_calendar_main:
                # -------------------------------------------------------------
                # 1. FIXED-WIDTH ROCK-SOLID MONTHLY CALENDAR TABLE
                # -------------------------------------------------------------
                if calendar_view_mode == "📆 Monthly Calendar View":
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

                    sel_m_idx = st.selectbox(
                        "Select Month to Inspect:",
                        range(len(month_keys)),
                        format_func=lambda idx: month_options[idx],
                        key="sel_month_dropdown"
                    )

                    chosen_y, chosen_m = month_keys[sel_m_idx]
                    month_events = month_map[(chosen_y, chosen_m)]
                    events_by_day = {e["date"].day: e for e in month_events}

                    # Render Fixed-Layout HTML Table (Never collapses on any device!)
                    cal_matrix = calendar.monthcalendar(chosen_y, chosen_m)
                    
                    rows_html = []
                    for week in cal_matrix:
                        tds_html = []
                        for day_num in week:
                            if day_num == 0:
                                tds_html.append("<td style='background:#F8FAFC;border:1px solid #E2E8F0;padding:6px;height:82px;vertical-align:top;'></td>")
                            elif day_num in events_by_day:
                                ev_m = events_by_day[day_num]
                                is_d = st.session_state.get(f"cal_task_{top_crop}_{ev_m['day_offset']}", False)
                                done_tag = "✅ Done" if is_d else "⏳ Action"
                                badge_bg = "#1B4D3E" if not is_d else "#2D6A4F"
                                
                                tds_html.append(f"""
                                <td style='background:#F0FDF4;border:1.5px solid #52B788;padding:6px;height:82px;vertical-align:top;'>
                                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                                        <span style='font-weight:800;color:#1B4D3E;font-size:0.92rem;'>{day_num}</span>
                                        <span style='font-size:0.68rem;background:#DCFCE7;color:#166534;font-weight:700;padding:1px 4px;border-radius:4px;'>{done_tag}</span>
                                    </div>
                                    <div style='background:{badge_bg};color:white;padding:3px 5px;border-radius:4px;font-size:0.7rem;font-weight:600;margin-top:4px;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;' title='{ev_m['what']}'>
                                        {ev_m['stage']}
                                    </div>
                                </td>
                                """)
                            else:
                                tds_html.append(f"""
                                <td style='background:white;border:1px solid #E2E8F0;padding:6px;height:82px;vertical-align:top;'>
                                    <div style='font-weight:700;color:#64748B;font-size:0.85rem;'>{day_num}</div>
                                </td>
                                """)
                        rows_html.append(f"<tr>{''.join(tds_html)}</tr>")

                    table_calendar_html = f"""
                    <div style="margin-bottom:1.5rem;box-shadow:0 2px 10px rgba(0,0,0,0.04);border-radius:10px;overflow:hidden;border:1px solid #CBD5E1;">
                        <div style="background:linear-gradient(135deg, {PRIMARY}, {SECONDARY});color:white;padding:10px 16px;font-weight:700;font-size:1.05rem;">
                            📅 {calendar.month_name[chosen_m]} {chosen_y} — Cropping Schedule
                        </div>
                        <table style="width:100%;border-collapse:collapse;table-layout:fixed;background:white;">
                            <thead>
                                <tr style="background:#E8F5E9;color:{PRIMARY};font-weight:700;font-size:0.82rem;text-align:center;">
                                    <th style="padding:8px 2px;width:14.28%;">Mon</th>
                                    <th style="padding:8px 2px;width:14.28%;">Tue</th>
                                    <th style="padding:8px 2px;width:14.28%;">Wed</th>
                                    <th style="padding:8px 2px;width:14.28%;">Thu</th>
                                    <th style="padding:8px 2px;width:14.28%;">Fri</th>
                                    <th style="padding:8px 2px;width:14.28%;">Sat</th>
                                    <th style="padding:8px 2px;width:14.28%;">Sun</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(rows_html)}
                            </tbody>
                        </table>
                    </div>
                    """
                    st.markdown(table_calendar_html, unsafe_allow_html=True)

                    # Interactive Highly Detailed Action Cards for Month
                    st.markdown(f"#### 📋 Farmer's Operational Action Cards for {calendar.month_name[chosen_m]} {chosen_y}")
                    for ev in month_events:
                        t_key = f"cal_task_{top_crop}_{ev['day_offset']}"
                        is_done = st.session_state.get(t_key, False)
                        p_badge_bg = "#FEE2E2" if ev["priority"] == "Critical" else "#FEF3C7" if ev["priority"] == "High" else "#E0F2FE"
                        p_badge_col = "#991B1B" if ev["priority"] == "Critical" else "#92400E" if ev["priority"] == "High" else "#075985"

                        with st.expander(f"📍 {ev['when']} ({ev['date'].strftime('%a, %b %d')}) — {ev['what']}", expanded=(not is_done)):
                            c_chk, c_body = st.columns([1, 4])
                            with c_chk:
                                st.checkbox("Mark Done", key=t_key)
                                st.markdown(f"""
                                <span style='background:{p_badge_bg};color:{p_badge_col};padding:3px 8px;border-radius:6px;font-size:0.75rem;font-weight:700;'>
                                    {ev['priority']} Priority
                                </span>
                                """, unsafe_allow_html=True)
                            with c_body:
                                st.markdown(f"**Stage:** `{ev['stage']}` • **Category:** `{ev['category']}`")
                                st.markdown(f"**🎯 Exact Dosages & Materials:**  \n{ev['dosage']}")
                                st.markdown(f"**🛠️ Step-by-Step Instructions (HOW):**  \n{ev['how']}")

                                if "Organic" in treatment_mode:
                                    st.info(f"🌱 **Organic Method:** {ev['organic']}")
                                elif "Chemical" in treatment_mode:
                                    st.warning(f"⚗️ **Chemical Treatment:** {ev['chemical']}")
                                else:
                                    st.success(f"⚖️ **IPM Recommended:** {ev['organic']}\n\n*Chemical Backup:* {ev['chemical']}")

                # -------------------------------------------------------------
                # 2. WEEKLY STAGE PROGRESSION
                # -------------------------------------------------------------
                elif calendar_view_mode == "🗓️ Weekly Stage Progression":
                    st.markdown("#### 🗓️ Weekly Chronological Sprint Cards")
                    st.caption("Step-by-step weekly guides covering land prep, vegetative care, flowering, and harvest.")

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

                        w_done = all(st.session_state.get(f"cal_task_{top_crop}_{e['day_offset']}", False) for e in w_events)
                        w_badge = "✅ Completed" if w_done else "⏳ Pending Action"
                        w_badge_bg = "#E8F5E9" if w_done else "#FEF3C7"
                        w_badge_col = PRIMARY if w_done else "#92400E"

                        date_span = f"{w_events[0]['date'].strftime('%b %d')} — {w_events[-1]['date'].strftime('%b %d')}"

                        with st.expander(f"📆 {w_title} [{date_span}]", expanded=(w_idx == 2 or (not w_done and w_idx < 4))):
                            st.markdown(f"<span style='background:{w_badge_bg};color:{w_badge_col};padding:3px 8px;border-radius:6px;font-size:0.75rem;font-weight:700;'>{w_badge}</span>", unsafe_allow_html=True)
                            st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

                            for ev in w_events:
                                t_key = f"cal_task_{top_crop}_{ev['day_offset']}"
                                c_chk, c_body = st.columns([0.8, 4])
                                with c_chk:
                                    st.checkbox(f"Day {ev['day_offset']:+d}", key=t_key)
                                with c_body:
                                    st.markdown(f"**{ev['what']}** ({ev['date'].strftime('%a, %d %b')})")
                                    st.caption(f"**Dosage:** {ev['dosage']}")
                                    st.caption(f"**How:** {ev['how']}")

                # -------------------------------------------------------------
                # 3. DAILY FARMER ACTION CHECKLIST
                # -------------------------------------------------------------
                else:
                    st.markdown("#### 📋 Highly Detailed Farmer Action Checklist (When, What, How)")
                    st.caption("Complete operational roadmap from pre-planting through harvesting.")

                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        status_filter = st.selectbox("Filter Status:", ["All Operations", "Pending Tasks Only", "Completed Tasks Only"], key="chk_status_sel")
                    with f_col2:
                        cat_filter = st.selectbox("Filter Category:", ["All Categories", "Nutrition", "Weed Management", "Irrigation", "Pest & Disease Defense", "Harvesting & Storage", "Soil & Land Prep", "Sowing"], key="chk_cat_sel")

                    filtered_events = all_events
                    if status_filter == "Pending Tasks Only":
                        filtered_events = [e for e in filtered_events if not st.session_state.get(f"cal_task_{top_crop}_{e['day_offset']}", False)]
                    elif status_filter == "Completed Tasks Only":
                        filtered_events = [e for e in filtered_events if st.session_state.get(f"cal_task_{top_crop}_{e['day_offset']}", False)]

                    if cat_filter != "All Categories":
                        filtered_events = [e for e in filtered_events if e["category"].lower() == cat_filter.lower()]

                    if not filtered_events:
                        st.info("No operations match the selected filters.")
                    else:
                        for ev in filtered_events:
                            t_key = f"cal_task_{top_crop}_{ev['day_offset']}"
                            is_done = st.session_state.get(t_key, False)
                            p_col = "#991B1B" if ev["priority"] == "Critical" else "#1B4D3E"

                            with st.container():
                                st.markdown(f"""
                                <div style="background:white;border:1px solid {BORDER_COLOR};border-left:5px solid {p_col};border-radius:8px;padding:1rem;margin-bottom:0.75rem;">
                                    <div style="display:flex;justify-content:space-between;align-items:center;">
                                        <div>
                                            <span style="font-weight:700;color:{PRIMARY};font-size:0.95rem;">{ev['when']} — {ev['date'].strftime('%a, %d %b %Y')}</span>
                                            <span style="background:#F1F5F9;color:{TEXT_MUTED};padding:2px 8px;border-radius:4px;font-size:0.75rem;margin-left:8px;font-weight:600;">{ev['category']}</span>
                                        </div>
                                        <span style="font-size:0.8rem;font-weight:700;color:{p_col};">{ev['priority']} Priority</span>
                                    </div>
                                    <div style="font-weight:700;color:{TEXT_MAIN};margin-top:0.4rem;font-size:1.02rem;">{ev['what']}</div>
                                    <div style="font-size:0.85rem;color:#166534;background:#F0FDF4;padding:6px 10px;border-radius:6px;margin-top:0.4rem;border:1px solid #BBF7D0;">
                                        <b>🎯 Exact Dosages:</b> {ev['dosage']}
                                    </div>
                                    <div style="font-size:0.85rem;color:{TEXT_MAIN};margin-top:0.4rem;">
                                        <b>🛠️ Action Guide (HOW):</b> {ev['how']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.checkbox(f"Mark Operation Completed ({ev['when']})", key=t_key)

            with col_companion:
                st.markdown(f"""
                <div class="content-panel">
                    <div class="panel-header">🌿 Companion Crops & Intercropping Hub</div>
                    <p style="color:{TEXT_MUTED};font-size:0.88rem;">
                        Agro-ecological biological synergy matrix for <b>{top_crop}</b> in <b>Tamil Nadu</b> to boost pest deterrence and land utilization.
                    </p>
                """, unsafe_allow_html=True)
                st.markdown(companion_text)
                st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================================
        # TAB 3: CROP DOCTOR VISION DIAGNOSTICS
        # =====================================================================
        with tab3:
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
        # TAB 4: FARM ANALYTICS & CHAT ASSISTANT
        # =====================================================================
        with tab4:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <div>
                    <h3 style="color:{PRIMARY};margin:0;">📊 Farm History Analytics & SQL Assistant</h3>
                    <p style="color:{TEXT_MUTED};margin:0.2rem 0 0 0;">
                        Automated data loading from SQLite database • Real-time query intelligence
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

            sub1, sub2, sub3 = st.tabs(["📈 Visual Dashboard", "💬 Ask Farm Assistant", "➕ Record New Harvest"])

            with sub1:
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

            with sub2:
                st.markdown("#### 💬 Ask Questions About Your Historical Field Data")
                st.caption("Ask questions like: *'Which crop had highest profit?'*, *'What was my yield in Kharif?'*, *'Average nitrogen used in Tamil Nadu?'*")

                if "history_chat" not in st.session_state:
                    st.session_state.history_chat = [
                        {"role": "assistant", "content": "Vanakkam Murugan! I can answer questions regarding your past yields, profits, and nutrient usage in Tamil Nadu. What would you like to explore?"}
                    ]

                for msg in st.session_state.history_chat:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

                if user_q := st.chat_input("Ask about your farm history..."):
                    st.session_state.history_chat.append({"role": "user", "content": user_q})
                    with st.chat_message("user"):
                        st.write(user_q)

                    q_res = components["sql_agent"].ask_farm_history(user_q)
                    ans = q_res.get("summary", "Could not query records.")

                    st.session_state.history_chat.append({"role": "assistant", "content": ans})
                    with st.chat_message("assistant"):
                        st.write(ans)
                        if q_res.get("query"):
                            st.caption(f"Executed SQL: `{q_res['query']}`")
                        if q_res.get("results"):
                            df_q = pd.DataFrame(q_res["results"])
                            render_styled_table(df_q, max_height="250px", key=f"tbl_chat_{len(st.session_state.history_chat)}")

            with sub3:
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

        sys_tab1, sys_tab2, sys_tab3, sys_tab4 = st.tabs([
            "🗄️ Database Tables Explorer",
            "📚 Policy Documents Viewer",
            "🔍 RAG Retrieval & Relevance Sandbox",
            "📊 ML Model Diagnostics & Telemetry"
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
                st.info("No policy documents found in dataAnalysis/agronomy_docs.")

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
                - **Training Data Scope:** `dataAnalysis/data/crop_yield.csv` (19,689 records), `state_soil_data.csv`, `state_weather_data_1997_2020.csv`
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
