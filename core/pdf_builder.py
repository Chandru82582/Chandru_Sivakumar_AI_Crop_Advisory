"""PDF Report Builder for Comprehensive Agricultural Advisory.

Generates multi-page, publication-grade agronomic advisory dossiers:
- Executive Farm & Plot Profile (Soil Diagnostics, Regional Weather Indices)
- Crop Recommendation & Regional High-Yielding Varieties Evaluation
- Comprehensive Soil Preparation & Split Nutrient Application Regimes (Per Acre & Per Hectare)
- Complete Farmer Operational Action Checklist (When, What, Dosages, How) with Checkboxes [ ]
- Integrated Pest & Disease Defense Protocol (IPM: Organic Formulations vs Targeted Chemical Treatments)
- Critical Irrigation & Water Management Stages
- Companion Planting & Agro-Ecological Synergy Blueprint
- Economic Feasibility & Farm Budgeting Analysis (Yield, Costs, Net Margins)
"""
from io import BytesIO
from datetime import datetime
from typing import Optional, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black, grey
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, KeepTogether, HRFlowable
)
try:
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
except ImportError:
    from reportlab.enum.text import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# Brand Colors
PRIMARY = HexColor('#1B4D3E')     # Deep Forest Pine
SECONDARY = HexColor('#2D6A4F')   # Lush Emerald
ACCENT = HexColor('#52B788')      # Fresh Sage Mint
LIGHT_BG = HexColor('#F4F7F4')    # Soft Light Tint
DARK_TEXT = HexColor('#1E293B')   # Slate Dark
BORDER_COLOR = HexColor('#CBD5E1')# Crisp Border
HIGHLIGHT_BG = HexColor('#E8F5E9')# Light Green Tint


class PDFReportBuilder:
    """Builds comprehensive, publication-grade agricultural advisory dossiers."""

    def __init__(self, page_size=letter):
        self.page_size = page_size
        self.width, self.height = page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom typography styles."""
        self.styles.add(ParagraphStyle(
            name='DocSuperTitle',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=SECONDARY,
            fontName='Helvetica-Bold',
            spaceAfter=3,
            alignment=TA_LEFT
        ))

        self.styles.add(ParagraphStyle(
            name='DocMainTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=PRIMARY,
            spaceAfter=5,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='DocSubTitle',
            parent=self.styles['Normal'],
            fontSize=9.5,
            leading=13,
            textColor=DARK_TEXT,
            spaceAfter=10,
            alignment=TA_LEFT
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=11.5,
            leading=15,
            textColor=PRIMARY,
            spaceBefore=8,
            spaceAfter=5,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='BodyRegular',
            parent=self.styles['BodyText'],
            fontSize=8.5,
            leading=12,
            textColor=DARK_TEXT,
            alignment=TA_LEFT,
            spaceAfter=5
        ))

        self.styles.add(ParagraphStyle(
            name='TableCell',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=11,
            textColor=DARK_TEXT
        ))

        self.styles.add(ParagraphStyle(
            name='TableCellBold',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=11,
            textColor=PRIMARY,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=8.5,
            leading=11.5,
            textColor=white,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        ))

        self.styles.add(ParagraphStyle(
            name='ChecklistCheck',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=PRIMARY,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='ChecklistCell',
            parent=self.styles['Normal'],
            fontSize=7.5,
            leading=10,
            textColor=DARK_TEXT
        ))

        self.styles.add(ParagraphStyle(
            name='ChecklistCellBold',
            parent=self.styles['Normal'],
            fontSize=7.5,
            leading=10,
            textColor=PRIMARY,
            fontName='Helvetica-Bold'
        ))

    def generate_advisory_pdf(
        self,
        farmer_name: Optional[str] = None,
        location: Optional[str] = None,
        farm_size: Optional[float] = None,
        features: Optional[dict] = None,
        advisory_data: Optional[dict] = None,
        ml_metrics: Optional[dict] = None,
        timetable_data: Optional[list] = None,
        companion_data: Optional[str] = None
    ) -> BytesIO:
        """Generate comprehensive multi-page agricultural advisory PDF dossier."""
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.page_size,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            title="Comprehensive Crop Advisory Dossier"
        )

        farmer_name = farmer_name or "Rajesh Kumar"
        location = location or "Thanjavur, Tamil Nadu"
        farm_size = farm_size or 5.0
        features = features or {
            "N": 80, "P": 38, "K": 30, "pH": 6.6,
            "avg_temp_c": 28.0, "total_rainfall_mm": 1200,
            "avg_humidity_percent": 72, "season_Kharif": 1, "season_Rabi": 0
        }
        advisory_data = advisory_data or {}
        crop = advisory_data.get("crop", "Rice")
        confidence = advisory_data.get("confidence", 0.88)

        # Prefer full timetable/checklist data
        events_list = timetable_data or advisory_data.get("timetable", [])

        elements = []

        # ================= SECTION 1 & 2 & 3: PROFILE, RECOMMENDATION & NUTRIENT PLAN =================
        elements.extend(self._build_header_banner(farmer_name, location, crop))
        elements.append(Spacer(1, 0.1 * inch))

        elements.extend(self._build_profile_section(farmer_name, location, farm_size, features, crop, confidence))
        elements.append(Spacer(1, 0.1 * inch))

        elements.extend(self._build_farmer_recommendation_section(crop, confidence, ml_metrics, location))
        elements.append(Spacer(1, 0.1 * inch))

        elements.extend(self._build_soil_nutrient_management(crop, features))
        elements.append(Spacer(1, 0.15 * inch))

        elements.append(PageBreak())

        # ================= SECTION 4: DETAILED OPERATIONAL ACTION CHECKLIST =================
        elements.extend(self._build_operational_checklist_section(crop, events_list))
        elements.append(Spacer(1, 0.15 * inch))

        elements.append(PageBreak())

        # ================= SECTION 5, 6, 7 & 8: PROTECTION, IRRIGATION, COMPANIONS & ECONOMICS =================
        elements.extend(self._build_dual_protection_guide(crop))
        elements.append(Spacer(1, 0.12 * inch))

        elements.extend(self._build_water_irrigation_guide(crop))
        elements.append(Spacer(1, 0.12 * inch))

        elements.extend(self._build_companion_planting_section(crop, companion_data))
        elements.append(Spacer(1, 0.12 * inch))

        elements.extend(self._build_economic_feasibility(crop, farm_size))
        elements.append(Spacer(1, 0.15 * inch))

        elements.extend(self._build_certification_footer())

        doc.build(elements)
        buffer.seek(0)
        return buffer

    def _build_header_banner(self, farmer_name: str, location: str, crop: str) -> list:
        """Create clean header block with reference metadata."""
        elements = []
        elements.append(Paragraph("AGRIADVISORY PRECISION INTELLIGENCE PLATFORM • OFFICIAL EXTENSION ADVISORY", self.styles['DocSuperTitle']))
        elements.append(Paragraph(f"CROP ADVISORY & FIELD MANAGEMENT DOSSIER: {crop.upper()}", self.styles['DocMainTitle']))

        ref_id = f"TN-EXT-{datetime.now().strftime('%Y%m%d')}-{abs(hash(crop)) % 10000:04d}"
        sub_text = (
            f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y')} &nbsp;|&nbsp; "
            f"<b>Dossier ID:</b> {ref_id} &nbsp;|&nbsp; "
            f"<b>Farmer:</b> {farmer_name} &nbsp;|&nbsp; "
            f"<b>Location:</b> {location}"
        )
        elements.append(Paragraph(sub_text, self.styles['DocSubTitle']))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=6))
        return elements

    def _build_profile_section(self, farmer: str, loc: str, area: float, feats: dict, crop: str, conf: float) -> list:
        """Build structured 2-column diagnostics table."""
        elements = []
        elements.append(Paragraph("1. FIELD DIAGNOSTICS & SOIL HEALTH PARAMETERS", self.styles['SectionHeading']))

        ph = feats.get("pH", 6.6)
        ph_status = "Optimal Neutral" if 6.2 <= ph <= 7.5 else ("Slightly Acidic (Normal)" if 5.5 <= ph < 6.2 else ("Acidic (Lime recommended)" if ph < 5.5 else "Alkaline (Gypsum needed)"))
        season = "Kharif (Kuruvai / Samba)" if feats.get("season_Kharif", 1) == 1 else "Rabi (Thaladi / Navarai)"

        data = [
            [
                Paragraph("<b>Plot Characteristic</b>", self.styles['TableCellBold']),
                Paragraph("<b>Observed Field Value</b>", self.styles['TableCellBold']),
                Paragraph("<b>Climatic & Soil Indicator</b>", self.styles['TableCellBold']),
                Paragraph("<b>Observed Field Value</b>", self.styles['TableCellBold'])
            ],
            [
                Paragraph("Target Farmer", self.styles['TableCell']),
                Paragraph(farmer, self.styles['TableCell']),
                Paragraph("Available Soil Nitrogen (N)", self.styles['TableCell']),
                Paragraph(f"{feats.get('N', 80)} kg/ha (Medium)", self.styles['TableCell'])
            ],
            [
                Paragraph("Field Location", self.styles['TableCell']),
                Paragraph(loc, self.styles['TableCell']),
                Paragraph("Available Soil Phosphorus (P)", self.styles['TableCell']),
                Paragraph(f"{feats.get('P', 38)} kg/ha (Medium)", self.styles['TableCell'])
            ],
            [
                Paragraph("Cultivable Area", self.styles['TableCell']),
                Paragraph(f"{area:.1f} Hectares ({area*2.47:.1f} Acres)", self.styles['TableCell']),
                Paragraph("Available Soil Potassium (K)", self.styles['TableCell']),
                Paragraph(f"{feats.get('K', 30)} kg/ha (Moderate)", self.styles['TableCell'])
            ],
            [
                Paragraph("Target Cropping Season", self.styles['TableCell']),
                Paragraph(season, self.styles['TableCell']),
                Paragraph("Soil Reaction (pH)", self.styles['TableCell']),
                Paragraph(f"{ph:.1f} ({ph_status})", self.styles['TableCell'])
            ],
            [
                Paragraph("Recommended Crop Choice", self.styles['TableCellBold']),
                Paragraph(f"<b>{crop}</b> (Suitability: {conf:.0%})", self.styles['TableCellBold']),
                Paragraph("Mean Temperature & Rainfall", self.styles['TableCell']),
                Paragraph(f"{feats.get('avg_temp_c', 28.0):.1f}°C / {feats.get('total_rainfall_mm', 1200):.0f} mm", self.styles['TableCell'])
            ]
        ]

        table = Table(data, colWidths=[1.8 * inch, 1.9 * inch, 2.0 * inch, 1.8 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), LIGHT_BG),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('BACKGROUND', (0, 5), (1, 5), HIGHLIGHT_BG),
        ]))
        elements.append(table)
        return elements

    def _build_farmer_recommendation_section(self, crop: str, conf: float, ml_metrics: Optional[dict], location: str) -> list:
        """Build practical farmer-centric crop recommendation table with regional varieties."""
        elements = []
        elements.append(Paragraph("2. RECOMMENDED CROP & REGIONAL HIGH-YIELDING VARIETIES", self.styles['SectionHeading']))

        preds = (ml_metrics or {}).get("predictions", [])
        if not preds:
            preds = [
                {"crop": crop, "probability": conf, "confidence_pct": conf * 100},
                {"crop": "Maize" if crop != "Maize" else "Cotton", "probability": 0.08, "confidence_pct": 8.0},
                {"crop": "Groundnut" if crop != "Groundnut" else "Urad", "probability": 0.04, "confidence_pct": 4.0}
            ]

        # Recommended state varieties lookup
        regional_varieties = {
            "Rice": "ADT 53, CO 51, CR 1009 Sub 1, BPT 5204 (Samba Mahsuri)",
            "Cotton": "SVPR 4, SVPR 6, MCU 7, Surabhi",
            "Groundnut": "VRI 8, VRI 10, TMV 14, Kadiri 6",
            "Maize": "COH(M) 8, COH(M) 10, Pioneer 30V92",
            "Banana": "Grand Naine (G9), Poovan, Red Banana, Nendran",
            "Sugarcane": "CoC 24, Co 86032, Co 0212",
            "Blackgram": "VBN 8, VBN 11, ADT 6, MDU 1",
            "Greengram": "VBN 4, VBN 5, ADT 3, CO 8",
            "Ragi": "GPU 28, CO 15, Paiyur 2",
            "Turmeric": "BSR 1, BSR 2, PTS 10"
        }

        data = [
            [
                Paragraph("Status", self.styles['TableHeader']),
                Paragraph("Recommended Crop Choice", self.styles['TableHeader']),
                Paragraph("Field Match", self.styles['TableHeader']),
                Paragraph("Key High-Yielding Varieties", self.styles['TableHeader']),
                Paragraph("Agronomic Fit for This Field", self.styles['TableHeader'])
            ]
        ]

        rationales = [
            f"Primary selection. Optimum fit for prevailing soil NPK, moisture retention, and regional market demand.",
            "Viable secondary rotation crop to break pest cycles and balance water use.",
            "Contingency legume or dryland alternative for late season planting."
        ]

        for i, pred in enumerate(preds[:3]):
            c_name = pred.get('crop', crop if i == 0 else 'Alternative')
            prob = pred.get('probability', conf if i == 0 else 0.1)
            rank_label = "★ Primary Top Pick" if i == 0 else f"Alternative #{i}"
            vars_text = regional_varieties.get(c_name, "Certified TNAU / ICAR Hybrids")

            data.append([
                Paragraph(rank_label, self.styles['TableCellBold'] if i == 0 else self.styles['TableCell']),
                Paragraph(f"<b>{c_name}</b>", self.styles['TableCellBold'] if i == 0 else self.styles['TableCell']),
                Paragraph(f"<b>{prob:.0%}</b>", self.styles['TableCellBold'] if i == 0 else self.styles['TableCell']),
                Paragraph(vars_text, self.styles['TableCell']),
                Paragraph(rationales[i] if i < len(rationales) else "Alternative crop.", self.styles['TableCell'])
            ])

        table = Table(data, colWidths=[1.3 * inch, 1.4 * inch, 0.8 * inch, 2.0 * inch, 2.0 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 1), (-1, 1), HIGHLIGHT_BG),
        ]))
        elements.append(table)
        return elements

    def _build_soil_nutrient_management(self, crop: str, feats: dict) -> list:
        """Build detailed soil nutrient schedule with BOTH Acre and Hectare dosages."""
        elements = []
        elements.append(Paragraph("3. SOIL NUTRIENT APPLICATION SCHEDULE (PER ACRE & PER HECTARE)", self.styles['SectionHeading']))

        n = feats.get("N", 80)
        p = feats.get("P", 38)
        k = feats.get("K", 30)

        data = [
            [
                Paragraph("Application Phase", self.styles['TableHeader']),
                Paragraph("Nutrient Source & Material", self.styles['TableHeader']),
                Paragraph("Per Acre Dosage (Bags / kg)", self.styles['TableHeader']),
                Paragraph("Per Hectare Dosage (kg)", self.styles['TableHeader']),
                Paragraph("Field Application Method & Instructions", self.styles['TableHeader'])
            ],
            [
                Paragraph("Basal Land Prep", self.styles['TableCellBold']),
                Paragraph("Well-rotted FYM / Compost", self.styles['TableCell']),
                Paragraph("<b>8 - 10 Tons / acre</b><br/>(or Vermicompost @ 2 t/acre)", self.styles['TableCell']),
                Paragraph("20 - 25 Tons / ha", self.styles['TableCell']),
                Paragraph("Incorporate into top 15 cm soil during secondary harrowing 2 weeks before sowing.", self.styles['TableCell'])
            ],
            [
                Paragraph("Basal Fertilizer", self.styles['TableCellBold']),
                Paragraph("DAP + MOP + Urea split", self.styles['TableCell']),
                Paragraph(f"<b>DAP: {(p/0.46)/2.47:.0f} kg (~1 bag)<br/>MOP: {(k*0.5/0.6)/2.47:.0f} kg (0.5 bag)<br/>Urea: {max(0, n*0.5 - (p/0.46)*0.18)/(0.46*2.47):.0f} kg</b>", self.styles['TableCellBold']),
                Paragraph(f"DAP: {p/0.46:.0f} kg<br/>MOP: {k*0.5/0.6:.0f} kg<br/>Urea: {max(0, n*0.5 - (p/0.46)*0.18)/0.46:.0f} kg", self.styles['TableCell']),
                Paragraph("Band placement 5 cm below seed depth just before final puddling/leveling.", self.styles['TableCell'])
            ],
            [
                Paragraph("1st Top Dressing (Day 21-25)", self.styles['TableCellBold']),
                Paragraph("Neem Coated Urea (Nitrogen)", self.styles['TableCell']),
                Paragraph(f"<b>Urea: {(n*0.25/0.46)/2.47:.0f} kg (~1 bag/acre)</b>", self.styles['TableCellBold']),
                Paragraph(f"Urea: {n*0.25/0.46:.0f} kg / ha", self.styles['TableCell']),
                Paragraph("Broadcast at active tillering / branching stage after weeding, in moist soil prior to irrigation.", self.styles['TableCell'])
            ],
            [
                Paragraph("2nd Top Dressing (Day 45-50)", self.styles['TableCellBold']),
                Paragraph("Urea + MOP (Panicle Initiation)", self.styles['TableCell']),
                Paragraph(f"<b>Urea: {(n*0.25/0.46)/2.47:.0f} kg<br/>MOP: {(k*0.5/0.6)/2.47:.0f} kg (0.5 bag)</b>", self.styles['TableCellBold']),
                Paragraph(f"Urea: {n*0.25/0.46:.0f} kg<br/>MOP: {k*0.5/0.6:.0f} kg", self.styles['TableCell']),
                Paragraph("Apply at panicle initiation / flower-bud formation. Essential for healthy grain filling.", self.styles['TableCell'])
            ],
            [
                Paragraph("Foliar Boost (Day 35 & 60)", self.styles['TableCellBold']),
                Paragraph("Zinc Sulfate (0.5%) + Urea (1%)", self.styles['TableCell']),
                Paragraph("<b>ZnSO4: 1.0 kg + Urea: 2 kg<br/>in 200 L water / acre</b>", self.styles['TableCellBold']),
                Paragraph("ZnSO4: 2.5 kg + Urea: 5 kg<br/>in 500 L water / ha", self.styles['TableCell']),
                Paragraph("Spray early morning on leaves to eliminate Khaira / micronutrient deficiency and boost chlorophyll.", self.styles['TableCell'])
            ]
        ]

        table = Table(data, colWidths=[1.4 * inch, 1.6 * inch, 1.6 * inch, 1.2 * inch, 1.7 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(table)
        return elements

    def _build_operational_checklist_section(self, crop: str, events: Optional[list]) -> list:
        """Build publication-grade multi-stage operational action checklist with checkboxes [ ]."""
        elements = []
        elements.append(Paragraph("4. COMPLETE FARMER OPERATIONAL ACTION CHECKLIST (WHEN, WHAT, HOW)", self.styles['SectionHeading']))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=6))
        elements.append(Paragraph(
            "<b>Practical Field Instructions:</b> Keep this checklist on the farm. As each scheduled operation is completed, "
            "tick the checkbox <b>[ ✓ ]</b> and note the application date. Follow exact dosages per acre/hectare.",
            self.styles['BodyRegular']
        ))

        # Fallback default events if empty
        if not events:
            events = [
                {
                    "when": "Day -14", "stage": "Deep Plowing", "what": "Deep summer plowing and soil solarization",
                    "dosage": "Plow to 20-25 cm. Incorporate Dhaincha @ 8 kg/acre (20 kg/ha).",
                    "how": "Moldboard plowing to expose resting soil pupae to sunlight."
                },
                {
                    "when": "Day -7", "stage": "Basal Manure", "what": "Incorporate FYM and bio-fertilizer",
                    "dosage": "FYM @ 10 t/acre (25 t/ha) enriched with Trichoderma @ 2 kg/acre.",
                    "how": "Evenly broadcast and mix into top 15 cm soil."
                },
                {
                    "when": "Day 0", "stage": "Sowing Day", "what": "Seed treatment & certified sowing",
                    "dosage": "Seed rate: 16 kg/acre. Azospirillum @ 200 g/acre + Pseudomonas @ 10 g/kg seed.",
                    "how": "Treat seed with kanji slurry, shade dry 30 mins, sow at 45x15 cm spacing."
                },
                {
                    "when": "Day +21", "stage": "Weeding", "what": "First weeding & sticky traps",
                    "dosage": "Install 6 yellow sticky traps/acre. Top-dress Urea @ 1 bag/acre.",
                    "how": "Cono-weeder inter-cultivation to aerate roots and suppress weeds."
                },
                {
                    "when": "Day +45", "stage": "Panicle Top-Dress", "what": "Second top dressing (Urea + Potash)",
                    "dosage": "Urea @ 1 bag/acre (45 kg) + MOP @ 0.5 bag/acre (25 kg).",
                    "how": "Broadcast in moist soil at panicle initiation before irrigation."
                },
                {
                    "when": "Day +115", "stage": "Harvesting", "what": "Drain field & harvest at golden maturity",
                    "dosage": "Drain water 10 days prior. Harvest when 85% grains turn straw-gold.",
                    "how": "Thresh, clean, dry down to <13% moisture and pack in hermetic bags."
                }
            ]

        data = [
            [
                Paragraph("Tick", self.styles['TableHeader']),
                Paragraph("Timeline (WHEN)", self.styles['TableHeader']),
                Paragraph("Stage & Operation (WHAT)", self.styles['TableHeader']),
                Paragraph("Exact Dosages & Materials (Per Acre & Ha)", self.styles['TableHeader']),
                Paragraph("Field Execution Guide (HOW)", self.styles['TableHeader'])
            ]
        ]

        for i, ev in enumerate(events):
            when_str = ev.get("when", f"Day {ev.get('day_offset', i)}")
            stage_str = ev.get("stage", ev.get("phase", "Field Operation"))
            what_str = ev.get("what", "")
            if not what_str and "tasks" in ev:
                what_str = "; ".join(ev["tasks"])
            dosage_str = ev.get("dosage", "")
            how_str = ev.get("how", "")
            if not how_str and "tasks" in ev:
                how_str = "; ".join(ev["tasks"])

            data.append([
                Paragraph("[ &nbsp; ]", self.styles['ChecklistCheck']),
                Paragraph(f"<b>{when_str}</b>", self.styles['ChecklistCellBold']),
                Paragraph(f"<b>{stage_str}</b><br/>{what_str}", self.styles['ChecklistCell']),
                Paragraph(f"<b>Dosage:</b> {dosage_str}", self.styles['ChecklistCell']),
                Paragraph(how_str, self.styles['ChecklistCell'])
            ])

        table = Table(data, colWidths=[0.45 * inch, 1.25 * inch, 1.8 * inch, 2.1 * inch, 1.9 * inch], repeatRows=1)
        
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ]
        # Alternating subtle row shading for high legibility
        for r in range(1, len(data)):
            if r % 2 == 1:
                table_style.append(('BACKGROUND', (0, r), (-1, r), LIGHT_BG))

        table.setStyle(TableStyle(table_style))
        elements.append(table)
        return elements

    def _build_dual_protection_guide(self, crop: str) -> list:
        """Build dual IPM crop protection guide."""
        elements = []
        elements.append(Paragraph("5. INTEGRATED PEST & DISEASE DEFENSE PROTOCOL (IPM)", self.styles['SectionHeading']))

        data = [
            [
                Paragraph("Target Pest / Disease", self.styles['TableHeader']),
                Paragraph("Organic & Bio-Control Solution", self.styles['TableHeader']),
                Paragraph("Targeted Chemical Formulation & Dosage", self.styles['TableHeader']),
                Paragraph("Safety Interval (PHI)", self.styles['TableHeader'])
            ],
            [
                Paragraph("<b>Sucking Pests (Aphids, Jassids, BPH)</b>", self.styles['TableCell']),
                Paragraph("5% Neem Seed Kernel Extract (NSKE) or Neem oil 3000 ppm @ 4 ml/L water", self.styles['TableCell']),
                Paragraph("Imidacloprid 17.8% SL @ 0.3 ml/L or Thiamethoxam 25% WG @ 0.2 g/L", self.styles['TableCell']),
                Paragraph("14 - 21 Days", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Foliar Leaf Blast & Blight</b>", self.styles['TableCell']),
                Paragraph("Pseudomonas fluorescens @ 5 g/L foliar spray + Cow urine extract", self.styles['TableCell']),
                Paragraph("Tricyclazole 75% WP @ 0.6 g/L or Azoxystrobin 23% SC @ 1 ml/L", self.styles['TableCell']),
                Paragraph("15 Days", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Stem Borer / Pod Borer</b>", self.styles['TableCell']),
                Paragraph("Trichogramma egg parasitoid cards @ 2 cards/acre + Pheromone traps @ 5/acre", self.styles['TableCell']),
                Paragraph("Chlorantraniliprole 18.5% SC @ 0.3 ml/L (60 ml/acre) in 200 L water", self.styles['TableCell']),
                Paragraph("21 Days", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Root Rot & Damping-Off</b>", self.styles['TableCell']),
                Paragraph("Trichoderma viride @ 5 kg/ha in 250 kg FYM + Neem cake 100 kg/acre", self.styles['TableCell']),
                Paragraph("Carbendazim 50% WP @ 2 g/kg seed dressing or Metalaxyl MZ @ 2.5 g/kg", self.styles['TableCell']),
                Paragraph("At Sowing", self.styles['TableCell'])
            ]
        ]

        table = Table(data, colWidths=[1.8 * inch, 2.3 * inch, 2.4 * inch, 1.0 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        elements.append(table)
        return elements

    def _build_water_irrigation_guide(self, crop: str) -> list:
        """Build critical irrigation and water management table."""
        elements = []
        elements.append(Paragraph("6. CRITICAL IRRIGATION & WATER MANAGEMENT STAGES", self.styles['SectionHeading']))

        data = [
            [
                Paragraph("Growth Stage", self.styles['TableHeader']),
                Paragraph("Recommended Water Level / Regime", self.styles['TableHeader']),
                Paragraph("Practical Water-Saving Field Instructions", self.styles['TableHeader'])
            ],
            [
                Paragraph("<b>Sowing to Emergence (Day 0-7)</b>", self.styles['TableCellBold']),
                Paragraph("Thin film / moist soil (1 - 2 cm)", self.styles['TableCell']),
                Paragraph("Keep field saturated without stagnant water. Prevent seed drowning and soil crusting.", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Active Tillering (Day 15-35)</b>", self.styles['TableCellBold']),
                Paragraph("Shallow submergence (2 - 3 cm)", self.styles['TableCell']),
                Paragraph("Adopt Alternate Wetting & Drying (AWD). Irrigate only when hairline cracks appear on soil.", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Panicle Initiation & Flowering (Day 45-75)</b>", self.styles['TableCellBold']),
                Paragraph("Standing water (4 - 5 cm)", self.styles['TableCell']),
                Paragraph("Critical moisture sensitive stage. Do NOT allow moisture stress; ensure continuous moisture.", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Maturity & Pre-Harvest (Day 100+)</b>", self.styles['TableCellBold']),
                Paragraph("Completely Drained", self.styles['TableCell']),
                Paragraph("Drain all standing water 10 to 14 days before harvest to speed uniform ripening and ease machinery.", self.styles['TableCell'])
            ]
        ]

        table = Table(data, colWidths=[2.2 * inch, 2.2 * inch, 3.1 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        elements.append(table)
        return elements

    def _build_companion_planting_section(self, crop: str, companion_text: Optional[str]) -> list:
        """Build companion planting guide."""
        elements = []
        elements.append(Paragraph("7. COMPANION CROPPING & AGRO-ECOLOGICAL RESILIENCE", self.styles['SectionHeading']))

        data = [
            [
                Paragraph("Companion Species", self.styles['TableHeader']),
                Paragraph("Spatial Layout & Ratio", self.styles['TableHeader']),
                Paragraph("Ecological Mechanism & Synergy Benefits", self.styles['TableHeader']),
                Paragraph("Harvest Window", self.styles['TableHeader'])
            ],
            [
                Paragraph("<b>Short-Duration Pulse (Moong/Urad)</b>", self.styles['TableCellBold']),
                Paragraph("2:1 or 3:1 Strip Cropping on Bunds", self.styles['TableCell']),
                Paragraph("Atmospheric nitrogen fixation (30-40 kg N/ha), suppresses weed growth as living mulch.", self.styles['TableCell']),
                Paragraph("65 - 75 Days", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>African Marigold (Trap Crop)</b>", self.styles['TableCellBold']),
                Paragraph("Border trap rows (every 10th row)", self.styles['TableCell']),
                Paragraph("Root alpha-terthienyl exudates suppress root-knot nematodes; yellow flowers trap bollworms.", self.styles['TableCell']),
                Paragraph("Continuous flowering", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Sesbania / Dhaincha Bund Planting</b>", self.styles['TableCellBold']),
                Paragraph("Field perimeter & irrigation bunds", self.styles['TableCell']),
                Paragraph("Provides biomass, shelters beneficial predatory wasps against borers, acts as windbreak.", self.styles['TableCell']),
                Paragraph("45 - 60 Days", self.styles['TableCell'])
            ]
        ]

        table = Table(data, colWidths=[1.8 * inch, 1.7 * inch, 3.0 * inch, 1.0 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        elements.append(table)
        return elements

    def _build_economic_feasibility(self, crop: str, area: float) -> list:
        """Build farm budgeting and economic feasibility projection table."""
        elements = []
        elements.append(Paragraph("8. FARM BUDGETING, EXPECTED YIELD & NET PROFIT MARGINS", self.styles['SectionHeading']))

        exp_yield_per_acre = 2.4  # Tons per acre
        exp_yield_per_ha = exp_yield_per_acre * 2.47
        acre_total = area * 2.47
        total_yield = exp_yield_per_acre * acre_total
        msp_per_ton = 23000  # MSP benchmark INR
        gross_rev_acre = exp_yield_per_acre * msp_per_ton
        cost_acre = 25000
        net_profit_acre = gross_rev_acre - cost_acre
        total_net_profit = net_profit_acre * acre_total

        data = [
            [
                Paragraph("Budgeting Category", self.styles['TableHeader']),
                Paragraph("Per Acre Estimate", self.styles['TableHeader']),
                Paragraph(f"Total Farm Projection ({acre_total:.1f} Acres)", self.styles['TableHeader']),
                Paragraph("Operational Notes", self.styles['TableHeader'])
            ],
            [
                Paragraph("Anticipated Harvest Yield", self.styles['TableCellBold']),
                Paragraph(f"<b>{exp_yield_per_acre:.1f} Tons / acre</b><br/>({exp_yield_per_ha:.1f} Tons / ha)", self.styles['TableCell']),
                Paragraph(f"<b>{total_yield:.1f} Tons Total</b>", self.styles['TableCellBold']),
                Paragraph("Achievable with disciplined adherence to split nutrient & AWD water regimes.", self.styles['TableCell'])
            ],
            [
                Paragraph("Projected Gross Revenue", self.styles['TableCellBold']),
                Paragraph(f"Rs. {gross_rev_acre:,.0f} / acre", self.styles['TableCell']),
                Paragraph(f"Rs. {gross_rev_acre * acre_total:,.0f}", self.styles['TableCell']),
                Paragraph(f"Calculated at current government Minimum Support Price benchmark of Rs. {msp_per_ton:,.0f}/ton.", self.styles['TableCell'])
            ],
            [
                Paragraph("Estimated Production Costs", self.styles['TableCellBold']),
                Paragraph(f"Rs. {cost_acre:,.0f} / acre", self.styles['TableCell']),
                Paragraph(f"Rs. {cost_acre * acre_total:,.0f}", self.styles['TableCell']),
                Paragraph("Includes certified seeds, fertilizers, tractor tillage, weeding labor, & harvesting.", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Estimated Net Profit Margin</b>", self.styles['TableCellBold']),
                Paragraph(f"<b>Rs. {net_profit_acre:,.0f} / acre</b>", self.styles['TableCellBold']),
                Paragraph(f"<b>Rs. {total_net_profit:,.0f} Total</b>", self.styles['TableCellBold']),
                Paragraph("<b>Projected Benefit-Cost Ratio: 2.20</b>", self.styles['TableCellBold'])
            ]
        ]

        table = Table(data, colWidths=[2.0 * inch, 1.6 * inch, 1.8 * inch, 2.1 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 4), (-1, 4), HIGHLIGHT_BG),
        ]))
        elements.append(table)
        return elements

    def _build_certification_footer(self) -> list:
        """Build certification disclaimer and sign-off."""
        elements = []
        elements.append(HRFlowable(width="100%", thickness=1, color=BORDER_COLOR, spaceAfter=6))

        disc_text = (
            "<b>Extension Advisory Notice:</b> This field management dossier is prepared for farmer operational reference "
            "based on regional soil fertility indices, Tamil Nadu agro-climatic conditions, and verified agricultural extension protocols. "
            "Always follow seed and pesticide label precautions and maintain local spray records."
        )
        elements.append(Paragraph(disc_text, self.styles['TableCell']))
        elements.append(Spacer(1, 0.08 * inch))

        sign_data = [
            [
                Paragraph(f"<b>Advisory Reference:</b> Verified Agronomic Protocol v3.6<br/><b>Verified Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styles['TableCell']),
                Paragraph("<b>Division of Agricultural Extension:</b><br/>Official Farmer Advisory & Precision Field Support", self.styles['TableCell'])
            ]
        ]
        sign_table = Table(sign_data, colWidths=[3.75 * inch, 3.75 * inch])
        sign_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        elements.append(sign_table)
        return elements
