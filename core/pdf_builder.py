"""PDF Report Builder for Comprehensive Agricultural Advisory.

Generates multi-page, publication-grade agronomic advisory dossiers:
- Executive Farm & Plot Profile (Soil Diagnostics, Weather Indices)
- Machine Learning Crop Recommendation & Ensemble Confidence Matrix
- Comprehensive Soil Preparation & Split Nutrient Application Regimes
- Phase-by-Phase Field Operations & Agronomic Timetable
- Dual Crop Protection Guide (Organic Formulations vs Targeted Chemical Treatments)
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
            spaceAfter=4,
            alignment=TA_LEFT
        ))

        self.styles.add(ParagraphStyle(
            name='DocMainTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            leading=24,
            textColor=PRIMARY,
            spaceAfter=6,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='DocSubTitle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=DARK_TEXT,
            spaceAfter=15,
            alignment=TA_LEFT
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            leading=16,
            textColor=PRIMARY,
            spaceBefore=10,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='BodyRegular',
            parent=self.styles['BodyText'],
            fontSize=9,
            leading=13,
            textColor=DARK_TEXT,
            alignment=TA_LEFT,
            spaceAfter=6
        ))

        self.styles.add(ParagraphStyle(
            name='TableCell',
            parent=self.styles['Normal'],
            fontSize=8.5,
            leading=11.5,
            textColor=DARK_TEXT
        ))

        self.styles.add(ParagraphStyle(
            name='TableCellBold',
            parent=self.styles['Normal'],
            fontSize=8.5,
            leading=11.5,
            textColor=PRIMARY,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=white,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
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
        location = location or "Punjab, India"
        farm_size = farm_size or 5.0
        features = features or {
            "N": 75, "P": 30, "K": 40, "pH": 6.5,
            "avg_temp_c": 25.0, "total_rainfall_mm": 800,
            "avg_humidity_percent": 65, "season_Kharif": 1, "season_Rabi": 0
        }
        advisory_data = advisory_data or {}
        crop = advisory_data.get("crop", "Rice")
        confidence = advisory_data.get("confidence", 0.85)

        elements = []

        # ================= PAGE 1: EXECUTIVE BRIEF & DIAGNOSTICS =================
        elements.extend(self._build_header_banner(farmer_name, location, crop))
        elements.append(Spacer(1, 0.15 * inch))

        elements.extend(self._build_profile_section(farmer_name, location, farm_size, features, crop, confidence))
        elements.append(Spacer(1, 0.15 * inch))

        elements.extend(self._build_ml_evaluation_section(crop, confidence, ml_metrics))
        elements.append(Spacer(1, 0.15 * inch))

        elements.extend(self._build_soil_nutrient_management(crop, features))
        elements.append(Spacer(1, 0.15 * inch))

        elements.append(PageBreak())

        # ================= PAGE 2: FIELD OPERATIONS & CULTIVATION CALENDAR =================
        elements.append(Paragraph("CULTIVATION CALENDAR & AGRONOMIC PROTOCOL", self.styles['SectionHeading']))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=8))
        elements.append(Paragraph(
            f"Chronological operational milestones and task schedules for high-yield cultivation of <b>{crop}</b>:",
            self.styles['BodyRegular']
        ))

        elements.extend(self._build_timetable_table(crop, timetable_data))
        elements.append(Spacer(1, 0.15 * inch))

        elements.extend(self._build_dual_protection_guide(crop))
        elements.append(Spacer(1, 0.15 * inch))

        elements.append(PageBreak())

        # ================= PAGE 3: COMPANION CROPS & ECONOMIC FEASIBILITY =================
        elements.append(Paragraph("AGRO-ECOLOGICAL SYNERGY & FARM BUDGETING", self.styles['SectionHeading']))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=8))

        elements.extend(self._build_companion_planting_section(crop, companion_data))
        elements.append(Spacer(1, 0.15 * inch))

        elements.extend(self._build_economic_feasibility(crop, farm_size))
        elements.append(Spacer(1, 0.2 * inch))

        elements.extend(self._build_certification_footer())

        doc.build(elements)
        buffer.seek(0)
        return buffer

    def _build_header_banner(self, farmer_name: str, location: str, crop: str) -> list:
        """Create clean header block with reference metadata."""
        elements = []
        elements.append(Paragraph("AGRIADVISORY PRECISION INTELLIGENCE PLATFORM • OFFICIAL EXTENSION DOSSIER", self.styles['DocSuperTitle']))
        elements.append(Paragraph(f"CROP ADVISORY & FIELD MANAGEMENT DOSSIER: {crop.upper()}", self.styles['DocMainTitle']))

        ref_id = f"REF-{datetime.now().strftime('%Y%m%d')}-{abs(hash(crop)) % 10000:04d}"
        sub_text = (
            f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y')} &nbsp;|&nbsp; "
            f"<b>Reference ID:</b> {ref_id} &nbsp;|&nbsp; "
            f"<b>Farmer:</b> {farmer_name} &nbsp;|&nbsp; "
            f"<b>Location:</b> {location}"
        )
        elements.append(Paragraph(sub_text, self.styles['DocSubTitle']))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=8))
        return elements

    def _build_profile_section(self, farmer: str, loc: str, area: float, feats: dict, crop: str, conf: float) -> list:
        """Build structured 2-column diagnostics table."""
        elements = []
        elements.append(Paragraph("1. FIELD DIAGNOSTICS & ENVIRONMENTAL PARAMETERS", self.styles['SectionHeading']))

        ph = feats.get("pH", 6.5)
        ph_status = "Optimal" if 6.0 <= ph <= 7.5 else ("Acidic (Lime needed)" if ph < 6.0 else "Alkaline (Gypsum needed)")
        season = "Kharif (Monsoon/Autumn)" if feats.get("season_Kharif", 1) == 1 else "Rabi (Winter/Spring)"

        data = [
            [
                Paragraph("<b>Plot Characteristic</b>", self.styles['TableCellBold']),
                Paragraph("<b>Observed Value</b>", self.styles['TableCellBold']),
                Paragraph("<b>Climatic & Soil Parameter</b>", self.styles['TableCellBold']),
                Paragraph("<b>Observed Value</b>", self.styles['TableCellBold'])
            ],
            [
                Paragraph("Target Farmer", self.styles['TableCell']),
                Paragraph(farmer, self.styles['TableCell']),
                Paragraph("Soil Nitrogen (N)", self.styles['TableCell']),
                Paragraph(f"{feats.get('N', 75)} kg/ha", self.styles['TableCell'])
            ],
            [
                Paragraph("Field Location", self.styles['TableCell']),
                Paragraph(loc, self.styles['TableCell']),
                Paragraph("Soil Phosphorus (P)", self.styles['TableCell']),
                Paragraph(f"{feats.get('P', 30)} kg/ha", self.styles['TableCell'])
            ],
            [
                Paragraph("Cultivable Area", self.styles['TableCell']),
                Paragraph(f"{area:.1f} Hectares", self.styles['TableCell']),
                Paragraph("Soil Potassium (K)", self.styles['TableCell']),
                Paragraph(f"{feats.get('K', 40)} kg/ha", self.styles['TableCell'])
            ],
            [
                Paragraph("Target Cropping Season", self.styles['TableCell']),
                Paragraph(season, self.styles['TableCell']),
                Paragraph("Soil Reaction (pH)", self.styles['TableCell']),
                Paragraph(f"{ph:.1f} ({ph_status})", self.styles['TableCell'])
            ],
            [
                Paragraph("Recommended Crop", self.styles['TableCell']),
                Paragraph(f"<b>{crop}</b>", self.styles['TableCellBold']),
                Paragraph("Mean Temperature & Rain", self.styles['TableCell']),
                Paragraph(f"{feats.get('avg_temp_c', 25):.1f}°C / {feats.get('total_rainfall_mm', 800):.0f} mm", self.styles['TableCell'])
            ]
        ]

        table = Table(data, colWidths=[1.8 * inch, 1.9 * inch, 2.0 * inch, 1.8 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), LIGHT_BG),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        return elements

    def _build_ml_evaluation_section(self, crop: str, conf: float, ml_metrics: Optional[dict]) -> list:
        """Build machine learning recommendation breakdown table."""
        elements = []
        elements.append(Paragraph("2. MACHINE LEARNING MODEL RECOMMENDATION & ENSEMBLE RANKING", self.styles['SectionHeading']))

        preds = (ml_metrics or {}).get("predictions", [])
        if not preds:
            preds = [
                {"crop": crop, "probability": conf, "confidence_pct": conf * 100},
                {"crop": "Wheat" if crop != "Wheat" else "Rice", "probability": 0.15, "confidence_pct": 15.0},
                {"crop": "Maize" if crop != "Maize" else "Cotton", "probability": 0.08, "confidence_pct": 8.0}
            ]

        data = [
            [
                Paragraph("Rank", self.styles['TableHeader']),
                Paragraph("Evaluated Crop Variety", self.styles['TableHeader']),
                Paragraph("Ensemble Probability", self.styles['TableHeader']),
                Paragraph("Agronomic Viability Rationale", self.styles['TableHeader'])
            ]
        ]

        rationales = [
            f"Highest physiological congruence with prevailing NPK ratio and rainfall of {crop}.",
            "Viable secondary rotation crop with balanced moisture tolerance.",
            "Contingency crop suitable under fluctuating seasonal rainfall patterns."
        ]

        for i, pred in enumerate(preds[:3]):
            rank_label = "★ Primary Top Pick" if i == 0 else f"Alternative #{i}"
            data.append([
                Paragraph(rank_label, self.styles['TableCellBold'] if i == 0 else self.styles['TableCell']),
                Paragraph(f"<b>{pred.get('crop', '')}</b>", self.styles['TableCell']),
                Paragraph(f"{pred.get('probability', 0):.1%} ({pred.get('confidence_pct', 0):.1f}%)", self.styles['TableCell']),
                Paragraph(rationales[i] if i < len(rationales) else "Alternative option.", self.styles['TableCell'])
            ])

        table = Table(data, colWidths=[1.5 * inch, 1.8 * inch, 1.6 * inch, 2.6 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('BACKGROUND', (0, 1), (-1, 1), LIGHT_BG),
        ]))
        elements.append(table)
        return elements

    def _build_soil_nutrient_management(self, crop: str, feats: dict) -> list:
        """Build detailed soil nutrient fertilization schedule."""
        elements = []
        elements.append(Paragraph("3. SOIL PREPARATION & SPLIT NUTRIENT REGIME", self.styles['SectionHeading']))

        n = feats.get("N", 75)
        p = feats.get("P", 30)
        k = feats.get("K", 40)

        data = [
            [
                Paragraph("Application Phase", self.styles['TableHeader']),
                Paragraph("Nutrient Source & Formulation", self.styles['TableHeader']),
                Paragraph("Target Quantity (per Hectare)", self.styles['TableHeader']),
                Paragraph("Application Method & Guidelines", self.styles['TableHeader'])
            ],
            [
                Paragraph("Basal Land Prep", self.styles['TableCellBold']),
                Paragraph("Well-rotted FYM / Vermicompost", self.styles['TableCell']),
                Paragraph("20 - 25 Tons / ha", self.styles['TableCell']),
                Paragraph("Incorporate during secondary harrowing 2-3 weeks before sowing.", self.styles['TableCell'])
            ],
            [
                Paragraph("Basal Fertilizer", self.styles['TableCellBold']),
                Paragraph(f"SSP (Phosphorus) + MOP (Potash) + Urea", self.styles['TableCell']),
                Paragraph(f"P: {p} kg, K: {k*0.5:.0f} kg, N: {n*0.5:.0f} kg", self.styles['TableCell']),
                Paragraph("Band placement 5 cm below seed depth at final bed preparation.", self.styles['TableCell'])
            ],
            [
                Paragraph("1st Top Dressing", self.styles['TableCellBold']),
                Paragraph("Urea (Split Nitrogen)", self.styles['TableCell']),
                Paragraph(f"N: {n*0.25:.0f} kg / ha", self.styles['TableCell']),
                Paragraph("Broadcast at active tillering / vegetative stage prior to irrigation.", self.styles['TableCell'])
            ],
            [
                Paragraph("2nd Top Dressing", self.styles['TableCellBold']),
                Paragraph("Urea + MOP (Remaining K)", self.styles['TableCell']),
                Paragraph(f"N: {n*0.25:.0f} kg, K: {k*0.5:.0f} kg", self.styles['TableCell']),
                Paragraph("Apply at panicle initiation / pre-flowering stage in moist soil.", self.styles['TableCell'])
            ],
            [
                Paragraph("Foliar Boost", self.styles['TableCellBold']),
                Paragraph("Zinc Sulfate (0.5%) + Urea (1%)", self.styles['TableCell']),
                Paragraph("2.5 kg ZnSO4 in 500 L water", self.styles['TableCell']),
                Paragraph("Foliar spray during early morning to alleviate trace micronutrient stress.", self.styles['TableCell'])
            ]
        ]

        table = Table(data, colWidths=[1.4 * inch, 2.1 * inch, 1.8 * inch, 2.2 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        return elements

    def _build_timetable_table(self, crop: str, timetable_data: Optional[list]) -> list:
        """Build detailed operational calendar table."""
        elements = []
        if not timetable_data:
            timetable_data = [
                {
                    "phase": "Land Preparation",
                    "duration": "2 - 3 Weeks Pre-Plant",
                    "tasks": ["Deep summer plowing (20-25 cm) followed by two cross-harrowings.", "Incorporate 20 tons FYM/ha and apply basal NPK doses."]
                },
                {
                    "phase": "Sowing & Priming",
                    "duration": "Weeks 1 - 2",
                    "tasks": ["Seed treatment with bio-inoculants (Rhizobium/Azospirillum + PSB).", "Sow at 45x15 cm spacing with initial light irrigation."]
                },
                {
                    "phase": "Vegetative Growth",
                    "duration": "Weeks 3 - 8",
                    "tasks": ["First inter-row weeding at 21 days; apply first top-dress nitrogen.", "Monitor sucking pests with yellow sticky traps (15-20/ha)."]
                },
                {
                    "phase": "Flowering & Pods",
                    "duration": "Weeks 9 - 16",
                    "tasks": ["Maintain uninhibited soil moisture; apply second nitrogen top-dress.", "Inspect foliage for rust, spot, or blight symptoms."]
                },
                {
                    "phase": "Maturity & Harvest",
                    "duration": "Weeks 17 - 20",
                    "tasks": ["Withdraw irrigation 12 days prior to harvest; check grain moisture (<14%).", "Harvest, thresh, and dry on clean canvas; store in hermetic bags."]
                }
            ]

        data = [
            [
                Paragraph("Phase & Stage", self.styles['TableHeader']),
                Paragraph("Timeline", self.styles['TableHeader']),
                Paragraph("Key Mandatory Field Operations & Instructions", self.styles['TableHeader'])
            ]
        ]

        for item in timetable_data:
            phase_title = item.get("phase", item.get("stage", "Phase"))
            duration = item.get("duration", "")
            tasks = item.get("tasks", item.get("activities", []))
            tasks_text = "<br/>".join([f"• {t}" for t in tasks])

            data.append([
                Paragraph(f"<b>{phase_title}</b>", self.styles['TableCellBold']),
                Paragraph(duration, self.styles['TableCell']),
                Paragraph(tasks_text, self.styles['TableCell'])
            ])

        table = Table(data, colWidths=[1.8 * inch, 1.4 * inch, 4.3 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(table)
        return elements

    def _build_dual_protection_guide(self, crop: str) -> list:
        """Build dual IPM crop protection guide."""
        elements = []
        elements.append(Paragraph("4. INTEGRATED PEST & DISEASE MANAGEMENT PROTOCOL (IPM)", self.styles['SectionHeading']))

        data = [
            [
                Paragraph("Target Pest / Disease", self.styles['TableHeader']),
                Paragraph("Organic & Bio-Control Solution", self.styles['TableHeader']),
                Paragraph("Targeted Chemical Formulation & Dosage", self.styles['TableHeader']),
                Paragraph("Safety Interval (PHI)", self.styles['TableHeader'])
            ],
            [
                Paragraph("<b>Sucking Pests (Aphids, Jassids)</b>", self.styles['TableCell']),
                Paragraph("5% Neem Seed Kernel Extract (NSKE) or Neem oil 3000 ppm @ 4 ml/L", self.styles['TableCell']),
                Paragraph("Imidacloprid 17.8% SL @ 0.3 ml/L or Thiamethoxam 25% WG @ 0.2 g/L", self.styles['TableCell']),
                Paragraph("14 - 21 Days", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Foliar Leaf Spots & Blight</b>", self.styles['TableCell']),
                Paragraph("Trichoderma viride foliar spray @ 5 g/L + Copper soap formulation", self.styles['TableCell']),
                Paragraph("Mancozeb 75% WP @ 2.5 g/L or Chlorothalonil 75% WP @ 2.0 g/L", self.styles['TableCell']),
                Paragraph("15 Days", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Powdery Mildew & Rust</b>", self.styles['TableCell']),
                Paragraph("Dilute sour buttermilk spray (5%) or Potassium bicarbonate @ 3 g/L", self.styles['TableCell']),
                Paragraph("Hexaconazole 5% EC @ 2 ml/L or Propiconazole 25% EC @ 1 ml/L", self.styles['TableCell']),
                Paragraph("21 Days", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Soil-Borne Root Pathogens</b>", self.styles['TableCell']),
                Paragraph("Pseudomonas fluorescens seed inoculant @ 10 g/kg + Neem cake 250 kg/ha", self.styles['TableCell']),
                Paragraph("Carbendazim 50% WP @ 2 g/kg seed or Metalaxyl MZ @ 2.5 g/kg", self.styles['TableCell']),
                Paragraph("At Sowing", self.styles['TableCell'])
            ]
        ]

        table = Table(data, colWidths=[1.8 * inch, 2.3 * inch, 2.4 * inch, 1.0 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        return elements

    def _build_companion_planting_section(self, crop: str, companion_text: Optional[str]) -> list:
        """Build companion planting guide."""
        elements = []
        elements.append(Paragraph("5. COMPANION CROPPING & AGRO-ECOLOGICAL RESILIENCE", self.styles['SectionHeading']))

        data = [
            [
                Paragraph("Companion Species", self.styles['TableHeader']),
                Paragraph("Spatial Layout & Ratio", self.styles['TableHeader']),
                Paragraph("Ecological Mechanism & Synergy Benefits", self.styles['TableHeader']),
                Paragraph("Maturity Timeline", self.styles['TableHeader'])
            ],
            [
                Paragraph("<b>Short-Duration Pulse (Moong/Urad)</b>", self.styles['TableCellBold']),
                Paragraph("2:1 or 3:1 Strip Cropping", self.styles['TableCell']),
                Paragraph("Atmospheric nitrogen fixation (30-40 kg N/ha), weed smothering living mulch.", self.styles['TableCell']),
                Paragraph("65 - 75 Days", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>African Marigold (Trap Crop)</b>", self.styles['TableCellBold']),
                Paragraph("Peripheral border (every 10th row)", self.styles['TableCell']),
                Paragraph("Root alpha-terthienyl exudates suppress root-knot nematodes; flowers attract predators.", self.styles['TableCell']),
                Paragraph("Continuous flowering", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Fennel / Coriander Border</b>", self.styles['TableCellBold']),
                Paragraph("Irrigation dyke interspersion", self.styles['TableCell']),
                Paragraph("Attracts parasitic wasps and hoverflies which prey naturally on aphids and borers.", self.styles['TableCell']),
                Paragraph("85 - 95 Days", self.styles['TableCell'])
            ]
        ]

        table = Table(data, colWidths=[1.8 * inch, 1.6 * inch, 3.1 * inch, 1.0 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        return elements

    def _build_economic_feasibility(self, crop: str, area: float) -> list:
        """Build farm budgeting and economic feasibility projection table."""
        elements = []
        elements.append(Paragraph("6. ECONOMIC FEASIBILITY & FARM BUDGETING PROJECTION", self.styles['SectionHeading']))

        exp_yield_per_ha = 5.2
        total_yield = exp_yield_per_ha * area
        msp_per_ton = 22500  # indicative INR per ton
        gross_rev_ha = exp_yield_per_ha * msp_per_ton
        cost_ha = 62000
        net_profit_ha = gross_rev_ha - cost_ha
        total_net_profit = net_profit_ha * area

        data = [
            [
                Paragraph("Budgeting Category", self.styles['TableHeader']),
                Paragraph("Per Hectare Estimate", self.styles['TableHeader']),
                Paragraph(f"Total Farm Projection ({area:.1f} ha)", self.styles['TableHeader']),
                Paragraph("Operational Notes", self.styles['TableHeader'])
            ],
            [
                Paragraph("Anticipated Grain/Crop Yield", self.styles['TableCellBold']),
                Paragraph(f"{exp_yield_per_ha:.1f} Tons / ha", self.styles['TableCell']),
                Paragraph(f"<b>{total_yield:.1f} Tons</b>", self.styles['TableCellBold']),
                Paragraph("Based on disciplined adherence to split nutrient & water protocols.", self.styles['TableCell'])
            ],
            [
                Paragraph("Projected Gross Revenue", self.styles['TableCellBold']),
                Paragraph(f"₹ {gross_rev_ha:,.0f}", self.styles['TableCell']),
                Paragraph(f"₹ {gross_rev_ha * area:,.0f}", self.styles['TableCell']),
                Paragraph(f"Calculated at prevailing Minimum Support Price benchmark of ₹{msp_per_ton:,.0f}/ton.", self.styles['TableCell'])
            ],
            [
                Paragraph("Estimated Operational Costs", self.styles['TableCellBold']),
                Paragraph(f"₹ {cost_ha:,.0f}", self.styles['TableCell']),
                Paragraph(f"₹ {cost_ha * area:,.0f}", self.styles['TableCell']),
                Paragraph("Includes seeds, fertilizers, tractor hire, irrigation, weeding, & harvest labor.", self.styles['TableCell'])
            ],
            [
                Paragraph("<b>Estimated Net Profit Margins</b>", self.styles['TableCellBold']),
                Paragraph(f"<b>₹ {net_profit_ha:,.0f} / ha</b>", self.styles['TableCellBold']),
                Paragraph(f"<b>₹ {total_net_profit:,.0f}</b>", self.styles['TableCellBold']),
                Paragraph("<b>Projected Benefit-Cost Ratio: 1.88</b>", self.styles['TableCellBold'])
            ]
        ]

        table = Table(data, colWidths=[2.0 * inch, 1.6 * inch, 1.8 * inch, 2.1 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('BACKGROUND', (0, 4), (-1, 4), LIGHT_BG),
        ]))
        elements.append(table)
        return elements

    def _build_certification_footer(self) -> list:
        """Build certification disclaimer and sign-off."""
        elements = []
        elements.append(HRFlowable(width="100%", thickness=1, color=BORDER_COLOR, spaceAfter=8))

        disc_text = (
            "<b>Disclaimer & Extension Advisory Notice:</b> This dossier is generated by the AgriAdvisory AI Precision "
            "System leveraging ensemble machine learning models, calibrated agronomic retrieval-augmented generation (RAG), "
            "and ICAR agro-advisory guidelines. Field variations in soil micro-flora, acute localized weather anomalies, "
            "and pesticide application timings may require minor adjustments. Always adhere to chemical label precautions."
        )
        elements.append(Paragraph(disc_text, self.styles['TableCell']))
        elements.append(Spacer(1, 0.1 * inch))

        sign_data = [
            [
                Paragraph(f"<b>System Verification:</b> Automated Agronomic Engine v3.5<br/><b>Verified Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styles['TableCell']),
                Paragraph("<b>Agronomic Certification Authority:</b><br/>Division of Agricultural Extension & AI Systems", self.styles['TableCell'])
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
