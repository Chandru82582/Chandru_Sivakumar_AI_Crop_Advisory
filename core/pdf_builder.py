"""PDF Report Builder for Agricultural Advisory.

Generates professional PDF reports with advisory, recommendations, and timetables.
"""
from io import BytesIO
from datetime import datetime
from typing import Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black, grey
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, KeepTogether
)
try:
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
except ImportError:
    from reportlab.enum.text import TA_CENTER, TA_LEFT, TA_JUSTIFY

from reportlab.pdfgen import canvas


class PDFReportBuilder:
    """Builds comprehensive PDF reports for crop advisory."""

    def __init__(self, page_size=letter):
        """Initialize PDF builder.

        Args:
            page_size: Page size (letter or A4)
        """
        self.page_size = page_size
        self.width, self.height = page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#3368A0'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#66A3BF'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))

    def generate_advisory_pdf(
        self,
        farmer_name: Optional[str] = None,
        location: Optional[str] = None,
        advisory_data: Optional[dict] = None,
        ml_metrics: Optional[dict] = None
    ) -> BytesIO:
        """Generate comprehensive advisory PDF.

        Args:
            farmer_name: Name of the farmer
            location: Farm location
            advisory_data: Dictionary with advisory content
            ml_metrics: Machine learning predictions and metrics

        Returns:
            BytesIO object containing PDF
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.page_size,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            title="Crop Advisory Report"
        )

        elements = []

        elements.extend(self._build_cover_page(farmer_name, location))
        elements.append(PageBreak())

        if advisory_data:
            elements.extend(self._build_advisory_section(advisory_data))
            elements.append(Spacer(1, 0.3*inch))

        if ml_metrics:
            elements.extend(self._build_metrics_section(ml_metrics))
            elements.append(Spacer(1, 0.3*inch))

        if advisory_data and advisory_data.get("timetable"):
            elements.extend(self._build_timetable_section(advisory_data["timetable"]))

        elements.append(Spacer(1, 0.3*inch))
        elements.extend(self._build_footer())

        doc.build(elements)
        buffer.seek(0)
        return buffer

    def _build_cover_page(
        self,
        farmer_name: Optional[str],
        location: Optional[str]
    ) -> list:
        """Build cover page section."""
        elements = []

        title = Paragraph(
            "CROP ADVISORY REPORT",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))

        date_str = datetime.now().strftime("%B %d, %Y")
        date_para = Paragraph(
            f"<b>Generated:</b> {date_str}",
            self.styles['CustomBody']
        )
        elements.append(date_para)

        if farmer_name or location:
            elements.append(Spacer(1, 0.5*inch))
            info_data = []

            if farmer_name:
                info_data.append(["Farmer Name:", farmer_name])
            if location:
                info_data.append(["Farm Location:", location])

            if info_data:
                info_table = Table(info_data, colWidths=[2*inch, 3.5*inch])
                info_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#3368A0')),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ]))
                elements.append(info_table)

        elements.append(Spacer(1, 0.5*inch))
        intro = Paragraph(
            "This comprehensive agricultural advisory has been generated based on "
            "soil analysis, weather patterns, historical crop performance, and "
            "machine learning predictions. Follow the recommendations for optimal "
            "crop yield and farm profitability.",
            self.styles['CustomBody']
        )
        elements.append(intro)

        return elements

    def _build_advisory_section(self, advisory_data: dict) -> list:
        """Build main advisory section."""
        elements = []

        heading = Paragraph("CROP ADVISORY & RECOMMENDATIONS", self.styles['CustomHeading'])
        elements.append(heading)

        if advisory_data.get("crop"):
            crop_line = Paragraph(
                f"<b>Recommended Crop:</b> {advisory_data['crop']} "
                f"<b>(Confidence: {advisory_data.get('confidence', 0):.1%})</b>",
                self.styles['CustomBody']
            )
            elements.append(crop_line)
            elements.append(Spacer(1, 0.1*inch))

        if advisory_data.get("advisory"):
            advisory_para = Paragraph(
                advisory_data['advisory'],
                self.styles['CustomBody']
            )
            elements.append(advisory_para)

        if advisory_data.get("alternative_crops"):
            elements.append(Spacer(1, 0.2*inch))
            alt_crops = ", ".join(advisory_data['alternative_crops'])
            alt_para = Paragraph(
                f"<b>Alternative Crops:</b> {alt_crops}",
                self.styles['CustomBody']
            )
            elements.append(alt_para)

        return elements

    def _build_metrics_section(self, ml_metrics: dict) -> list:
        """Build ML metrics section."""
        elements = []

        heading = Paragraph("PREDICTION METRICS", self.styles['CustomHeading'])
        elements.append(heading)

        if ml_metrics.get("predictions"):
            data = [["Crop", "Probability", "Confidence %"]]
            for pred in ml_metrics["predictions"][:3]:
                data.append([
                    pred.get("crop", "Unknown"),
                    f"{pred.get('probability', 0):.3f}",
                    f"{pred.get('confidence_pct', 0):.1f}%"
                ])

            table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#66A3BF')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#F2EFE7')])
            ]))
            elements.append(table)

        return elements

    def _build_timetable_section(self, timetable: dict) -> list:
        """Build timetable/calendar section."""
        elements = []

        heading = Paragraph("CULTIVATION TIMETABLE", self.styles['CustomHeading'])
        elements.append(heading)

        if timetable.get("stages"):
            for stage in timetable["stages"]:
                stage_data = []

                stage_header = Paragraph(
                    f"<b>{stage.get('stage', 'Stage')}</b> - {stage.get('duration', '')}",
                    self.styles['CustomBody']
                )
                stage_data.append([stage_header])

                if stage.get("activities"):
                    activities_text = "<br/>".join([
                        f"• {activity}"
                        for activity in stage["activities"]
                    ])
                    activities = Paragraph(activities_text, self.styles['CustomBody'])
                    stage_data.append([activities])

                stage_table = Table(stage_data, colWidths=[5.5*inch])
                stage_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#C8DFDB')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('BORDER', (0, 0), (-1, -1), 1, grey)
                ]))
                elements.append(stage_table)
                elements.append(Spacer(1, 0.15*inch))

        return elements

    def _build_footer(self) -> list:
        """Build footer section."""
        elements = []

        footer_text = (
            "This report is generated using AI-powered agricultural analytics. "
            "For best results, follow the recommendations and monitor crop health regularly. "
            "Adjust strategies based on actual field conditions and weather patterns."
        )

        footer_para = Paragraph(footer_text, self.styles['CustomBody'])
        elements.append(footer_para)

        elements.append(Spacer(1, 0.2*inch))

        timestamp = Paragraph(
            f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            self.styles['Normal']
        )
        elements.append(timestamp)

        return elements

    def generate_summary_pdf(
        self,
        summary_data: dict
    ) -> BytesIO:
        """Generate a quick summary PDF.

        Args:
            summary_data: Dictionary with summary information

        Returns:
            BytesIO object containing PDF
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.page_size,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )

        elements = []

        title = Paragraph("FARM SUMMARY REPORT", self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))

        if summary_data.get("statistics"):
            stats = summary_data["statistics"]
            stats_data = [
                ["Metric", "Value"],
                ["Total Farm Records", str(stats.get("total_records", 0))],
                ["Crops Grown", str(stats.get("unique_crops", 0))],
                ["Average Yield", f"{stats.get('avg_yield', 0):.2f} tons/ha"],
                ["Average Profit", f"₹{stats.get('avg_profit', 0):.0f}"],
                ["Avg Nitrogen Usage", f"{stats.get('avg_nitrogen', 0):.2f} kg/ha"]
            ]

            stats_table = Table(stats_data, colWidths=[3*inch, 2.5*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3368A0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#F2EFE7')])
            ]))
            elements.append(stats_table)

        doc.build(elements)
        buffer.seek(0)
        return buffer
