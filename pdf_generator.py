"""
Forensic PDF Report Generator
Generates signed PDF reports with hash verification for vehicle analysis pipelines.
"""

import hashlib
import json
import base64
import os
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, PageBreak, HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Line, Rect
from reportlab.graphics.charts.piecharts import Pie
from PIL import Image as PILImage


class ForensicPDFGenerator:
    """Generates forensic-grade PDF reports for vehicle analysis."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for forensic report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ForensicTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a1a2e'),
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ForensicSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#4a4a6a'),
            fontName='Helvetica'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#16213e'),
            fontName='Helvetica-Bold',
            borderPadding=(5, 5, 5, 5),
            backColor=colors.HexColor('#e8e8f0')
        ))
        
        # Step header style
        self.styles.add(ParagraphStyle(
            name='StepHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#0f3460'),
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='ForensicBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#333333'),
            fontName='Helvetica'
        ))
        
        # Code/JSON style
        self.styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=self.styles['Normal'],
            fontSize=8,
            fontName='Courier',
            textColor=colors.HexColor('#2d2d2d'),
            backColor=colors.HexColor('#f5f5f5'),
            borderPadding=10,
            spaceAfter=10,
            leftIndent=10,
            rightIndent=10
        ))
        
        # Hash style
        self.styles.add(ParagraphStyle(
            name='HashStyle',
            parent=self.styles['Normal'],
            fontSize=8,
            fontName='Courier',
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='FooterStyle',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#888888'),
            alignment=TA_CENTER
        ))
        
        # Metadata style
        self.styles.add(ParagraphStyle(
            name='MetadataStyle',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#555555'),
            fontName='Helvetica'
        ))

    def _calculate_hash(self, data: Dict[str, Any], image_path: Optional[str] = None) -> str:
        """Calculate SHA-256 hash of the report data and image."""
        hash_content = json.dumps(data, sort_keys=True, default=str)
        
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                hash_content += base64.b64encode(f.read()).decode()
        
        return hashlib.sha256(hash_content.encode()).hexdigest()
    
    def _format_json_for_display(self, data: Any, indent: int = 0) -> str:
        """Format JSON data for readable display in PDF."""
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                formatted_value = self._format_json_for_display(value, indent + 1)
                lines.append(f"{'  ' * indent}<b>{key}:</b> {formatted_value}")
            return '<br/>'.join(lines) if indent == 0 else '<br/>' + '<br/>'.join(lines)
        elif isinstance(data, list):
            if len(data) == 0:
                return "[]"
            items = [self._format_json_for_display(item, indent + 1) for item in data]
            return '<br/>' + '<br/>'.join(f"{'  ' * indent}- {item}" for item in items)
        else:
            return str(data)

    def _create_header(self, report_id: str, timestamp: str) -> List:
        """Create the report header section."""
        elements = []
        
        # Main title
        elements.append(Paragraph(
            "FORENSIC ANALYSIS REPORT",
            self.styles['ForensicTitle']
        ))
        
        # Subtitle
        elements.append(Paragraph(
            "Vehicle & Occupant Analysis - Vision LLM Pipeline",
            self.styles['ForensicSubtitle']
        ))
        
        elements.append(Spacer(1, 10))
        
        # Horizontal line
        elements.append(HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor('#1a1a2e'),
            spaceBefore=5,
            spaceAfter=15
        ))
        
        # Report metadata table
        metadata = [
            ['Report ID:', report_id],
            ['Generated:', timestamp],
            ['Classification:', 'CONFIDENTIAL - FORENSIC EVIDENCE'],
            ['Version:', '1.0']
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#333333')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        
        elements.append(metadata_table)
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_image_section(self, image_path: str, image_hash: str) -> List:
        """Create the analyzed image section."""
        elements = []
        
        elements.append(Paragraph(
            "1. ANALYZED IMAGE",
            self.styles['SectionHeader']
        ))
        
        if os.path.exists(image_path):
            # Get image dimensions and resize for PDF
            with PILImage.open(image_path) as img:
                orig_width, orig_height = img.size
                
            # Calculate aspect ratio and fit to page
            max_width = 5.5 * inch
            max_height = 4 * inch
            
            aspect = orig_width / orig_height
            if aspect > max_width / max_height:
                display_width = max_width
                display_height = max_width / aspect
            else:
                display_height = max_height
                display_width = max_height * aspect
            
            # Create image with border
            img_element = Image(image_path, width=display_width, height=display_height)
            
            # Image info table
            img_info = [
                ['Original Resolution:', f'{orig_width} x {orig_height} pixels'],
                ['Image Hash (SHA-256):', image_hash[:32] + '...'],
                ['File:', os.path.basename(image_path)]
            ]
            
            img_table = Table(img_info, colWidths=[2*inch, 4*inch])
            img_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Courier'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#555555')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            
            # Center the image
            image_container = Table([[img_element]], colWidths=[6*inch])
            image_container.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#1a1a2e')),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fafafa')),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ]))
            
            elements.append(image_container)
            elements.append(Spacer(1, 10))
            elements.append(img_table)
        else:
            elements.append(Paragraph(
                f"<i>Image not found: {image_path}</i>",
                self.styles['ForensicBody']
            ))
        
        elements.append(Spacer(1, 15))
        return elements

    def _create_context_section(self, context: Dict[str, Any]) -> List:
        """Create the context/metadata section."""
        elements = []
        
        elements.append(Paragraph(
            "2. INPUT CONTEXT DATA",
            self.styles['SectionHeader']
        ))
        
        if context:
            # Flatten context for display
            context_data = []
            
            def flatten_dict(d, prefix=''):
                for key, value in d.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        flatten_dict(value, full_key)
                    else:
                        context_data.append([full_key, str(value)])
            
            flatten_dict(context)
            
            if context_data:
                context_table = Table(
                    [['Parameter', 'Value']] + context_data,
                    colWidths=[2.5*inch, 3.5*inch]
                )
                context_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 1), (1, -1), 'Courier'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8e8f0')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#333333')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(context_table)
        else:
            elements.append(Paragraph(
                "<i>No context data provided</i>",
                self.styles['ForensicBody']
            ))
        
        elements.append(Spacer(1, 15))
        return elements

    def _create_step_section(self, step_num: int, step_data: Dict[str, Any]) -> List:
        """Create a section for a single pipeline step."""
        elements = []
        
        step_name = step_data.get('step_name', f'Step {step_num}')
        
        elements.append(Paragraph(
            f"3.{step_num} {step_name.upper()}",
            self.styles['StepHeader']
        ))
        
        # Step metadata
        step_meta = []
        if 'llm_used' in step_data:
            step_meta.append(['LLM Model:', step_data['llm_used']])
        if 'latency_ms' in step_data:
            step_meta.append(['Processing Time:', f"{step_data['latency_ms']:.0f} ms"])
        if 'input_tokens' in step_data:
            step_meta.append(['Input Tokens:', str(step_data['input_tokens'])])
        if 'output_tokens' in step_data:
            step_meta.append(['Output Tokens:', str(step_data['output_tokens'])])
        
        if step_meta:
            meta_table = Table(step_meta, colWidths=[1.5*inch, 2*inch])
            meta_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#666666')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
            elements.append(meta_table)
            elements.append(Spacer(1, 8))
        
        # Prompt used (if available)
        if 'prompt' in step_data:
            elements.append(Paragraph(
                "<b>Prompt Used:</b>",
                self.styles['ForensicBody']
            ))
            prompt_text = step_data['prompt'][:500] + '...' if len(step_data.get('prompt', '')) > 500 else step_data.get('prompt', '')
            elements.append(Paragraph(
                f"<font face='Courier' size='8'>{prompt_text}</font>",
                self.styles['ForensicBody']
            ))
            elements.append(Spacer(1, 8))
        
        # Response/Output
        elements.append(Paragraph(
            "<b>Analysis Output:</b>",
            self.styles['ForensicBody']
        ))
        
        response = step_data.get('response', step_data.get('output', 'No output'))
        
        # Try to parse as JSON for better formatting
        try:
            if isinstance(response, str):
                response_data = json.loads(response)
            else:
                response_data = response
            
            # Create a formatted table for JSON data
            formatted_rows = self._json_to_table_rows(response_data)
            if formatted_rows:
                response_table = Table(
                    [['Field', 'Value']] + formatted_rows,
                    colWidths=[2.5*inch, 3.5*inch]
                )
                response_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),
                    ('FONTNAME', (1, 1), (1, -1), 'Courier'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f5')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#333333')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                ]))
                elements.append(response_table)
            else:
                elements.append(Paragraph(
                    json.dumps(response_data, indent=2, ensure_ascii=False)[:1000],
                    self.styles['CodeBlock']
                ))
        except (json.JSONDecodeError, TypeError):
            # Display as plain text
            elements.append(Paragraph(
                str(response)[:1000],
                self.styles['CodeBlock']
            ))
        
        elements.append(Spacer(1, 15))
        return elements

    def _json_to_table_rows(self, data: Any, prefix: str = '') -> List[List[str]]:
        """Convert JSON data to table rows."""
        rows = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    rows.extend(self._json_to_table_rows(value, full_key))
                elif isinstance(value, list):
                    if len(value) > 0 and isinstance(value[0], dict):
                        for i, item in enumerate(value):
                            rows.extend(self._json_to_table_rows(item, f"{full_key}[{i}]"))
                    else:
                        rows.append([full_key, str(value)])
                else:
                    rows.append([full_key, str(value)])
        
        return rows

    def _create_signature_section(self, report_hash: str, timestamp: str) -> List:
        """Create the digital signature section."""
        elements = []
        
        elements.append(Spacer(1, 20))
        elements.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor('#cccccc'),
            spaceBefore=10,
            spaceAfter=10
        ))
        
        elements.append(Paragraph(
            "DIGITAL SIGNATURE & INTEGRITY VERIFICATION",
            self.styles['SectionHeader']
        ))
        
        signature_info = [
            ['Report Hash (SHA-256):', report_hash],
            ['Hash Algorithm:', 'SHA-256'],
            ['Timestamp:', timestamp],
            ['Generator:', 'Vision LLM Comparator - Forensic Report Module'],
            ['Integrity Status:', 'VERIFIED - ORIGINAL DOCUMENT']
        ]
        
        sig_table = Table(signature_info, colWidths=[2.5*inch, 4*inch])
        sig_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -2), 'Courier'),
            ('FONTNAME', (1, -1), (1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -2), colors.HexColor('#333333')),
            ('TEXTCOLOR', (1, -1), (1, -1), colors.HexColor('#228B22')),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#f0fff0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1a1a2e')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ]))
        
        elements.append(sig_table)
        
        elements.append(Spacer(1, 15))
        elements.append(Paragraph(
            "<i>This document has been digitally signed using SHA-256 cryptographic hash. "
            "Any modification to the content will invalidate the signature. "
            "To verify integrity, recalculate the hash and compare with the value above.</i>",
            self.styles['FooterStyle']
        ))
        
        return elements

    def _create_footer(self, canvas, doc):
        """Add footer to each page."""
        canvas.saveState()
        
        # Footer line
        canvas.setStrokeColor(colors.HexColor('#cccccc'))
        canvas.setLineWidth(0.5)
        canvas.line(inch, 0.75*inch, doc.pagesize[0] - inch, 0.75*inch)
        
        # Footer text
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#888888'))
        
        footer_text = f"Vision LLM Comparator - Forensic Report | Page {doc.page}"
        canvas.drawCentredString(doc.pagesize[0]/2, 0.5*inch, footer_text)
        
        # Confidential watermark
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawRightString(doc.pagesize[0] - inch, 0.5*inch, "CONFIDENTIAL")
        
        canvas.restoreState()

    def generate_report(
        self,
        execution_id: str,
        image_path: str,
        context: Dict[str, Any],
        steps: List[Dict[str, Any]],
        output_path: str,
        pipeline_name: str = "Vehicle Analysis Pipeline"
    ) -> Dict[str, Any]:
        """
        Generate a complete forensic PDF report.
        
        Args:
            execution_id: Unique identifier for this execution
            image_path: Path to the analyzed image
            context: Input context/metadata
            steps: List of step results from the pipeline
            output_path: Where to save the PDF
            pipeline_name: Name of the pipeline used
            
        Returns:
            Dict with report_path, report_hash, and metadata
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Calculate image hash
        image_hash = ""
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                image_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Build the document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )
        
        elements = []
        
        # Header
        elements.extend(self._create_header(execution_id, timestamp))
        
        # Image section
        elements.extend(self._create_image_section(image_path, image_hash))
        
        # Context section
        elements.extend(self._create_context_section(context))
        
        # Pipeline results header
        elements.append(Paragraph(
            f"3. PIPELINE ANALYSIS RESULTS: {pipeline_name.upper()}",
            self.styles['SectionHeader']
        ))
        
        # Each step
        for i, step in enumerate(steps, 1):
            elements.extend(self._create_step_section(i, step))
        
        # Calculate report hash (before signature section)
        report_data = {
            'execution_id': execution_id,
            'timestamp': timestamp,
            'image_hash': image_hash,
            'context': context,
            'steps': steps,
            'pipeline_name': pipeline_name
        }
        report_hash = self._calculate_hash(report_data, image_path)
        
        # Signature section
        elements.extend(self._create_signature_section(report_hash, timestamp))
        
        # Build PDF
        doc.build(elements, onFirstPage=self._create_footer, onLaterPages=self._create_footer)
        
        return {
            'report_path': output_path,
            'report_hash': report_hash,
            'image_hash': image_hash,
            'timestamp': timestamp,
            'execution_id': execution_id,
            'page_count': doc.page
        }


def generate_forensic_report(
    execution_id: str,
    image_path: str,
    context: Dict[str, Any],
    steps: List[Dict[str, Any]],
    output_dir: str = "/tmp",
    pipeline_name: str = "Vehicle Analysis Pipeline"
) -> Dict[str, Any]:
    """
    Convenience function to generate a forensic report.
    
    Args:
        execution_id: Unique identifier for this execution
        image_path: Path to the analyzed image
        context: Input context/metadata
        steps: List of step results from the pipeline
        output_dir: Directory to save the PDF
        pipeline_name: Name of the pipeline used
        
    Returns:
        Dict with report details including path and hash
    """
    generator = ForensicPDFGenerator()
    
    # Create output filename
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"forensic_report_{execution_id}_{timestamp_str}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    return generator.generate_report(
        execution_id=execution_id,
        image_path=image_path,
        context=context,
        steps=steps,
        output_path=output_path,
        pipeline_name=pipeline_name
    )
