#!/usr/bin/env python
"""
Convert markdown files to PDF using markdown2 and reportlab.
Falls back to HTML if PDF generation fails.
"""

import sys
import os
from pathlib import Path

# Try to import required libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("reportlab not available, will create HTML files instead")

def markdown_to_html(md_file):
    """Convert markdown to HTML with nice styling."""
    with open(md_file, 'r') as f:
        content = f.read()

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{Path(md_file).stem}</title>
    <style>
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.4;
            max-width: 900px;
            margin: 15px auto;
            padding: 15px;
            background: #ffffff;
            color: #333;
            font-size: 11pt;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 6px;
            margin-top: 15px;
            margin-bottom: 10px;
            font-size: 18pt;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 1px solid #95a5a6;
            padding-bottom: 4px;
            margin-top: 12px;
            margin-bottom: 8px;
            font-size: 14pt;
        }}
        h3 {{
            color: #555;
            margin-top: 10px;
            margin-bottom: 6px;
            font-size: 12pt;
        }}
        p {{
            margin: 6px 0;
        }}
        code {{
            background: #f4f4f4;
            padding: 1px 4px;
            border-radius: 2px;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
        }}
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 8px;
            border-radius: 3px;
            overflow-x: auto;
            line-height: 1.3;
            margin: 8px 0;
            font-size: 9pt;
        }}
        pre code {{
            background: transparent;
            color: #ecf0f1;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 6px 8px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        blockquote {{
            border-left: 3px solid #3498db;
            margin: 8px 0;
            padding-left: 12px;
            color: #555;
            font-style: italic;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ecf0f1;
            margin: 15px 0;
        }}
        .emoji {{
            font-size: 1.1em;
        }}
        ul, ol {{
            margin: 6px 0;
            padding-left: 25px;
        }}
        li {{
            margin: 3px 0;
        }}
        @media print {{
            body {{
                margin: 10mm;
                padding: 0;
                max-width: 100%;
                font-size: 10pt;
            }}
            h1 {{
                page-break-before: auto;
                margin-top: 10px;
                font-size: 16pt;
            }}
            h1:first-child {{
                page-break-before: avoid;
            }}
            h2 {{
                font-size: 13pt;
                margin-top: 10px;
            }}
            h3 {{
                font-size: 11pt;
            }}
            pre {{
                page-break-inside: avoid;
                font-size: 8pt;
            }}
            table {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
"""

    # Simple markdown to HTML conversion
    lines = content.split('\n')
    in_code_block = False
    in_table = False

    for line in lines:
        # Code blocks
        if line.startswith('```'):
            if in_code_block:
                html += '</code></pre>\n'
                in_code_block = False
            else:
                html += '<pre><code>'
                in_code_block = True
            continue

        if in_code_block:
            html += line + '\n'
            continue

        # Headers
        if line.startswith('# '):
            html += f'<h1>{line[2:]}</h1>\n'
        elif line.startswith('## '):
            html += f'<h2>{line[3:]}</h2>\n'
        elif line.startswith('### '):
            html += f'<h3>{line[4:]}</h3>\n'
        # Horizontal rule
        elif line.strip() == '---':
            html += '<hr>\n'
        # Table detection (simple)
        elif '|' in line and line.strip().startswith('|'):
            if not in_table:
                html += '<table>\n'
                in_table = True
            if line.strip().replace('|', '').replace('-', '').strip() == '':
                continue  # Skip separator line
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if 'Configuration' in line or 'Aspect' in line:
                html += '<tr>' + ''.join([f'<th>{cell}</th>' for cell in cells]) + '</tr>\n'
            else:
                html += '<tr>' + ''.join([f'<td>{cell}</td>' for cell in cells]) + '</tr>\n'
        else:
            if in_table and '|' not in line:
                html += '</table>\n'
                in_table = False

            # Bold
            line = line.replace('**', '<strong>').replace('**', '</strong>')
            # Inline code
            import re
            line = re.sub(r'`([^`]+)`', r'<code>\1</code>', line)

            # Paragraphs
            if line.strip():
                html += f'<p>{line}</p>\n'
            else:
                html += ''  # Removed <br> for blank lines

    if in_table:
        html += '</table>\n'

    html += """
</body>
</html>
"""

    return html

def convert_to_pdf_html(md_file, output_dir):
    """Convert markdown to HTML (printable to PDF)."""
    html_content = markdown_to_html(md_file)

    output_file = Path(output_dir) / (Path(md_file).stem + '.html')
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"✓ Created HTML: {output_file}")
    print(f"  → To convert to PDF, open in browser and use 'Print to PDF'")
    return output_file

def main():
    # Files to convert
    docs_dir = Path(__file__).parent.parent / 'docs'
    md_files = [
        docs_dir / 'TRM_Model_Architecture_Explained.md',
        docs_dir / 'TRM_Training_Pipeline_Explained.md',
    ]

    output_dir = docs_dir
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("Converting Markdown to Compact Printable Format")
    print("="*70 + "\n")

    for md_file in md_files:
        if not md_file.exists():
            print(f"✗ File not found: {md_file}")
            continue

        print(f"\nProcessing: {md_file.name}")

        # Create HTML version (always)
        html_file = convert_to_pdf_html(md_file, output_dir)

    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print("\nTo create PDFs:")
    print("1. Open the HTML files in a web browser")
    print("2. Use 'Print' (Ctrl+P) and select 'Save as PDF'")
    print("3. Recommended settings:")
    print("   - Margins: Minimum or Custom (10mm)")
    print("   - Scale: 100%")
    print("   - Background graphics: ON")
    print("\nFiles created in:", output_dir)
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
