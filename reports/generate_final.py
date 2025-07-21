# reports/generate_final.py
from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import os
import textwrap

# Create PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=16)
pdf.cell(200, 10, txt="Amharic E-commerce Data Extractor", ln=True, align='C')
pdf.set_font("Arial", size=14)
pdf.cell(200, 10, txt="Final Project Report", ln=True, align='C')
pdf.ln(10)

# Introduction
pdf.set_font("Arial", size=12)
intro = textwrap.fill(
    "This project developed a Named Entity Recognition system for Amharic Telegram commerce data. "
    "Our solution extracts products, prices, and locations to power EthioMart's vendor scoring system "
    "for micro-lending decisions.", 100
)
pdf.multi_cell(0, 10, txt=intro)
pdf.ln(10)

# Model Performance
pdf.add_page()
pdf.set_font("Arial", size=14)
pdf.cell(200, 10, txt="Model Comparison Results", ln=True)
pdf.set_font("Arial", size=10)

# Add table with fixed column names
try:
    results = pd.read_csv("results/model_comparison.csv")
    # Rename columns if needed
    results = results.rename(columns={
        'Unnamed: 0': 'Model',
        'f1': 'F1-Score',
        'precision': 'Precision',
        'recall': 'Recall'
    })
    
    col_width = 40
    pdf.cell(col_width, 10, "Model", border=1)
    pdf.cell(col_width, 10, "F1-Score", border=1)
    pdf.cell(col_width, 10, "Precision", border=1)
    pdf.cell(col_width, 10, "Recall", border=1)
    pdf.ln()

    for i, row in results.iterrows():
        model_name = row.get('Model', results.index[i])
        pdf.cell(col_width, 10, str(model_name), border=1)
        pdf.cell(col_width, 10, f"{row.get('F1-Score', 0):.4f}", border=1)
        pdf.cell(col_width, 10, f"{row.get('Precision', 0):.4f}", border=1)
        pdf.cell(col_width, 10, f"{row.get('Recall', 0):.4f}", border=1)
        pdf.ln()
        
except Exception as e:
    pdf.cell(200, 10, txt=f"Could not load model comparison data: {str(e)}", ln=True)

# Vendor Analysis
try:
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Vendor Scorecard", ln=True)
    pdf.set_font("Arial", size=10)

    scorecard = pd.read_csv("data/processed/vendor_scorecard.csv")
    col_width = 40
    headers = ["Vendor", "Avg Views", "Posts/Week", "Avg Price", "Lending Score"]
    for header in headers:
        pdf.cell(col_width, 10, header, border=1)
    pdf.ln()

    for _, row in scorecard.iterrows():
        pdf.cell(col_width, 10, str(row.get('vendor', 'Unknown')), border=1)
        pdf.cell(col_width, 10, f"{row.get('Avg. Views/Post', 0):.0f}", border=1)
        pdf.cell(col_width, 10, f"{row.get('Posts/Week', 0):.1f}", border=1)
        pdf.cell(col_width, 10, f"{row.get('Avg. Price (ETB)', 0):.2f}", border=1)
        pdf.cell(col_width, 10, f"{row.get('Lending Score', 0):.2f}", border=1)
        pdf.ln()
        
except Exception as e:
    pdf.cell(200, 10, txt=f"Could not load vendor scorecard: {str(e)}", ln=True)

# Business Impact
pdf.add_page()
pdf.set_font("Arial", size=14)
pdf.cell(200, 10, txt="Business Impact Analysis", ln=True)
pdf.set_font("Arial", size=12)

impact = textwrap.fill(
    "Our solution enables EthioMart to identify high-potential vendors for micro-loans: "
    "1. ShegerOnline shows strong engagement (3,112 avg views/post) "
    "2. Active vendors post 48+ times/week "
    "3. Premium products average 51,869 ETB "
    "Recommendation: Prioritize loans for vendors with lending scores > 3.0", 
    100
)
pdf.multi_cell(0, 10, txt=impact)

# Save
pdf.output("reports/final_report.pdf")
print("✅ Generated final_report.pdf")