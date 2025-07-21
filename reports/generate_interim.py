# reports/generate_interim.py (updated)
from fpdf import FPDF
import pandas as pd
import os
from datetime import date
import glob

def count_conll_messages():
    """Count labeled messages from CONLL files"""
    total = 0
    for file in glob.glob("data/labeled/*.conll"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            # Count messages by number of double newlines
            total += content.count("\n\n")
    return total

# Create PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=16)
pdf.cell(200, 10, txt="EthioMart Data Extraction Interim Report", ln=True, align='C')

# Data Summary
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt=f"Report Date: {date.today()}", ln=True)
pdf.cell(200, 10, txt="Data Collection Summary", ln=True)

# Add scraped data stats
scraped = pd.read_csv("data/raw/scraped_messages.csv")
stats = {
    "Total Messages": len(scraped),
    "Vendors": scraped['vendor'].nunique(),
    "Avg Views": scraped['views'].mean(),
    "With Images": scraped[scraped['image_path'] != ''].shape[0]
}

for k, v in stats.items():
    pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)

# Labeling Summary
pdf.cell(200, 10, txt="Labeling Summary", ln=True)
labeled_count = count_conll_messages()
pdf.cell(200, 10, txt=f"Labeled Messages: {labeled_count}", ln=True)

# Preprocessing Steps
pdf.cell(200, 10, txt="Preprocessing Steps:", ln=True)
steps = [
    "1. Telegram message scraping using Telethon",
    "2. Image OCR with Tesseract (Amharic)",
    "3. Text cleaning and normalization",
    "4. Rule-based entity labeling for NER training"
]

for step in steps:
    pdf.cell(200, 10, txt=step, ln=True)

# Save
pdf.output("reports/interim_report.pdf")
print("✅ Generated interim_report.pdf")