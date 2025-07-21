# Amharic E-commerce Data Extractor
*Transform Telegram commerce data into actionable business insights*

## Project Overview
This solution addresses EthioMart's need to consolidate decentralized Telegram commerce channels by:
1. Extracting key entities (products, prices, locations) from Amharic messages
2. Analyzing vendor performance for micro-lending decisions
3. Creating a centralized vendor evaluation platform

**Key Features**:
- Telegram data ingestion pipeline
- Fine-tuned Amharic NER models
- Automated vendor scorecard
- Model interpretability reports
- Business intelligence reporting

## Business Impact
```mermaid
graph LR
A[Telegram Channels] --> B(Data Ingestion)
B --> C[Entity Extraction]
C --> D[Vendor Scoring]
D --> E[Loan Decisions]
E --> F[Increased Platform Engagement]

## File Structure

├── data/               # All project data
│   ├── labeled/        # Labeled datasets (CoNLL format)
│   ├── processed/      # Processed results and analytics
│   └── raw/            # Raw scraped data
│
├── models/             # Trained ML models
├── reports/            # Generated PDF reports
├── results/            # Model evaluation outputs
│
├── scripts/            # Execution pipelines
│   ├── data_ingestion.py
│   ├── data_labeling.py
│   ├── model_comparison.py
│   ├── model_interpretability.py
│   └── process_scraped_data.py
│
├── .gitignore
├── LICENSE
├── README.md           # This document
└── requirements.txt    # Python dependencies

## Key Metrics
Component	Result
Messages Processed	146
Entities Extracted	587
Best Model F1-Score	0.3571
Top Vendor Score	3.10 (ShegerOnline)


## Installation

# Clone repository
git clone https://github.com/David-De-Mozart/amharic-ecommerce-data-extractor.git
cd amharic-ecommerce-data-extractor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Step-by-Step Execution

1. Data Collection:
    python scripts/data_ingestion.py

2. Data Labeling:
    python scripts/data_labeling.py

3. Model Training:
    python scripts/model_comparison.py

4. Vendor Analysis:
    python scripts/process_scraped_data.py

5. Generate Reports:
    python reports/generate_interim.py
    python reports/generate_final.py


## Results

**Model Performance**

Model	   F1-Score	Precision	Recall
DistilBERT	0.3571	0.3947	0.3261
XLM-Roberta	0.2600	0.0000	0.0000
spaCy CNN	0.2200	0.0000	0.0000

**Top Vendors**

Vendor	       Avg Views	Posts/Week	Avg Price	Lending Score
ShegerOnline	3,112	        48.0	51,869 ETB	3.10
Ethio Shop	     642	        0.2	    97,122 ETB	0.45
Addis	         141	        3.3	    7,025 ETB	0.30


License
This project is licensed under the MIT License - see LICENSE for details.