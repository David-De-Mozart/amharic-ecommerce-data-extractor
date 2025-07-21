import pandas as pd
import spacy
from collections import defaultdict
import os
import re
import numpy as np

def load_model():
    """Load our custom trained spaCy model"""
    try:
        nlp = spacy.load("models/transformer-ner")
        print("✅ NER model loaded successfully")
        return nlp
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def extract_entities(text, nlp):
    """Extract entities from text using our spaCy model"""
    if not text or str(text).strip() == "":
        return []
    
    try:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "PRICE", "LOC"]:
                entities.append((ent.text, ent.label_))
        return entities
    except Exception as e:
        print(f"Error processing text: {e}")
        return []

def normalize_price(price_text):
    """Extract numeric value from price text"""
    try:
        # Extract digits
        price_str = re.sub(r"[^\d]", "", price_text)
        if price_str:
            return float(price_str)
    except:
        pass
    return None

def main():
    # Load the processed data
    data_file = "data/processed/ner_results.csv"
    if not os.path.exists(data_file):
        print(f"❌ Processed data not found: {data_file}")
        print("Run process_scraped_data.py first")
        return
    
    print(f"📊 Loading data from {data_file}")
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    if df.empty:
        print("❌ No data loaded")
        return
    
    # Load NER model
    nlp = load_model()
    if not nlp:
        return
    
    # Initialize vendor metrics
    vendor_metrics = defaultdict(lambda: {
        'total_posts': 0,
        'total_views': 0,
        'products': set(),
        'prices': [],
        'max_views': 0,
        'top_post_text': '',
        'top_product': ''
    })
    
    # Process each row
    for _, row in df.iterrows():
        vendor = row.get('vendor', 'Unknown')
        views = row.get('views', 0)
        text = row.get('full_text', '') or row.get('message', '')
        
        metrics = vendor_metrics[vendor]
        metrics['total_posts'] += 1
        metrics['total_views'] += views
        
        # Track top post
        if views > metrics['max_views']:
            metrics['max_views'] = views
            metrics['top_post_text'] = text
            
            # Extract entities from top post
            entities = extract_entities(text, nlp)
            for ent_text, label in entities:
                if label == "PRODUCT":
                    metrics['top_product'] = ent_text
        
        # Extract all entities for price calculation
        entities = extract_entities(text, nlp)
        for ent_text, label in entities:
            if label == "PRODUCT":
                metrics['products'].add(ent_text)
            elif label == "PRICE":
                price = normalize_price(ent_text)
                if price:
                    metrics['prices'].append(price)
    
    # Calculate metrics
    results = []
    for vendor, metrics in vendor_metrics.items():
        # Basic metrics
        avg_views = metrics['total_views'] / metrics['total_posts'] if metrics['total_posts'] > 0 else 0
        posts_per_week = metrics['total_posts'] / 4  # Assuming 4 weeks of data
        
        # Price metrics with filtering
        valid_prices = [p for p in metrics['prices'] if 10 < p < 1000000]
        avg_price = np.mean(valid_prices) if valid_prices else 0
        
        # Lending Score formula
        lending_score = (
            0.5 * min(avg_views / 1000, 1) +  # Normalize views
            0.3 * min(posts_per_week / 10, 1) +  # Normalize frequency
            0.2 * min(avg_price / 50000, 1)  # Normalize price
        )
        
        results.append({
            "Vendor": vendor,
            "Avg Views/Post": avg_views,
            "Posts/Week": posts_per_week,
            "Avg Price (ETB)": avg_price,
            "Top Product": metrics['top_product'],
            "Lending Score": lending_score
        })
    
    # Create DataFrame and save
    score_df = pd.DataFrame(results).sort_values("Lending Score", ascending=False)
    output_path = "results/vendor_scorecard.csv"
    os.makedirs("results", exist_ok=True)
    score_df.to_csv(output_path, index=False)
    
    print("\n🏆 Vendor Scorecard:")
    print(score_df.to_string(index=False))
    print(f"\n✅ Saved to {output_path}")

if __name__ == "__main__":
    main()