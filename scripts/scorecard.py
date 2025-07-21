import pandas as pd
import os
import re
import numpy as np
from collections import defaultdict
import logging
import ast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_scorecard():
    # Load processed data
    data_file = "data/processed/ner_results.csv"
    if not os.path.exists(data_file):
        logger.error("⚠️ No processed data found. Run process_scraped_data.py first.")
        return
    
    logger.info(f"📊 Loading data from {data_file}")
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    if df.empty:
        logger.warning("⚠️ No valid data loaded")
        return
    
    # Convert entities string to list of tuples
    try:
        df['entities'] = df['entities'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    except:
        logger.warning("⚠️ Error parsing entities. Creating empty list.")
        df['entities'] = [[] for _ in range(len(df))]
    
    # Initialize vendor metrics
    vendor_metrics = defaultdict(lambda: {
        'total_posts': 0,
        'total_views': 0,
        'prices': [],
        'max_views': 0,
        'top_product': '',
        'top_post_text': ''
    })
    
    # Process each row
    for _, row in df.iterrows():
        vendor = row.get('vendor', 'Unknown')
        views = row.get('views', 0)
        vendor_metrics[vendor]['total_posts'] += 1
        vendor_metrics[vendor]['total_views'] += views
        
        # Track post with highest views
        if views > vendor_metrics[vendor]['max_views']:
            vendor_metrics[vendor]['max_views'] = views
            vendor_metrics[vendor]['top_post_text'] = row.get('full_text', '')
            
            # Find the most significant product in this top post
            products = [ent for ent, label in row['entities'] 
                       if label == "PRODUCT" and len(ent) > 3]
            if products:
                # Select the longest product description
                vendor_metrics[vendor]['top_product'] = max(products, key=len)
        
        # Process entities for prices
        for ent, label in row['entities']:
            if label == 'PRICE':
                try:
                    # Handle both string and numeric prices
                    price = float(ent) if isinstance(ent, str) else ent
                    if price > 0:  # Filter invalid prices
                        vendor_metrics[vendor]['prices'].append(price)
                except:
                    continue
    
    # Calculate metrics
    results = []
    for vendor, metrics in vendor_metrics.items():
        # Basic metrics
        avg_views = metrics['total_views'] / metrics['total_posts'] if metrics['total_posts'] > 0 else 0
        posts_per_week = metrics['total_posts'] / 4  # Assuming 4 weeks of data
        
        # Price metrics with filtering
        valid_prices = [p for p in metrics['prices'] if 10 < p < 100000]
        avg_price = np.mean(valid_prices) if valid_prices else 0
        
        # FIXED: Simplified lending_score calculation
        # Normalize price component
        normalized_price = min(avg_price, 10000) / 1000
        lending_score = (
            0.4 * avg_views + 
            0.3 * posts_per_week + 
            0.3 * normalized_price
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
    
    logger.info(f"✅ Vendor scorecard saved to {output_path}")
    
    # Print top vendors
    print("\nTop Vendors for Loans:")
    print(score_df.head().to_string(index=False))

if __name__ == "__main__":
    generate_scorecard()