import pandas as pd
import spacy
import os
import re
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_entities(text, nlp):
    """Extract entities from text using our trained model with enhanced processing"""
    if not text or str(text).strip() == "":
        return []
    
    try:
        # Preprocess text for better entity recognition
        text = text.replace('\n', ' ').replace('።', '.').replace('፣', ',')
        doc = nlp(str(text))
        
        entities = []
        for ent in doc.ents:
            # Filter out bad predictions and handle Amharic-specific cases
            if len(ent.text) > 1 and ent.label_ in ["PRODUCT", "PRICE", "LOC", "CONTACT"]:
                # Special handling for PRICE entities
                if ent.label_ == "PRICE":
                    normalized = normalize_price(ent.text)
                    if normalized:
                        entities.append((normalized, "PRICE"))
                
                # Special handling for CONTACT entities
                elif ent.label_ == "CONTACT":
                    normalized = normalize_phone(ent.text)
                    if normalized:
                        entities.append((normalized, "CONTACT"))
                
                # For PRODUCT and LOC, use the original text
                else:
                    entities.append((ent.text, ent.label_))
                    
        return entities
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return []

def clean_text(text):
    """Advanced text cleaning for Amharic and mixed-language content"""
    if not text:
        return ""
    
    # Remove excessive whitespace and line breaks
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs and special characters but preserve Amharic
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s.,:;!?፣።፤፥፦አ-ዅ]', '', text)
    
    # Fix common OCR errors in Amharic
    corrections = {
        'ሀ': 'ሀ', 'ሁ': 'ሁ', 'ሂ': 'ሂ', 'ሃ': 'ሃ', 'ሄ': 'ሄ', 'ህ': 'ህ', 'ሆ': 'ሆ',
        'ለ': 'ለ', 'ሉ': 'ሉ', 'ሊ': 'ሊ', 'ላ': 'ላ', 'ሌ': 'ሌ', 'ል': 'ል', 'ሎ': 'ሎ',
        'ጸ': 'ጸ', 'ጹ': 'ጹ', 'ጺ': 'ጺ', 'ጻ': 'ጻ', 'ጼ': 'ጼ', 'ጽ': 'ጽ', 'ጾ': 'ጾ'
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
        
    return text.strip()

def normalize_price(price_text):
    """Extract and normalize price values from text with Amharic numbers"""
    try:
        # Convert Amharic numbers to Western digits
        amh_to_eng = {
            '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
            '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁰': '0',
            '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
            '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10',
            '፳': '20', '፴': '30', '፵': '40', '፶': '50', '፷': '60',
            '፸': '70', '፹': '80', '፺': '90', '፻': '100', '፼': '10000'
        }
        
        # Replace Amharic numbers
        for amh, eng in amh_to_eng.items():
            price_text = price_text.replace(amh, eng)
        
        # Extract all numeric values
        numbers = re.findall(r'[\d,]+\.?\d*', price_text.replace(',', ''))
        if numbers:
            # Return the largest number found
            return max([float(num) for num in numbers if num.replace('.', '').isdigit()])
    except Exception as e:
        logger.warning(f"Price normalization failed: {price_text} - {e}")
    return None

def normalize_phone(phone_text):
    """Extract and format Ethiopian phone numbers"""
    # Ethiopian phone number patterns
    patterns = [
        r'(09\d{8})',  # 0912345678
        r'(\d{3}[- ]?\d{3}[- ]?\d{4})',  # 091-234-5678
        r'(09\d\s?\d{3}\s?\d{4})'  # 09x xxx xxxx
    ]
    
    for pattern in patterns:
        match = re.search(pattern, phone_text)
        if match:
            # Format as 09XX XXX XXX
            number = re.sub(r'\D', '', match.group(1))
            if len(number) == 10:
                return f"{number[:4]} {number[4:7]} {number[7:]}"
    return None

def standardize_vendor_names(vendor_name):
    """Standardize vendor names for consistent grouping"""
    if not vendor_name or pd.isna(vendor_name):
        return "Unknown"
    
    vendor_map = {
        r'sheger\s*online': 'ShegerOnline',
        r'addis\s*bazaar': 'AddisBazaar',
        r'ethio[\s\-_]?electronics': 'EthioElectronics',
        r'ethiopian\s*marketplace': 'EthiopianMarketplace',
        r'shager\s*online': 'ShagerOnline'
    }
    
    vendor_name = str(vendor_name).lower().strip()
    for pattern, standard in vendor_map.items():
        if re.search(pattern, vendor_name):
            return standard
            
    # Capitalize if no match
    return vendor_name.title()

def is_valid_product(text):
    """Filter out invalid product names"""
    if not text or len(text) < 2:
        return False
    if text.isdigit():
        return False
    if re.match(r'^(http|www|:|//)', text):
        return False
    return True

def calculate_vendor_scorecard(df):
    """Calculate vendor metrics and lending scores"""
    vendor_stats = {}
    
    for vendor, group in df.groupby('vendor'):
        # Basic metrics
        post_count = len(group)
        total_views = group['views'].sum()
        avg_views = total_views / post_count if post_count > 0 else 0
        
        # Calculate posting frequency (posts per week)
        try:
            # Convert timestamps to datetime
            group['datetime'] = pd.to_datetime(group['timestamp'])
            time_span = group['datetime'].max() - group['datetime'].min()
            weeks = max(time_span.days / 7, 1)  # At least 1 week
            posts_per_week = post_count / weeks
        except Exception as e:
            logger.warning(f"Time calculation error for {vendor}: {e}")
            posts_per_week = post_count  # Fallback to total posts
        
        # Extract prices and products
        prices = []
        products = set()
        for entities in group['entities']:
            for ent, label in entities:
                if label == "PRICE" and isinstance(ent, (int, float)):
                    prices.append(ent)
                elif label == "PRODUCT":
                    products.add(ent)
        
        # Calculate average price (filter extremes)
        valid_prices = [p for p in prices if 10 < p < 1000000]
        avg_price = np.mean(valid_prices) if valid_prices else 0
        
        # Find top performing post
        top_post = group.loc[group['views'].idxmax()] if not group.empty else None
        top_product = None
        top_price = None
        if top_post is not None:
            for ent, label in top_post['entities']:
                if label == "PRODUCT" and top_product is None:
                    top_product = ent
                elif label == "PRICE" and isinstance(ent, (int, float)) and top_price is None:
                    top_price = ent
        
        # Calculate lending score (customizable weights)
        lending_score = (
            0.5 * (avg_views / 1000) +  # Normalize views
            0.3 * (posts_per_week / 10) +  # Normalize frequency
            0.2 * (len(products) / 50)  # Normalize product count
        )
        
        vendor_stats[vendor] = {
            'Avg. Views/Post': avg_views,
            'Posts/Week': posts_per_week,
            'Top Post Product': top_product,
            'Top Post Price': top_price,
            'Avg. Price (ETB)': avg_price,
            'Total Products': len(products),
            'Lending Score': lending_score
        }
    
    return pd.DataFrame.from_dict(vendor_stats, orient='index')

def main():
    # Load the latest scraped data
    input_file = "data/raw/scraped_messages.csv"
    if not os.path.exists(input_file):
        logger.error("⚠️ No scraped data found. Run data_ingestion.py first.")
        return
    
    logger.info(f"📊 Loading scraped data from {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"⚠️ Error loading CSV: {e}")
        return
    
    if df.empty:
        logger.warning("⚠️ No valid data loaded")
        return
    
    logger.info(f"📊 Total messages loaded: {len(df)}")
    
    # Clean and standardize vendor names
    df['vendor'] = df['vendor'].fillna('Unknown').apply(standardize_vendor_names)
    
    # Clean text data
    df['message'] = df['message'].fillna('').apply(clean_text)
    df['ocr_text'] = df['ocr_text'].fillna('').apply(clean_text)
    
    # Combine text sources
    df['full_text'] = df['message'] + " " + df['ocr_text']
    df['full_text'] = df['full_text'].apply(lambda x: x[:2000])  # Limit to first 2000 chars
    
    # Load NER model
    try:
        nlp = spacy.load("models/amharic-ner")
        logger.info("✅ NER model loaded successfully")
    except Exception as e:
        logger.error(f"⚠️ Error loading NER model: {e}")
        return
    
    # Extract entities
    logger.info("🔍 Extracting entities from messages...")
    df['entities'] = [extract_entities(text, nlp) for text in tqdm(df['full_text'], desc="Processing")]
    
    # Post-process entities
    def post_process_entities(entities):
        results = []
        seen = set()
        
        for ent_text, label in entities:
            # Deduplicate entities
            if ent_text in seen:
                continue
            seen.add(ent_text)
                
            # Filter products
            if label == "PRODUCT" and not is_valid_product(ent_text):
                continue
                
            results.append((ent_text, label))
        return results
    
    df['entities'] = df['entities'].apply(post_process_entities)
    
    # Count entities
    df['entity_count'] = df['entities'].apply(len)
    total_entities = df['entity_count'].sum()
    
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    output_file = "data/processed/ner_results.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"\n💾 Saved processed data to {output_file}")
    logger.info(f"📊 Total entities extracted: {total_entities}")
    
    # Show sample results
    sample = df[df['entity_count'] > 0].head(3)
    if not sample.empty:
        logger.info("\nSample extracted entities:")
        for _, row in sample.iterrows():
            print(f"\n📝 Vendor: {row['vendor']}")
            print(f"👁️ Views: {row['views']}")
            print(f"📄 Message: {row['full_text'][:150]}{'...' if len(row['full_text']) > 150 else ''}")
            
            # Group entities by type for better display
            entity_groups = defaultdict(list)
            for ent, label in row['entities']:
                entity_groups[label].append(str(ent))
                
            for label, items in entity_groups.items():
                print(f"  - {label}: {', '.join(items)}")
    else:
        logger.warning("\n⚠️ No entities extracted. Check your model and data.")
    
    # Calculate and save vendor scorecard
    logger.info("\n📈 Calculating vendor scorecard...")
    scorecard_df = calculate_vendor_scorecard(df)
    scorecard_file = "data/processed/vendor_scorecard.csv"
    scorecard_df.to_csv(scorecard_file)
    
    logger.info(f"💾 Saved vendor scorecard to {scorecard_file}")
    logger.info("\nVendor Scorecard Summary:")
    logger.info(scorecard_df[['Avg. Views/Post', 'Posts/Week', 'Avg. Price (ETB)', 'Lending Score']].to_string())
    
    # Show top vendors by lending score
    top_vendors = scorecard_df.sort_values('Lending Score', ascending=False).head(3)
    if not top_vendors.empty:
        logger.info("\n🏆 Top Performing Vendors:")
        for vendor, row in top_vendors.iterrows():
            logger.info(f"  - {vendor}: Lending Score {row['Lending Score']:.2f}")
            logger.info(f"    Avg Views: {row['Avg. Views/Post']:.0f} | "
                       f"Posts/Week: {row['Posts/Week']:.1f} | "
                       f"Avg Price: {row['Avg. Price (ETB)']:.2f} ETB")
    else:
        logger.warning("⚠️ No vendor statistics calculated")

if __name__ == "__main__":
    main()