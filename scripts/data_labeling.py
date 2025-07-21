import pandas as pd
import re
import os
from tqdm import tqdm

def label_text(text):
    labels = []
    # Improved tokenization for Amharic
    tokens = re.findall(r"[\w፩-፼]+|[^\w\s]", text)  # Better tokenization
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Price patterns (number + currency)
        if re.match(r"^[\d,]+$", token) and i+1 < len(tokens) and tokens[i+1] in ["ብር", "birr"]:
            labels.append(f"{token}\tB-PRICE")
            labels.append(f"{tokens[i+1]}\tI-PRICE")
            i += 2
            continue
        
        # Location patterns
        loc_phrases = ["አዲስ አበባ", "ቦሌ", "ፒያሳ", "ስማርያ", "መከናነብ"]
        matched = False
        for phrase in loc_phrases:
            phrase_tokens = phrase.split()
            if tokens[i:i+len(phrase_tokens)] == phrase_tokens:
                labels.append(f"{phrase_tokens[0]}\tB-LOC")
                for pt in phrase_tokens[1:]:
                    labels.append(f"{pt}\tI-LOC")
                i += len(phrase_tokens)
                matched = True
                break
        if matched:
            continue
        
        # Product patterns
        if re.search(r"ማሽን|ስልክ|ኮምፒውተር|ማጠቢያ|እንቅልፍ", token):
            labels.append(f"{token}\tB-PRODUCT")
        # Contact patterns
        elif re.match(r"09\d{8}|@\w+", token):
            labels.append(f"{token}\tB-CONTACT")
        else:
            labels.append(f"{token}\tO")
        i += 1
            
    return "\n".join(labels)

def convert_to_conll(input_file, output_file):
    # Load and preprocess data
    df = pd.read_csv(input_file)
    
    # Drop rows with missing messages and filter out empty strings
    df = df[df['message'].notna() & (df['message'] != '')]
    
    # If no messages left, use OCR text as fallback
    if len(df) == 0:
        df = df[df['ocr_text'].notna() & (df['ocr_text'] != '')]
        df['message'] = df['ocr_text']
    
    if len(df) == 0:
        print(f"⚠️ No valid text found in {input_file}")
        return
    
    # Label min(50, available samples) messages
    num_samples = min(50, len(df))
    labeled_samples = []
    
    for _, row in tqdm(df.sample(num_samples).iterrows(), total=num_samples, desc=f"Labeling {os.path.basename(input_file)}"):
        labeled = label_text(row['message'])
        labeled_samples.append(labeled + "\n\n")
    
    # Save in CoNLL format
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(labeled_samples)
    
    print(f"✅ Saved {num_samples} labeled samples to {output_file}")

if __name__ == "__main__":
    # Run for each vendor file
    import glob
    os.makedirs("data/labeled", exist_ok=True)
    
    for file in glob.glob("data/raw/*.csv"):
        output_file = file.replace("raw", "labeled").replace(".csv", ".conll")
        convert_to_conll(file, output_file)