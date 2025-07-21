import spacy
import json
from pathlib import Path

def load_model(model_path="models/amharic-ner"):
    if not Path(model_path).exists():
        raise ValueError(f"Model not found at {model_path}. Train first!")
    return spacy.load(model_path)

def extract_entities(text, nlp):
    doc = nlp(text)
    results = []
    for ent in doc.ents:
        results.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return results

if __name__ == "__main__":
    nlp = load_model()
    
    # Test samples
    samples = [
        "በአዲስ አበባ ቦሌ ላይ እንጀራ ማሽን በ 5000 ብር ይሸጣል",
        "የቤት እቃዎች በሰሜን ሐዋሳ በ 250 ብር ብቻ"
    ]
    
    for text in samples:
        entities = extract_entities(text, nlp)
        print(f"\n📝 Text: {text}")
        print("📌 Entities:")
        for ent in entities:
            print(f"  - {ent['label']}: {ent['text']}")