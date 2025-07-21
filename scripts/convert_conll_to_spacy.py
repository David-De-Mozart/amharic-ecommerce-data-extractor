import spacy
from spacy.tokens import DocBin, Span
import os
from tqdm import tqdm

def convert_conll_to_spacy(input_dir, output_dir):
    """Convert all CONLL files in a directory to SpaCy binary format"""
    nlp = spacy.blank("am")  # Blank Amharic model
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".conll"):
            continue
            
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name.replace(".conll", ".spacy"))
        
        # Read CONLL file
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        # Parse CONLL format
        db = DocBin()
        for sent in content.split("\n\n"):
            if not sent.strip():
                continue
                
            tokens = []
            labels = []
            for line in sent.split("\n"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    labels.append(parts[1])
            
            # Create SpaCy doc
            doc = nlp.make_doc(" ".join(tokens))
            ents = []
            current_ent = []
            current_label = None
            
            # Build entities using token positions instead of character spans
            for i, label in enumerate(labels):
                if label.startswith("B-"):
                    if current_ent:
                        # Create span for previous entity
                        start_token = current_ent[0]
                        end_token = current_ent[-1] + 1
                        span = doc[start_token:end_token]
                        if span:
                            ents.append(Span(doc, start_token, end_token, label=current_label))
                        current_ent = []
                    current_ent.append(i)
                    current_label = label[2:]
                elif label.startswith("I-") and current_label == label[2:]:
                    current_ent.append(i)
                else:
                    if current_ent:
                        start_token = current_ent[0]
                        end_token = current_ent[-1] + 1
                        span = doc[start_token:end_token]
                        if span:
                            ents.append(Span(doc, start_token, end_token, label=current_label))
                    current_ent = []
                    current_label = None
            
            # Add last entity if exists
            if current_ent:
                start_token = current_ent[0]
                end_token = current_ent[-1] + 1
                span = doc[start_token:end_token]
                if span:
                    ents.append(Span(doc, start_token, end_token, label=current_label))
            
            # Set entities in doc
            try:
                doc.ents = ents
            except Exception as e:
                print(f"⚠️ Error setting entities: {e}")
                # Create empty entities if there's an error
                doc.ents = []
            
            db.add(doc)
        
        # Save as SpaCy binary
        db.to_disk(output_path)
        print(f"✅ Converted {file_name} to SpaCy format with {len(db)} docs")

if __name__ == "__main__":
    convert_conll_to_spacy(
        input_dir="data/labeled",
        output_dir="data/labeled/spacy"
    )