import spacy
from spacy.tokens import DocBin

def convert_conll_to_spacy(conll_path, output_path):
    """Convert CoNLL data to spaCy format"""
    nlp = spacy.blank("am")
    db = DocBin()
    
    with open(conll_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    tokens = []
    tags = []
    for line in lines:
        if line.strip() == "":
            if tokens:
                doc = nlp.make_doc(" ".join(tokens))
                ents = []
                start = 0
                for i, token in enumerate(doc):
                    end = start + len(token)
                    if tags[i] != "O":
                        ents.append((start, end, tags[i]))
                    start = end + 1  # +1 for space
                
                try:
                    doc.ents = [doc.char_span(s, e, label=l) for s, e, l in ents]
                    db.add(doc)
                except Exception as e:
                    print(f"⚠️ Error processing: {tokens} - {e}")
            tokens = []
            tags = []
        else:
            parts = line.strip().split()
            tokens.append(parts[0])
            tags.append(parts[1])
    
    db.to_disk(output_path)
    print(f"✅ Converted {len(db)} documents to {output_path}")

if __name__ == "__main__":
    # Example usage:
    convert_conll_to_spacy("data/raw/am_train.conll", "data/labeled/train.spacy")