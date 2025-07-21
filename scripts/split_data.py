import random
from spacy.tokens import DocBin
import spacy
import os

def combine_and_split(input_dir, train_path, dev_path, ratio=0.8):
    """Combine all SpaCy files in a directory and split into train/dev"""
    nlp = spacy.blank("am")
    all_docs = []
    
    # Load all .spacy files in directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".spacy"):
            file_path = os.path.join(input_dir, file_name)
            db = DocBin().from_disk(file_path)
            docs = list(db.get_docs(nlp.vocab))
            all_docs.extend(docs)
            print(f"📥 Loaded {len(docs)} docs from {file_name}")
    
    if not all_docs:
        print("⚠️ No SpaCy files found in directory")
        return
    
    # Shuffle and split
    random.shuffle(all_docs)
    split_idx = int(len(all_docs) * ratio)
    train_docs = all_docs[:split_idx]
    dev_docs = all_docs[split_idx:]
    
    # Save train set
    train_db = DocBin()
    for doc in train_docs:
        train_db.add(doc)
    train_db.to_disk(train_path)
    print(f"✅ Saved {len(train_docs)} train docs to {train_path}")
    
    # Save dev set
    dev_db = DocBin()
    for doc in dev_docs:
        dev_db.add(doc)
    dev_db.to_disk(dev_path)
    print(f"✅ Saved {len(dev_docs)} dev docs to {dev_path}")

if __name__ == "__main__":
    combine_and_split(
        input_dir="data/labeled/spacy",
        train_path="data/labeled/spacy/train.spacy",
        dev_path="data/labeled/spacy/dev.spacy",
        ratio=0.8
    )