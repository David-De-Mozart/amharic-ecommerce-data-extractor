import spacy
from spacy.tokens import DocBin
from collections import defaultdict

def inspect_data(data_path):
    nlp = spacy.blank("am")
    db = DocBin().from_disk(data_path)
    docs = list(db.get_docs(nlp.vocab))
    
    print(f"\n📊 Found {len(docs)} documents in {data_path}")
    
    entity_counts = defaultdict(int)
    label_types = set()
    
    for i, doc in enumerate(docs[:5]):  # Inspect first 5 docs
        print(f"\n📝 Document {i+1}:")
        print(f"   Text: {doc.text[:100]}{'...' if len(doc.text) > 100 else ''}")
        
        if not doc.ents:
            print("   ⚠️ No entities found!")
        else:
            for ent in doc.ents:
                label = ent.label_
                label_types.add(label)
                entity_counts[label] += 1
                print(f"   - Entity: '{ent.text}' ({label})")
    
    print("\nEntity Distribution:")
    for label, count in entity_counts.items():
        print(f"   {label}: {count} entities")
    
    print("\nUnique Labels Found:", label_types)
    
    # Print full text of first document
    if docs:
        print("\nFull text of first document:")
        print(docs[0].text)

if __name__ == "__main__":
    print("Inspecting training data:")
    inspect_data("data/labeled/spacy/train.spacy")
    
    print("\nInspecting dev data:")
    inspect_data("data/labeled/spacy/dev.spacy")