import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
import wandb
import os
from spacy.util import minibatch
from spacy_transformers import Transformer
import torch

# Initialize Weights & Biases
wandb.init(project="amharic-ner")

def load_data(data_path):
    """Load SpaCy binary data"""
    nlp = spacy.blank("am")
    db = DocBin().from_disk(data_path)
    return list(db.get_docs(nlp.vocab))

def main():
    # Create blank Amharic model
    nlp = spacy.blank("am")
    
    # Add transformer pipeline component
    config = {
        "model": {
            "@architectures": "spacy-transformers.TransformerModel.v3",
            "name": "Davlan/afro-xlmr-base",
            "tokenizer_config": {"use_fast": True}
        }
    }
    transformer = nlp.add_pipe("transformer", config=config)
    
    # Add NER pipeline
    ner = nlp.add_pipe("ner")
    
    # Add entity labels
    labels = ["PRODUCT", "PRICE", "LOC", "CONTACT"]
    for label in labels:
        ner.add_label(label)
    
    # Load training and dev data
    train_docs = load_data("data/labeled/spacy/train.spacy")
    dev_docs = load_data("data/labeled/spacy/dev.spacy")
    
    # Create training examples
    train_examples = []
    for doc in train_docs:
        example = Example(nlp.make_doc(doc.text), doc)
        train_examples.append(example)
    
    # Create dev examples for evaluation
    dev_examples = []
    for doc in dev_docs:
        example = Example(nlp.make_doc(doc.text), doc)
        dev_examples.append(example)
    
    # Initialize model
    optimizer = nlp.initialize()
    
    # Training configuration
    epochs = 30
    batch_size = 4
    best_f1 = 0.0
    
    # Training loop
    for epoch in range(epochs):
        losses = {}
        random.shuffle(train_examples)
        batches = minibatch(train_examples, size=batch_size)
        
        for batch in batches:
            nlp.update(
                batch,
                drop=0.3,
                losses=losses,
                sgd=optimizer
            )
        
        # Evaluate on dev set
        scores = nlp.evaluate(dev_examples)
        f1_score = scores.get('ents_f', 0.0)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "loss": losses.get("ner", 0),
            "f1": f1_score,
            "precision": scores.get('ents_p', 0.0),
            "recall": scores.get('ents_r', 0.0)
        })
        
        print(f"Epoch {epoch}, Loss: {losses.get('ner', 0):.2f}, F1: {f1_score:.2f}")
        
        # Save best model
        if f1_score > best_f1:
            best_f1 = f1_score
            os.makedirs("models", exist_ok=True)
            nlp.to_disk("models/transformer-ner")
            print(f"🔥 New best model saved with F1: {f1_score:.2f}")
    
    # Test the model
    print("\nFinal Test:")
    test_texts = [
        "በ አዲስ አበባ ቦሌ ላይ እንጀራ ማሽን በ 5000 ብር ይሸጣል",
        "ሳምሰንግ ስልክ በ 8000 ብር በ ሃያ �ይ",
        "ኮምፒውተር በ 25000 ብር በ አዲስ አበባ"
    ]
    
    for text in test_texts:
        doc = nlp(text)
        print(f"\nText: {text}")
        for ent in doc.ents:
            print(f"- {ent.text} ({ent.label_})")

if __name__ == "__main__":
    main()