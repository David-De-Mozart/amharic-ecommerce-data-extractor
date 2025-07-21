import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
import wandb
import os
from spacy.util import minibatch
from spacy.scorer import Scorer  # For evaluation

# Initialize Weights & Biases
wandb.init(project="amharic-ner")

def load_data(data_path):
    """Load SpaCy binary data"""
    nlp = spacy.blank("am")
    db = DocBin().from_disk(data_path)
    return list(db.get_docs(nlp.vocab))

def evaluate_model(nlp, examples):
    """Evaluate model on a dataset"""
    scorer = Scorer()
    scores = scorer.score(examples)
    return scores["ents_per_type"]

def main():
    # Create blank Amharic model
    nlp = spacy.blank("am")
    
    # Add NER pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add entity labels
    for label in ["PRODUCT", "PRICE", "LOC"]:
        ner.add_label(label)
    
    # Load training and dev data
    train_docs = load_data("data/labeled/spacy/train.spacy")
    dev_docs = load_data("data/labeled/spacy/dev.spacy")
    
    # Create training examples
    train_examples = []
    for doc in train_docs:
        example_doc = nlp.make_doc(doc.text)
        train_examples.append(Example(example_doc, doc))
    
    # Create dev examples for evaluation
    dev_examples = []
    for doc in dev_docs:
        example_doc = nlp.make_doc(doc.text)
        dev_examples.append(Example(example_doc, doc))
    
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
                drop=0.3,  # Dropout rate
                losses=losses,
                sgd=optimizer
            )
        
        # Evaluate on dev set
        dev_scores = evaluate_model(nlp, dev_examples)
        f1_score = dev_scores["PRODUCT"]["f"]  # Using PRODUCT F1 as main metric
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "loss": losses.get("ner", 0),
            "f1": f1_score,
            "precision": dev_scores["PRODUCT"]["p"],
            "recall": dev_scores["PRODUCT"]["r"]
        })
        
        print(f"Epoch {epoch}, Loss: {losses.get('ner', 0):.2f}, F1: {f1_score:.2f}")
        
        # Save best model
        if f1_score > best_f1:
            best_f1 = f1_score
            os.makedirs("models", exist_ok=True)
            nlp.to_disk("models/amharic-ner")
            print(f"🔥 New best model saved with F1: {f1_score:.2f}")
    
    # Test the model
    print("\nFinal Test:")
    test_texts = [
        "በ አዲስ አበባ ቦሌ ላይ እንጀራ ማሽን በ 5000 ብር ይሸጣል",
        "ሳምሰንግ ስልክ በ 8000 ብር በ ሃያ ላይ",
        "ኮምፒውተር በ 25000 ብር በ አዲስ አበባ"
    ]
    
    for text in test_texts:
        doc = nlp(text)
        print(f"\nText: {text}")
        for ent in doc.ents:
            print(f"- {ent.text} ({ent.label_})")

if __name__ == "__main__":
    main()