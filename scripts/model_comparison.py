import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
import os
from pathlib import Path
import pandas as pd
from spacy.scorer import Scorer
from spacy.util import minibatch
import wandb

# Initialize Weights & Biases
wandb.init(project="amharic-ner-comparison")

def load_spacy_data(file_path):
    """Load data from SpaCy binary format"""
    nlp = spacy.blank("am")
    db = DocBin().from_disk(file_path)
    return list(db.get_docs(nlp.vocab))

def train_model(model_name, train_data, dev_data):
    """Fine-tune a transformer model for NER"""
    # Create blank model
    if model_name == "xlm-roberta":
        nlp = spacy.blank("am")
        config = {
            "model": {
                "@architectures": "spacy-transformers.TransformerModel.v3",
                "name": "Davlan/afro-xlmr-base",
                "tokenizer_config": {"use_fast": True}
            }
        }
        nlp.add_pipe("transformer", config=config)
    elif model_name == "distilbert":
        nlp = spacy.blank("am")
        config = {
            "model": {
                "@architectures": "spacy-transformers.TransformerModel.v3",
                "name": "distilbert-base-multilingual-cased",
                "tokenizer_config": {"use_fast": True}
            }
        }
        nlp.add_pipe("transformer", config=config)
    else:  # Default to spaCy CNN
        nlp = spacy.blank("am")
    
    # Add NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add entity labels
    labels = set()
    for doc in train_data + dev_data:
        for ent in doc.ents:
            labels.add(ent.label_)
    for label in labels:
        ner.add_label(label)
    
    # Disable other pipes if using transformer
    if "transformer" in nlp.pipe_names:
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in ("ner", "transformer")]
    else:
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    # Create training examples
    train_examples = []
    for doc in train_data:
        train_examples.append(Example.from_dict(nlp.make_doc(doc.text), {
            "entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        }))
    
    # Create dev examples
    dev_examples = []
    for doc in dev_data:
        dev_examples.append(Example.from_dict(nlp.make_doc(doc.text), {
            "entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        }))
    
    # Training configuration
    nlp.initialize()
    optimizer = nlp.create_optimizer()
    epochs = 10
    batch_size = 8
    best_f1 = 0
    model_dir = f"models/{model_name}"
    
    # Training loop
    print(f"⚙️ Training {model_name}...")
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
        
        # Evaluate
        scorer = Scorer()
        dev_preds = []
        for example in dev_examples:
            # FIXED: Use example.predicted.text instead of example.reference.text
            pred_doc = nlp(example.predicted.text)
            dev_preds.append(Example(pred_doc, example.reference))
        
        # Score all dev examples at once
        scores = scorer.score(dev_preds)
        f1 = scores['ents_f']
        precision = scores['ents_p']
        recall = scores['ents_r']
        
        # Log to wandb
        wandb.log({
            "model": model_name,
            "epoch": epoch,
            "loss": losses.get('ner', 0),
            "f1": f1,
            "precision": precision,
            "recall": recall
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {losses.get('ner', 0):.2f}, F1: {f1:.2f}, P: {precision:.2f}, R: {recall:.2f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(model_dir, exist_ok=True)
            nlp.to_disk(model_dir)
            print(f"🔥 Saved best {model_name} model with F1: {f1:.2f}")
    
    return nlp

def evaluate_model(nlp, examples):
    """Evaluate a model on test data"""
    scorer = Scorer()
    preds = []
    for gold_doc in examples:
        # FIXED: Use gold_doc.text directly
        pred_doc = nlp(gold_doc.text)
        preds.append(Example(pred_doc, gold_doc))
    
    scores = scorer.score(preds)
    return {
        "f1": scores['ents_f'],
        "precision": scores['ents_p'],
        "recall": scores['ents_r'],
        "speed": nlp.meta.get("speed", 0)
    }

def main():
    # Load datasets
    train_data = load_spacy_data("data/labeled/spacy/train.spacy")
    dev_data = load_spacy_data("data/labeled/spacy/dev.spacy")
    
    # Models to compare
    models = ["spacy-cnn", "xlm-roberta", "distilbert"]
    results = {}
    
    # Train and evaluate each model
    for model_name in models:
        nlp = train_model(model_name, train_data, dev_data)
        results[model_name] = evaluate_model(nlp, dev_data)
    
    # Print comparison table
    print("\n" + "="*60)
    print("Model Comparison Results:")
    print("="*60)
    print(f"{'Model':<15} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'Speed (words/s)':<15}")
    print("-"*60)
    for model, scores in results.items():
        print(f"{model:<15} {scores['f1']:.4f}    {scores['precision']:.4f}    {scores['recall']:.4f}    {scores['speed']:<15}")
    
    # Save results to CSV
    results_df = pd.DataFrame.from_dict(results, orient='index')
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/model_comparison.csv")
    print("\n💾 Saved results to results/model_comparison.csv")
    
    # Determine best model
    best_model = max(results, key=lambda x: results[x]['f1'])
    print(f"\n🏆 Best model: {best_model} with F1: {results[best_model]['f1']:.4f}")

if __name__ == "__main__":
    main()