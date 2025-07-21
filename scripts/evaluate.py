import spacy
from spacy.training import Example
from spacy.scorer import Scorer
import pandas as pd
import os

def evaluate_model(model_path, test_data_path):
    """Evaluate NER model performance"""
    nlp = spacy.load(model_path)
    db = DocBin().from_disk(test_data_path)
    examples = []
    vocab = nlp.vocab
    
    for doc in db.get_docs(vocab):
        pred_doc = nlp(doc.text)
        examples.append(Example(pred_doc, doc))
    
    # Calculate scores
    scorer = Scorer()
    scores = scorer.score(examples)
    ner_scores = scores["ents_per_type"]
    
    # Format results
    results = []
    for entity, metrics in ner_scores.items():
        results.append({
            "Entity": entity,
            "Precision": metrics["p"],
            "Recall": metrics["r"],
            "F1": metrics["f"]
        })
    
    # Save and display
    results_df = pd.DataFrame(results)
    output_path = "results/model_evaluation.csv"
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print("\nModel Evaluation Results:")
    print(results_df.to_string(index=False))
    return results_df

if __name__ == "__main__":
    evaluate_model(
        model_path="models/amharic-ner",
        test_data_path="data/labeled/spacy/dev.spacy"
    )