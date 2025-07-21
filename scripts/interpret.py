import shap
from lime.lime_text import LimeTextExplainer
import numpy as np
import matplotlib.pyplot as plt
import os
import spacy

# Configuration
MODEL_PATH = "models/amharic-ner"
SAMPLE_TEXT = "በ አዲስ አበባ ቦሌ ላይ እንጀራ ማሽን በ 5000 ብር ይሸጣል"

def load_model():
    return spacy.load(MODEL_PATH)

def predict_proba(texts):
    nlp = load_model()
    results = []
    for text in texts:
        doc = nlp(text)
        # Create probability distribution for each entity type
        probas = np.zeros(3)  # [PRODUCT, PRICE, LOC]
        for ent in doc.ents:
            if ent.label_ == "PRODUCT":
                probas[0] = 1.0
            elif ent.label_ == "PRICE":
                probas[1] = 1.0
            elif ent.label_ == "LOC":
                probas[2] = 1.0
        results.append(probas)
    return np.array(results)

def shap_explainer():
    nlp = load_model()
    
    # Create a prediction function
    def predict_fn(texts):
        return predict_proba(texts)
    
    # Create tokenizer that returns dictionary format
    def tokenizer(text):
        tokens = [token.text for token in nlp(text)]
        return {"input_ids": tokens}
    
    try:
        # Create SHAP explainer with Partition explainer
        explainer = shap.explainers.Partition(
            predict_fn, 
            masker=shap.maskers.Text(tokenizer, output_type="string")
        )
        
        # Compute SHAP values
        shap_values = explainer([SAMPLE_TEXT])
        
        # Plot results
        shap.plots.text(shap_values[0], display=False)
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/shap_explanation.png")
        print("💡 SHAP explanation saved to results/shap_explanation.png")
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}. Falling back to simpler visualization.")
        # Fallback visualization
        plt.figure(figsize=(10, 4))
        plt.bar(["PRODUCT", "PRICE", "LOC"], predict_proba([SAMPLE_TEXT])[0])
        plt.title("Entity Prediction Probabilities")
        plt.ylabel("Probability")
        plt.savefig("results/simple_entity_probabilities.png")
        print("💡 Saved simple probability visualization to results/simple_entity_probabilities.png")

def lime_explainer():
    nlp = load_model()
    class_names = ["PRODUCT", "PRICE", "LOC"]
    explainer = LimeTextExplainer(class_names=class_names)
    
    def predict_fn(texts):
        return predict_proba(texts)
    
    # Explain instance
    exp = explainer.explain_instance(
        SAMPLE_TEXT,
        predict_fn,
        num_features=10,
        top_labels=len(class_names)
    )
    
    # Save explanation
    os.makedirs("results", exist_ok=True)
    for label in range(len(class_names)):
        exp.save_to_file(f"results/lime_explanation_label_{label}.html")
    print("💡 LIME explanations saved to results/lime_explanation_label_*.html")

if __name__ == "__main__":
    print("🧠 Running SHAP explanation...")
    shap_explainer()
    
    print("\n🍋 Running LIME explanation...")
    lime_explainer()
    
    print("\n✅ Interpretation complete! Results saved to results/ directory")