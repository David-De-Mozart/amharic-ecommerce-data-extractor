import spacy
import numpy as np
from lime.lime_text import LimeTextExplainer
import os
import csv
import html

def explain_model(model_path):
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Load the spaCy model
    nlp = spacy.load(model_path)
    print("✅ Model loaded successfully")
    
    # Sample text for explanation
    sample_text = "በአዲስ አበባ ቦሌ ላይ እንጀራ ማሽን በ 5000 ብር ይሸጣል ለመግዛት ወደ 0912345678 ይደውሉ"
    
    # 1. Manual Feature Importance (output as CSV and HTML)
    print("\nGenerating manual feature importance...")
    try:
        doc = nlp(sample_text)
        tokens = [token.text for token in doc]
        token_importance = []
        
        # Calculate importance by removing tokens
        for i, token in enumerate(tokens):
            modified_tokens = tokens.copy()
            del modified_tokens[i]
            modified_text = " ".join(modified_tokens)
            
            modified_doc = nlp(modified_text)
            
            # Compare entities
            original_entities = {(ent.text, ent.label_) for ent in doc.ents}
            modified_entities = {(ent.text, ent.label_) for ent in modified_doc.ents}
            
            # Calculate importance as entity difference
            importance = len(original_entities - modified_entities) + len(modified_entities - original_entities)
            token_importance.append((token, importance))
        
        # Save as CSV
        with open("results/manual_importance.csv", "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Token", "Importance"])
            writer.writerows(token_importance)
        
        # Generate HTML table
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Token Importance</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>Token Importance for NER Prediction</h1>
            <table>
                <tr><th>Token</th><th>Importance Score</th></tr>
        """
        
        for token, importance in token_importance:
            html_content += f"<tr><td>{html.escape(token)}</td><td>{importance}</td></tr>"
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open("results/manual_importance.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("✅ Manual importance saved to results/manual_importance.csv and results/manual_importance.html")
    except Exception as e:
        print(f"⚠️ Manual importance failed: {e}")

    # 2. LIME Explanation
    print("\nGenerating LIME explanation...")
    try:
        def lime_predict(texts):
            results = []
            for text in texts:
                doc = nlp(text)
                probs = [
                    1.0 if any(e.label_ == "PRODUCT" for e in doc.ents) else 0.0,
                    1.0 if any(e.label_ == "PRICE" for e in doc.ents) else 0.0,
                    1.0 if any(e.label_ == "LOC" for e in doc.ents) else 0.0,
                    1.0 if any(e.label_ == "CONTACT" for e in doc.ents) else 0.0
                ]
                results.append(probs)
            return np.array(results)

        explainer = LimeTextExplainer(
            class_names=["PRODUCT", "PRICE", "LOC", "CONTACT"]
        )
        
        exp = explainer.explain_instance(
            sample_text, 
            lime_predict,
            num_features=10,
            num_samples=500
        )
        
        # Save HTML explanation
        exp.save_to_file("results/lime_explanation.html")
        print("✅ LIME HTML explanation saved to results/lime_explanation.html")
    except Exception as e:
        print(f"⚠️ LIME failed: {e}")

    # 3. Generate sample prediction report
    print("\nGenerating sample prediction report...")
    try:
        doc = nlp(sample_text)
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Sample Prediction</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .entity { padding: 4px; border-radius: 4px; margin-right: 4px; }
                .PRODUCT { background-color: #ffb3ba; }
                .PRICE { background-color: #baffc9; }
                .LOC { background-color: #bae1ff; }
                .CONTACT { background-color: #ffffba; }
            </style>
        </head>
        <body>
            <h1>Sample Prediction</h1>
            <p>Text: <span id="sample-text">"""
        
        # Create highlighted text
        text_parts = []
        last_end = 0
        for ent in doc.ents:
            text_parts.append(html.escape(sample_text[last_end:ent.start_char]))
            text_parts.append(f'<span class="entity {ent.label_}">{html.escape(ent.text)}</span>')
            last_end = ent.end_char
        text_parts.append(html.escape(sample_text[last_end:]))
        
        html_content += "".join(text_parts)
        html_content += """</span></p>
            
            <h2>Extracted Entities</h2>
            <table>
                <tr><th>Text</th><th>Label</th></tr>
        """
        
        for ent in doc.ents:
            html_content += f"<tr><td>{html.escape(ent.text)}</td><td>{ent.label_}</td></tr>"
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open("results/sample_prediction.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("✅ Sample prediction saved to results/sample_prediction.html")
    except Exception as e:
        print(f"⚠️ Sample prediction report failed: {e}")

    # 4. Print sample prediction to console
    print("\nSample Prediction:")
    print(f"Text: {sample_text}")
    for ent in doc.ents:
        print(f"  - {ent.text} ({ent.label_})")

if __name__ == "__main__":
    explain_model("models/transformer-ner")