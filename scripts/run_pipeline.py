import os
import subprocess
import time

def run_pipeline():
    """Run the complete e-commerce data extraction pipeline"""
    print("🚀 Starting EthioMart E-commerce Data Extractor Pipeline")
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/labeled", exist_ok=True)
    os.makedirs("data/labeled/spacy", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    steps = [
        ("1. Data Ingestion", "python data_ingestion.py"),
        ("2. Data Labeling", "python data_labeling.py"),
        ("3. Convert to SpaCy Format", "python convert_conll_to_spacy.py"),
        ("4. Train NER Model", "python train_ner.py"),
        ("5. Evaluate Model", "python evaluate_model.py"),
        ("6. Process Scraped Data", "python process_scraped_data.py"),
        ("7. Generate Scorecard", "python scorecard.py"),
        ("8. Model Interpretability", "python model_interpretability.py")
    ]
    
    for name, command in steps:
        print(f"\n🔹 {name}")
        print(f"   Running: {command}")
        start_time = time.time()
        
        try:
            # Run the command and capture output
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(result.stdout)
            
            if result.stderr:
                print("⚠️ Warnings:")
                print(result.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error in {name}:")
            print(e.stderr)
            return
            
        elapsed = time.time() - start_time
        print(f"   ✅ Completed in {elapsed:.2f} seconds")
    
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()