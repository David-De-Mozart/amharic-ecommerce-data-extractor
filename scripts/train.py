import spacy
from spacy.cli.train import train
from spacy.util import load_config
import wandb
import subprocess

def main():
    # Initialize Weights & Biases
    wandb.init(project="amharic-ner")
    
    # Load config
    config = load_config("configs/base_config.cfg")
    
    # Train the model
    train(
        config_path="configs/base_config.cfg",
        output_path="models/amharic-ner",
        use_gpu=0,  # Use GPU if available
        overrides={
            "training.seed": 42,
            "training.dropout": 0.2,
            "components.transformer.model.name": "xlm-roberta-base"
        }
    )
    
    # Log model to W&B
    wandb.save("models/amharic-ner/*")
    print("✅ Training complete! Model saved to models/amharic-ner")

if __name__ == "__main__":
    main()