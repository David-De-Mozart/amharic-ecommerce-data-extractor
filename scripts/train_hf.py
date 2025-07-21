import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate  # Updated metric loading
import torch
import wandb
from spacy.tokens import DocBin
from spacy.lang.am import Amharic
import random
from sklearn.model_selection import train_test_split

# Initialize Weights & Biases
wandb.init(project="amharic-ner", tags=["assignment"])

# Configuration
MODELS = [
    "xlm-roberta-base",
    "distilbert-base-multilingual-cased",
    "bert-base-multilingual-cased"
]
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5

# Load seqeval metric
seqeval_metric = evaluate.load("seqeval")

# 1. Load and prepare dataset
def load_data():
    # Load spaCy training data
    nlp = Amharic()
    db = DocBin().from_disk("data/labeled/train.spacy")
    docs = list(db.get_docs(nlp.vocab))
    
    # Prepare examples in Hugging Face format
    examples = []
    label_set = set()
    
    for doc in docs:
        tokens = [token.text for token in doc]
        tags = ["O"] * len(tokens)
        
        for ent in doc.ents:
            tags[ent.start] = f"B-{ent.label_}"
            for i in range(ent.start + 1, ent.end):
                tags[i] = f"I-{ent.label_}"
                
        # Add to label set
        label_set.update(tags)
        
        examples.append({
            "tokens": tokens,
            "ner_tags": tags
        })
    
    # Create label mapping
    label_list = sorted(label_set)
    label2id = {tag: i for i, tag in enumerate(label_list)}
    id2label = {i: tag for i, tag in enumerate(label_list)}
    
    # Convert tags to IDs
    for example in examples:
        example["ner_tags"] = [label2id[tag] for tag in example["ner_tags"]]
    
    # Split into train/validation
    train_examples, val_examples = train_test_split(
        examples, test_size=0.2, random_state=42
    )
    
    return {
        "train": train_examples,
        "validation": val_examples
    }, label_list, id2label, label2id

# 2. Tokenize and align labels
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 3. Train and evaluate model
def train_model(model_name, dataset, label_list, id2label):
    print(f"\n🚀 Training {model_name}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id={tag: i for i, tag in enumerate(label_list)}
    )
    
    # Tokenize dataset
    tokenized_dataset = {
        "train": dataset["train"].map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer),
            batched=True
        ),
        "validation": dataset["validation"].map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer),
            batched=True
        )
    }
    
    # Training arguments
    args = TrainingArguments(
        f"{model_name}-finetuned-ner",
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        report_to="wandb",
        logging_steps=50,
        save_strategy="no",
        push_to_hub=False
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = seqeval_metric.compute(
            predictions=true_predictions, 
            references=true_labels
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"✅ {model_name} evaluation results:")
    print(eval_results)
    
    # Save model
    output_dir = f"models/{model_name.replace('/', '-')}"
    trainer.save_model(output_dir)
    print(f"💾 Model saved to {output_dir}")
    
    return eval_results

if __name__ == "__main__":
    # Load data
    dataset, label_list, id2label, label2id = load_data()
    print(f"📊 Loaded dataset with {len(dataset['train'])} training examples")
    print(f"📊 Validation examples: {len(dataset['validation'])}")
    print(f"🏷️ Label list: {label_list}")
    
    # Train and compare models
    results = {}
    for model_name in MODELS:
        results[model_name] = train_model(model_name, dataset, label_list, id2label)
    
    # Print comparison
    print("\n🔍 Model Comparison:")
    print("Model\t\tF1 Score\tPrecision\tRecall")
    for model, metrics in results.items():
        print(f"{model[:15]}\t{metrics['eval_f1']:.4f}\t\t{metrics['eval_precision']:.4f}\t\t{metrics['eval_recall']:.4f}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/model_comparison.txt", "w") as f:
        f.write("Model,F1,Precision,Recall\n")
        for model, metrics in results.items():
            f.write(f"{model},{metrics['eval_f1']},{metrics['eval_precision']},{metrics['eval_recall']}\n")
    
    print("✅ Training and evaluation complete!")