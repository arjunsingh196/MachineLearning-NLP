import os
import torch
import numpy as np
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, pipeline
from evaluate import load
import re

def setup_environment():
    """Set up the environment by cleaning cache and installing dependencies."""
    print(f"Transformers version: {transformers.__version__}")
    os.system("rm -rf ~/.cache/huggingface/datasets/wikitext")
    os.system("rm -rf /content/cache")
    os.system("pip install -U 'fsspec<2023.9.0' datasets transformers --quiet")

def initialize_tokenizer(model_name="gpt2"):
    """Initialize and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_and_preprocess_dataset(dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", cache_dir="/content/cache"):
    """Load and preprocess the dataset."""
    dataset = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    
    def preprocess_text(sample):
        if not sample["text"].strip():
            return {"input_ids": [], "attention_mask": []}
        cleaned_text = re.sub(r'\d+', '', sample["text"])
        cleaned_text = ' '.join(cleaned_text.split())
        tokens = tokenizer(
            cleaned_text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_attention_mask=True
        )
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
    
    processed_dataset = dataset.map(preprocess_text, batched=False, remove_columns=["text"])
    processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    return processed_dataset

def chunk_dataset(dataset, chunk_size=256):
    """Group dataset into fixed-size chunks."""
    def chunk_texts(data):
        combined = {key: [] for key in data.keys()}
        for key in data.keys():
            for item in data[key]:
                combined[key].extend(item)
        
        total_len = len(combined["input_ids"])
        total_len = (total_len // chunk_size) * chunk_size
        if total_len < chunk_size:
            return {key: [] for key in combined.keys()}
        
        chunked = {
            key: [combined[key][i:i + chunk_size] for i in range(0, total_len, chunk_size)]
            for key in combined.keys()
        }
        chunked["labels"] = chunked["input_ids"].copy()
        return chunked
    
    return dataset.map(chunk_texts, batched=True)

def initialize_model(model_name="gpt2", tokenizer=None):
    """Initialize and configure the GPT-2 model."""
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    return model

def configure_training_args(output_dir="./gpt2-wikitext-finetuned"):
    """Configure training arguments for the Trainer."""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=500,
        num_train_epochs=2,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_dir="./training_logs",
        logging_steps=50,
        warmup_steps=100,
        report_to="none",
        fp16=True
    )

def train_model(model, training_args, train_dataset, eval_dataset, tokenizer):
    """Train the model using the Trainer API."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    print("Starting model training...")
    trainer.train()
    print("Training completed.")
    return trainer

def calculate_top_k_precision(predictions, true_labels, k=3):
    """Calculate top-k precision for predictions."""
    top_k_indices = torch.topk(predictions, k, dim=-1)[1]
    correct_preds = top_k_indices.eq(true_labels.unsqueeze(-1)).any(-1).float()
    return correct_preds.mean().item()

def evaluate_model(model, dataset, tokenizer, model_device, batch_size=8):
    """Evaluate the model for perplexity and top-k accuracy."""
    model.eval()
    
    # Compute perplexity
    perplexity_metric = load("perplexity")
    val_batch = dataset.select(range(10))
    input_texts = [tokenizer.decode(sample["input_ids"], skip_special_tokens=True) for sample in val_batch]
    perp_results = perplexity_metric.compute(model_id=model_name, predictions=input_texts)
    print(f"Perplexity Score: {perp_results['mean_perplexity']:.2f}")
    
    # Compute top-k accuracy
    top_k_scores = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            input_ids = torch.tensor([sample["input_ids"] for sample in batch]).to(model_device)
            label_ids = torch.tensor([sample["labels"] for sample in batch]).to(model_device)
            
            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :]
            labels = label_ids[:, 1:]
            
            for j in range(logits.shape[0]):
                score = calculate_top_k_precision(logits[j], labels[j])
                top_k_scores.append(score)
    
    avg_top_k = np.mean(top_k_scores) * 100
    print(f"Top-3 Accuracy: {avg_top_k:.3f}%")

def save_model_and_tokenizer(trainer, tokenizer, save_dir="./gpt2-wikitext-finetuned-final"):
    """Save the trained model and tokenizer."""
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to {save_dir}")

def load_model(save_dir, model_device):
    """Load the trained model and set it to evaluation mode."""
    try:
        model = GPT2LMHeadModel.from_pretrained(save_dir)
        model.to(model_device)
        model.eval()
        print(f"Model loaded from {save_dir} and set to evaluation mode on {model_device}")
        return model
    except Exception as e:
        print(f"Error loading model from {save_dir}: {str(e)}")
        return None

def generate_text(model, tokenizer, prompts, model_device):
    """Generate text using the trained model."""
    try:
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=model_device.index if model_device.type == "cuda" else -1
        )
        
        for prompt in prompts:
            print(f"Prompt: {prompt}")
            output = text_generator(
                prompt,
                max_new_tokens=50,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                temperature=0.75
            )
            print(f"Generated: {output[0]['generated_text']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error during text generation: {str(e)}")

# Main execution
if __name__ == "__main__":
    model_name = "gpt2"
    cache_dir = "/content/cache"
    save_dir = "./gpt2-wikitext-finetuned-final"
    
    # Setup environment
    setup_environment()
    
    # Initialize tokenizer and model
    tokenizer = initialize_tokenizer(model_name)
    model = initialize_model(model_name, tokenizer)
    
    # Load and preprocess dataset
    processed_dataset = load_and_preprocess_dataset(cache_dir=cache_dir)
    print("Sample processed data:", processed_dataset["train"][0])
    
    # Chunk dataset
    training_dataset = chunk_dataset(processed_dataset)
    
    # Configure and train model
    training_args = configure_training_args()
    trainer = train_model(
        model,
        training_args,
        training_dataset["train"],
        training_dataset["validation"],
        tokenizer
    )
    
    # Evaluate model
    model_device = next(model.parameters()).device
    evaluate_model(model, training_dataset["validation"], tokenizer, model_device)
    
    # Save model and tokenizer
    save_model_and_tokenizer(trainer, tokenizer, save_dir)
    
    # Reload model
    model = load_model(save_dir, model_device)
    
    # Generate text
    test_prompts = [
        "The theory of relativity states that",
        "Quantum computers are expected to",
        "Artificial intelligence can help in",
        "The capital of India is"
    ]
    if model:
        generate_text(model, tokenizer, test_prompts, model_device)
