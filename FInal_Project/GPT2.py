import os
import torch
import numpy as np
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, pipeline
from evaluate import load
import re

# Print transformers version
print(f"Transformers version: {transformers.__version__}")

# Clean broken cache
os.system("rm -rf ~/.cache/huggingface/datasets/wikitext")
os.system("rm -rf /content/cache")

# Downgrade fsspec to avoid glob error
os.system("pip install -U 'fsspec<2023.9.0' datasets transformers --quiet")

# Initialize tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="/content/cache")

def preprocess_text(sample):
    # Handle empty text
    if not sample["text"].strip():
        return {"input_ids": [], "attention_mask": []}
    
    # Clean text: remove numbers and extra whitespace
    cleaned_text = re.sub(r'\d+', '', sample["text"])
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Tokenize
    tokens = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_attention_mask=True
    )
    
    return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

# Process dataset
processed_dataset = dataset.map(preprocess_text, batched=False, remove_columns=["text"])
processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)

# Test output
print("Sample processed data:", processed_dataset["train"][0])

# Group texts into chunks
chunk_size = 256

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

# Apply chunking
training_dataset = processed_dataset.map(chunk_texts, batched=True)

# Initialize model
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Configure training parameters
training_args = TrainingArguments(
    output_dir="./gpt2-wikitext-finetuned",
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

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset["train"],
    eval_dataset=training_dataset["validation"],
    tokenizer=tokenizer
)

# Start training
print("Starting model training...")
trainer.train()
print("Training completed.")

# Evaluation
def calc_top_k_precision(predictions, true_labels, k=3):
    top_k_indices = torch.topk(predictions, k, dim=-1)[1]
    correct_preds = top_k_indices.eq(true_labels.unsqueeze(-1)).any(-1).float()
    return correct_preds.mean().item()

model.eval()
model_device = next(model.parameters()).device

# Compute perplexity
perplexity_metric = load("perplexity")
val_batch = training_dataset["validation"].select(range(10))
input_texts = [tokenizer.decode(sample["input_ids"], skip_special_tokens=True) for sample in val_batch]
perp_results = perplexity_metric.compute(model_id=model_name, predictions=input_texts)
print(f"Perplexity Score: {perp_results['mean_perplexity']:.2f}")

# Compute top-k accuracy
batch_size = 8
top_k_scores = []
with torch.no_grad():
    for i in range(0, len(training_dataset["validation"]), batch_size):
        batch = training_dataset["validation"][i:i + batch_size]
        input_ids = torch.tensor([sample["input_ids"] for sample in batch]).to(model_device)
        label_ids = torch.tensor([sample["labels"] for sample in batch]).to(model_device)
        
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = label_ids[:, 1:]
        
        for j in range(logits.shape[0]):
            score = calc_top_k_precision(logits[j], labels[j])
            top_k_scores.append(score)

avg_top_k = np.mean(top_k_scores) * 100
print(f"Top-3 Accuracy: {avg_top_k:.3f}%")

# Save model and tokenizer
save_dir = "./gpt2-wikitext-finetuned-final"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model and tokenizer saved to {save_dir}")

# Reload model
try:
    model = GPT2LMHeadModel.from_pretrained(save_dir)
    model.to(model_device)
    model.eval()
    print(f"Model loaded from {save_dir} and set to evaluation mode on {model_device}")
except Exception as e:
    print(f"Error loading model from {save_dir}: {str(e)}")

# Text generation
try:
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=model_device.index if model_device.type == "cuda" else -1
    )

    initial_prompt = "The theory of relativity states that"
    generated = text_generator(
        initial_prompt,
        max_new_tokens=100,
        do_sample=True,
        top_k=20,
        top_p=0.95,
        temperature=0.9
    )
    print(f"Generated text for '{initial_prompt}':")
    print(generated[0]["generated_text"])
    print("\n" + "="*50 + "\n")

    test_prompts = [
        "The theory of relativity states that",
        "Quantum computers are expected to",
        "Artificial intelligence can help in",
        "The capital of India is"
    ]

    for prompt in test_prompts:
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
