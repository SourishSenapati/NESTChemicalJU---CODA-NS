"""
Project: CODA-NS (Integrated Insight-Driven Data-Flow Agent)
Module: Deterministic Template Alignment Protocol (DTAP) Trainer
Author: Sourish Senapati
GitHub: https://github.com/SourishSenapati

Description:
Executes the 'Zero-Entropy' training phase.
Features:
- Neuro-Symbolic Alignment: Forces model to memorize structural templates.
- Safety-First Execution: Implements 'Save-on-Interrupt' (Signal Handling).
- Auto-Resume: Automatically detects and loads the last checkpoint if training was disrupted.
"""

import os
import signal
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint

# --- CONFIGURATION ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./coda_neuro_symbolic"
# Use the optimized data for Zero-Entropy Alignment
DATA_FILE = "zero_loss_data.csv" 

# --- 1. SIGNAL HANDLER (Safe Exit) ---
def save_and_exit(trainer):
    print("\n[SYSTEM ALERT] Interrupt Signal Detected (SIGINT).")
    print("[ACTION] Serializing model state to storage...")
    trainer.save_model(OUTPUT_DIR)
    trainer.save_state()
    print("[SUCCESS] Checkpoint secured. Terminating process safely.")
    sys.exit(0)

# --- 2. MODEL INITIALIZATION ---
print(f"[INIT] Loading Base Architecture: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load Corpus - Explicit column name to handle optimize_data.py output
dataset = load_dataset("csv", data_files=DATA_FILE, split="train", column_names=["text"])

def tokenize_fn(examples):
    # Fixed padding ensures structural consistency for templates
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_ds = dataset.map(tokenize_fn, batched=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
# Enable Gradient Checkpointing - Trades tiny bit of speed for MASSIVE VRAM savings
model.gradient_checkpointing_enable()

# --- 3. TRAINING ARGUMENTS (Scientific Configuration) ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=False,    # Important: Do not overwrite if resuming
    per_device_train_batch_size=8, # Optimized for Template Memorization
    gradient_accumulation_steps=4,
    learning_rate=5e-4,            # Aggressive convergence rate
    num_train_epochs=5,            # Sufficient for Overfitting to Distribution
    save_strategy="steps",
    save_steps=200,                # Frequent checkpoints
    save_total_limit=2,            # Storage optimization
    logging_steps=10,
    fp16=False,
    bf16=True,                     # Ampere Optimization
    gradient_checkpointing=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# --- 4. EXECUTION WITH AUTO-RESUME ---
if __name__ == "__main__":
    print(f"[EXECUTION] Initializing Deterministic Template Alignment Protocol by Sourish Senapati...")
    
    # Check for existing checkpoints
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    
    try:
        if last_checkpoint:
            print(f"[RESUME] Detected previous checkpoint at: {last_checkpoint}")
            print(f"[ACTION] Resuming training state...")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            print(f"[START] No checkpoint found. Initiating fresh training run.")
            trainer.train()
            
    except KeyboardInterrupt:
        save_and_exit(trainer)
    except Exception as e:
        print(f"[CRITICAL FAILURE] {e}")
        # Try to save anyway if possible
        try:
           save_and_exit(trainer)
        except:
           pass
    
    # Finalization
    print("[COMPLETION] Training protocol finished.")
    trainer.save_model(OUTPUT_DIR)
    print(f"[ARTIFACT] Model saved to {OUTPUT_DIR}")
