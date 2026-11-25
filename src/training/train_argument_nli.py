"""
Fine-tuning script for Argument-NLI Cross-Encoder (DeBERTa-v3-small).
Input formulation: Direct Gold Perspective vs Candidate Perspective.
Resolves claim-topic attention collapse and accurately suppresses opposite-stance perspectives.
"""
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Hyperparameters
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
OUTPUT_DIR = "models/argument_nli_deberta_v3"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_LENGTH = 256
LABEL_MAP = {"contradiction": 0, "entailment": 1, "neutral": 2}
ID_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}

class ArgumentNLIDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts_a = df["text_a"].fillna("").astype(str).tolist()
        self.texts_b = df["text_b"].fillna("").astype(str).tolist()
        self.labels = [LABEL_MAP.get(str(lbl).lower(), 2) for lbl in df["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Direct Gold Perspective versus Candidate Perspective
        premise = self.texts_a[idx]
        hypothesis = self.texts_b[idx]

        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def evaluate(model, val_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = total_correct / total_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Class-wise metrics
    # Contradiction (label 0)
    tp_c = np.sum((all_preds == 0) & (all_labels == 0))
    fp_c = np.sum((all_preds == 0) & (all_labels != 0))
    fn_c = np.sum((all_preds != 0) & (all_labels == 0))
    prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
    rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0

    # Entailment (label 1)
    tp_e = np.sum((all_preds == 1) & (all_labels == 1))
    fp_e = np.sum((all_preds == 1) & (all_labels != 1))
    fn_e = np.sum((all_preds != 1) & (all_labels == 1))
    prec_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0.0
    rec_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0.0
    f1_e = 2 * prec_e * rec_e / (prec_e + rec_e) if (prec_e + rec_e) > 0 else 0.0

    # Neutral (label 2)
    tp_n = np.sum((all_preds == 2) & (all_labels == 2))
    fp_n = np.sum((all_preds == 2) & (all_labels != 2))
    fn_n = np.sum((all_preds != 2) & (all_labels == 2))
    prec_n = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
    rec_n = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
    f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if (prec_n + rec_n) > 0 else 0.0

    return {
        "accuracy": acc,
        "contra_prec": prec_c, "contra_rec": rec_c, "contra_f1": f1_c,
        "entail_prec": prec_e, "entail_rec": rec_e, "entail_f1": f1_e,
        "neutral_prec": prec_n, "neutral_rec": rec_n, "neutral_f1": f1_n
    }

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading datasets...")
    train_df = pd.read_csv("data/nli_training/train_nli_pairs.csv")
    val_df = pd.read_csv("data/nli_training/val_nli_pairs.csv")
    print(f"Loaded Train: {len(train_df)}, Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_MAP,
        label2id=LABEL_MAP
    )
    model.to(device)

    train_dataset = ArgumentNLIDataset(train_df, tokenizer)
    val_dataset = ArgumentNLIDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    print(f"\n--- Initial Zero-Shot Evaluation on Validation Set ---")
    val_metrics = evaluate(model, val_loader)
    print(f"Zero-Shot Val Acc: {val_metrics['accuracy']:.4f} | Entail Recall: {val_metrics['entail_rec']:.4f} | Contra Recall: {val_metrics['contra_rec']:.4f} | Neutral Recall: {val_metrics['neutral_rec']:.4f}")

    best_f1 = 0.0
    print(f"\nStarting fine-tuning for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s")

        # Evaluate after epoch
        val_metrics = evaluate(model, val_loader)
        print(f"\n=== Epoch {epoch+1} Validation Results ===")
        print(f"  Overall Accuracy:   {val_metrics['accuracy']:.4f}")
        print(f"  Contradiction (0):  P={val_metrics['contra_prec']:.4f}, R={val_metrics['contra_rec']:.4f}, F1={val_metrics['contra_f1']:.4f}")
        print(f"  Entailment (1):     P={val_metrics['entail_prec']:.4f}, R={val_metrics['entail_rec']:.4f}, F1={val_metrics['entail_f1']:.4f}")
        print(f"  Neutral (2):        P={val_metrics['neutral_prec']:.4f}, R={val_metrics['neutral_rec']:.4f}, F1={val_metrics['neutral_f1']:.4f}")

        # Save best model based on Entailment F1 + Contradiction Recall
        score = val_metrics["entail_f1"] + val_metrics["contra_rec"]
        if score > best_f1:
            best_f1 = score
            print(f"--> New best model (Score {best_f1:.4f})! Saving to {OUTPUT_DIR}...")
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nFine-tuning Complete! Best Model saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    train()
