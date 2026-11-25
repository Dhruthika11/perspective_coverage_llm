"""
Fine-tuning Script for Argument-NLI V2 and V3 (Cross-Encoder DeBERTa-v3).

Supports:
- V2: Direction-Corrected Baseline (Premise = AI Generation, Hypothesis = Human Perspective)
- V3: Coverage-Aware Model (Direction-Corrected + Length-Stratified Positives + Same-Stance Hard Negatives)

Key features:
- Proper directional input order: Premise = Generated Argument, Hypothesis = Human Reference
- Strict claim-level disjoint validation monitoring (Macro-F1, Class-wise Precision/Recall)
- Saves best checkpoint and training logs to models/
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("train_argument_nli_v2_v3")

LABEL_MAP = {"contradiction": 0, "entailment": 1, "neutral": 2}
ID_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}


class ArgumentNLIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Identify premise and hypothesis columns
        if "premise" in df.columns and "hypothesis" in df.columns:
            self.premises = df["premise"].fillna("").astype(str).tolist()
            self.hypotheses = df["hypothesis"].fillna("").astype(str).tolist()
        elif "text_a" in df.columns and "text_b" in df.columns:
            # In V2/V3 format: text_a is Premise (LLM), text_b is Hypothesis (Human)
            self.premises = df["text_a"].fillna("").astype(str).tolist()
            self.hypotheses = df["text_b"].fillna("").astype(str).tolist()
        else:
            raise ValueError(f"Could not find premise/hypothesis or text_a/text_b columns in {df.columns}")

        self.labels = [LABEL_MAP.get(str(lbl).lower(), 2) for lbl in df["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p = self.premises[idx]
        h = self.hypotheses[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            p,
            h,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def evaluate_split(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1, 2], zero_division=0)

    class_metrics = {
        "contradiction": {"precision": round(float(prec[0]), 4), "recall": round(float(rec[0]), 4), "f1": round(float(f1[0]), 4), "support": int(support[0])},
        "entailment": {"precision": round(float(prec[1]), 4), "recall": round(float(rec[1]), 4), "f1": round(float(f1[1]), 4), "support": int(support[1])},
        "neutral": {"precision": round(float(prec[2]), 4), "recall": round(float(rec[2]), 4), "f1": round(float(f1[2]), 4), "support": int(support[2])}
    }

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "class_metrics": class_metrics
    }


def train_model(
    version: str = "v3",
    data_dir: str = "",
    output_dir: str = "",
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_length: int = 256,
    dry_run: bool = False
) -> Dict[str, Any]:
    if not data_dir:
        data_dir = f"data/nli_training_{version}"
    if not output_dir:
        output_dir = f"models/argument_nli_deberta_v3_{version}"

    data_dir_resolved = resolve(data_dir)
    output_dir_resolved = resolve(output_dir)
    os.makedirs(output_dir_resolved, exist_ok=True)

    train_path = os.path.join(data_dir_resolved, "train.csv")
    val_path = os.path.join(data_dir_resolved, "val.csv")
    test_path = os.path.join(data_dir_resolved, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing train or val data at {data_dir_resolved}")

    train_df = load_csv(train_path)
    val_df = load_csv(val_path)
    test_df = load_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()

    if dry_run:
        logger.info("DRY-RUN mode enabled: Truncating datasets for verification.")
        train_df = train_df.head(64)
        val_df = val_df.head(32)
        test_df = test_df.head(32)
        num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Version={version.upper()} on device: {device}")
    logger.info(f"Loaded: Train={len(train_df)} pairs, Val={len(val_df)} pairs, Test={len(test_df)} pairs")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

    train_dataset = ArgumentNLIDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = ArgumentNLIDataset(val_df, tokenizer, max_length=max_length)
    test_dataset = ArgumentNLIDataset(test_df, tokenizer, max_length=max_length) if not test_df.empty else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    training_history = []

    logger.info("Starting training loop...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        start_t = time.time()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if step % max(1, len(train_loader) // 5) == 0 or step == len(train_loader):
                logger.info(f"Epoch [{epoch}/{num_epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")

        epoch_loss = running_loss / max(1, len(train_loader))
        val_res = evaluate_split(model, val_loader, device)
        elapsed = time.time() - start_t

        logger.info(f"Epoch {epoch} complete ({elapsed:.1f}s) - Train Loss: {epoch_loss:.4f} | Val Loss: {val_res['loss']:.4f} | Val Macro-F1: {val_res['macro_f1']:.4f} | Val Acc: {val_res['accuracy']:.4f}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "val_loss": val_res["loss"],
            "val_macro_f1": val_res["macro_f1"],
            "val_accuracy": val_res["accuracy"],
            "val_class_metrics": val_res["class_metrics"]
        }
        training_history.append(epoch_record)

        if val_res["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_res["macro_f1"]
            logger.info(f"New best validation Macro-F1: {best_macro_f1:.4f}. Saving checkpoint to {output_dir_resolved}...")
            model.save_pretrained(output_dir_resolved)
            tokenizer.save_pretrained(output_dir_resolved)

    # Final test evaluation on locked test set
    test_metrics = {}
    if test_loader:
        logger.info("Evaluating best model on locked held-out test split...")
        best_model = AutoModelForSequenceClassification.from_pretrained(output_dir_resolved).to(device)
        test_metrics = evaluate_split(best_model, test_loader, device)
        logger.info(f"Locked Test Results: Macro-F1={test_metrics['macro_f1']:.4f}, Accuracy={test_metrics['accuracy']:.4f}")

    final_report = {
        "version": version,
        "base_model": model_name,
        "parameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": max_length,
        },
        "training_history": training_history,
        "best_val_macro_f1": round(best_macro_f1, 4),
        "locked_test_metrics": test_metrics,
        "output_directory": output_dir_resolved
    }

    save_json(final_report, os.path.join(output_dir_resolved, "training_report.json"))
    logger.info(f"Training completed successfully. Report saved to {output_dir_resolved}/training_report.json")
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Argument-NLI V2 or V3")
    parser.add_argument("--version", type=str, default="v3", choices=["v2", "v3"], help="Dataset version (v2 or v3)")
    parser.add_argument("--model_name", type=str, default="cross-encoder/nli-deberta-v3-small", help="Pretrained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=256, help="Max token sequence length")
    parser.add_argument("--dry_run", action="store_true", help="Quick verification run with truncated data")
    args = parser.parse_args()

    train_model(
        version=args.version,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        dry_run=args.dry_run
    )

