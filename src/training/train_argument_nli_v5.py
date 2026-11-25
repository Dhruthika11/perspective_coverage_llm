"""
Argument-NLI V5 Training Script (Warm-Started 4-Class DeBERTa-v3 with Multi-Genre Balance).

Solves the two root causes identified in V4:
  1. Head Preservation: Transfers pretrained 3-class MNLI weights (contradiction, entailment, neutral)
     into classes [0, 1, 2], initializing class 3 (partial_coverage) as a semantic midpoint interpolation.
     The classification head is NEVER randomly reset.
  2. Multi-Genre Balance: Trained on balanced short human pairs and long LLM arguments (Qwen, Gemma, Llama, OLMo).
  3. Disaggregated Validation: Monitors short-human and long-LLM validation performance separately during training.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("train_argument_nli_v5")

ID_TO_LABEL_V5 = {
    0: "contradiction",
    1: "full_coverage",
    2: "no_coverage",
    3: "partial_coverage"
}

LABEL_TO_ID_V5 = {
    "contradiction": 0,
    "full_coverage": 1,
    "entailment": 1,
    "no_coverage": 2,
    "neutral": 2,
    "partial_coverage": 3
}


class ArgumentNLIV5Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Handle column variants
        if "premise" in df.columns and "hypothesis" in df.columns:
            self.premises = df["premise"].fillna("").astype(str).tolist()
            self.hypotheses = df["hypothesis"].fillna("").astype(str).tolist()
        elif "llm_argument" in df.columns and "human_perspective" in df.columns:
            self.premises = df["llm_argument"].fillna("").astype(str).tolist()
            self.hypotheses = df["human_perspective"].fillna("").astype(str).tolist()
        else:
            raise ValueError(f"Missing premise/hypothesis columns in DataFrame: {df.columns.tolist()}")

        lbl_col = "proposed_label" if "proposed_label" in df.columns else ("label_4class" if "label_4class" in df.columns else "label")
        self.labels = []
        for val in df[lbl_col]:
            if isinstance(val, (int, np.integer)):
                self.labels.append(int(val))
            else:
                norm_str = str(val).strip().lower()
                self.labels.append(LABEL_TO_ID_V5.get(norm_str, 2))

        self.regimes = df["length_regime"].tolist() if "length_regime" in df.columns else ["unknown"] * len(df)
        self.subtypes = df["subtype"].tolist() if "subtype" in df.columns else [""] * len(df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypo = self.hypotheses[idx]
        encoding = self.tokenizer(
            premise,
            hypo,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "regime": self.regimes[idx]
        }


def warm_start_4class_deberta(
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    partial_coverage_init: str = "interpolate"
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Loads pretrained 3-class NLI model and expands the classification head to 4 classes,
    strictly preserving the pretrained weights for contradiction, entailment, and neutral.
    """
    logger.info(f"Loading pretrained 3-class base model from '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    base_id2label = {int(i): str(l).lower() for i, l in base_model.config.id2label.items()}
    contra_idx = next(i for i, l in base_id2label.items() if "contra" in l)
    entail_idx = next(i for i, l in base_id2label.items() if "entail" in l)
    neutral_idx = next(i for i, l in base_id2label.items() if "neutral" in l)

    old_classifier = base_model.classifier
    old_weight = old_classifier.weight.data.clone()  # Shape: (3, hidden_size)
    old_bias = old_classifier.bias.data.clone()      # Shape: (3,)
    hidden_size = old_weight.shape[1]

    # Create new 4-class head
    new_classifier = nn.Linear(hidden_size, 4)
    with torch.no_grad():
        # Class 0: contradiction
        new_classifier.weight[0] = old_weight[contra_idx]
        new_classifier.bias[0] = old_bias[contra_idx]

        # Class 1: full_coverage
        new_classifier.weight[1] = old_weight[entail_idx]
        new_classifier.bias[1] = old_bias[entail_idx]

        # Class 2: no_coverage
        new_classifier.weight[2] = old_weight[neutral_idx]
        new_classifier.bias[2] = old_bias[neutral_idx]

        # Class 3: partial_coverage
        if partial_coverage_init == "interpolate":
            jitter = torch.randn_like(old_weight[entail_idx]) * 0.005
            new_classifier.weight[3] = 0.5 * (old_weight[entail_idx] + old_weight[neutral_idx]) + jitter
            new_classifier.bias[3] = 0.5 * (old_bias[entail_idx] + old_bias[neutral_idx])
        else:
            new_classifier.weight[3] = torch.randn(hidden_size) * 0.02
            new_classifier.bias[3] = torch.tensor(0.0)

    # Assign new head and update config
    base_model.classifier = new_classifier
    base_model.num_labels = 4
    base_model.config.num_labels = 4
    base_model.config.id2label = ID_TO_LABEL_V5
    base_model.config.label2id = {v: k for k, v in ID_TO_LABEL_V5.items()}

    logger.info("Warm-started 4-class head: preserved classes [0: contra, 1: full, 2: none] from MNLI.")
    return base_model, tokenizer


def evaluate_split_v5(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_labels = []
    all_regimes = []
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

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_regimes.extend(batch["regime"])

    avg_loss = total_loss / max(1, len(dataloader))
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_regimes = np.array(all_regimes)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1, 2, 3], zero_division=0)

    class_metrics = {}
    for i in range(4):
        c_name = ID_TO_LABEL_V5[i]
        class_metrics[c_name] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    # Disaggregated Regime Performance
    regime_metrics = {}
    for r in ["short_human", "long_llm"]:
        r_mask = all_regimes == r
        if r_mask.sum() > 0:
            r_acc = accuracy_score(all_labels[r_mask], all_preds[r_mask])
            r_f1 = f1_score(all_labels[r_mask], all_preds[r_mask], average="macro", zero_division=0)
            regime_metrics[r] = {
                "accuracy": round(float(r_acc), 4),
                "macro_f1": round(float(r_f1), 4),
                "support": int(r_mask.sum())
            }

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "class_metrics": class_metrics,
        "regime_metrics": regime_metrics
    }


def train_v5_model(
    train_csv: str = "data/nli_training_v5/train.csv",
    val_csv: str = "data/nli_training_v5/val.csv",
    output_dir: str = "models/argument_nli_deberta_v3_v5",
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 10,
    max_length: int = 256,
    use_contrastive_loss: bool = True,
    contrastive_weight: float = 0.2,
    margin: float = 0.2,
    patience: int = 4,
    dry_run: bool = False
) -> Dict[str, Any]:
    output_dir_resolved = resolve(output_dir)
    os.makedirs(output_dir_resolved, exist_ok=True)

    train_path = resolve(train_csv)
    val_path = resolve(val_csv)

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing train/val data at {train_path} or {val_path}")

    train_df = load_csv(train_path)
    val_df = load_csv(val_path)

    if dry_run:
        logger.info("Executing in DRY-RUN mode on 32 training pairs...")
        train_df = train_df.head(32)
        val_df = val_df.head(16)
        num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training V5 on device: {device} | epochs={num_epochs} | lr={learning_rate} | contrastive={use_contrastive_loss}")
    logger.info(f"Loaded: Train={len(train_df)} pairs, Val={len(val_df)} pairs")

    # Warm-start 4-class head from pretrained 3-class base
    model, tokenizer = warm_start_4class_deberta(model_name=model_name)
    model.to(device)

    train_dataset = ArgumentNLIV5Dataset(train_df, tokenizer, max_length=max_length)
    val_dataset = ArgumentNLIV5Dataset(val_df, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    criterion_ce = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    training_history = []

    logger.info("Starting V5 training loop (10 Epochs with Cosine Decay)...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        start_t = time.time()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion_ce(logits, labels)

            # Contrastive Margin Regularization: ensure full_coverage > partial_coverage for full pairs
            if use_contrastive_loss:
                is_full = (labels == 1)
                if is_full.sum() > 0:
                    z_full = logits[is_full, 1]
                    z_part = logits[is_full, 3]
                    margin_loss = torch.clamp(margin - (z_full - z_part), min=0.0).mean()
                    loss = loss + contrastive_weight * margin_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_t

        # Validation evaluation
        val_res = evaluate_split_v5(model, val_loader, device)
        logger.info(
            f"Epoch {epoch}/{num_epochs} [{epoch_time:.1f}s] - Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_res['loss']:.4f} | Val Acc: {val_res['accuracy']:.4f} | "
            f"Val Macro-F1: {val_res['macro_f1']:.4f} | "
            f"Short F1: {val_res['regime_metrics'].get('short_human', {}).get('macro_f1', 0):.4f} | "
            f"Long F1: {val_res['regime_metrics'].get('long_llm', {}).get('macro_f1', 0):.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "val_loss": val_res["loss"],
            "val_accuracy": val_res["accuracy"],
            "val_macro_f1": val_res["macro_f1"],
            "val_class_metrics": val_res["class_metrics"],
            "val_regime_metrics": val_res["regime_metrics"]
        }
        training_history.append(epoch_record)

        if val_res["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_res["macro_f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            logger.info(f"★ NEW BEST CHECKPOINT at Epoch {epoch}: Val Macro-F1 = {best_macro_f1:.4f}. Saving model...")
            model.save_pretrained(output_dir_resolved)
            tokenizer.save_pretrained(output_dir_resolved)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered: No improvement in validation Macro-F1 for {patience} epochs.")
                break

    final_report = {
        "version": "v5",
        "taxonomy": "4-class-warmstarted",
        "base_model": model_name,
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "actual_epochs_trained": len(training_history),
            "best_epoch": best_epoch,
            "max_length": max_length,
            "use_contrastive_loss": use_contrastive_loss,
            "contrastive_weight": contrastive_weight,
            "margin": margin,
            "scheduler": "cosine"
        },
        "training_history": training_history,
        "best_val_macro_f1": round(best_macro_f1, 4),
        "best_epoch": best_epoch,
        "output_directory": output_dir_resolved
    }

    report_path = os.path.join(output_dir_resolved, "training_report.json")
    save_json(final_report, report_path)
    logger.info(f"Training completed. Best checkpoint from Epoch {best_epoch} (Macro-F1: {best_macro_f1:.4f}) saved to {output_dir_resolved}")
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Argument-NLI V5 (Warm-Started 4-Class DeBERTa-v3)")
    parser.add_argument("--train_csv", type=str, default="data/nli_training_v5/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/nli_training_v5/val.csv")
    parser.add_argument("--output_dir", type=str, default="models/argument_nli_deberta_v3_v5")
    parser.add_argument("--model_name", type=str, default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    train_v5_model(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_length=args.max_length,
        dry_run=args.dry_run
    )

