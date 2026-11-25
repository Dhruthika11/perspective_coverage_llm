"""
Argument-NLI V4 Fine-tuning Script (4-Class Coverage Taxonomy & Contrastive Loss).

Implements Priority 2 (Contrastive Margin Loss) and Priority 3 (4-Class Coverage Taxonomy):
  - 0: contradiction (direct rebuttal / opposite stance)
  - 1: full_coverage (complete entailment of policy + rationale)
  - 2: no_coverage (same-stance distractor or topical distractor)
  - 3: partial_coverage (policy alignment, incomplete/unelaborated rationale)

Key Architectural Features:
  - Directional order: Premise = LLM Argument, Hypothesis = Human Perspective.
  - Cross-Encoder DeBERTa-v3 architecture.
  - Contrastive margin regularization against same-policy distractors.
  - Multi-class and binary coverage evaluation metrics.
  - Checkpoint tracking by validation Macro-F1.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Optional
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

logger = get_logger("train_argument_nli_v4")

# 4-Class Taxonomy Mapping
LABEL_MAP_4CLASS = {
    "contradiction": 0,
    "full_coverage": 1,
    "entailment": 1,
    "no_coverage": 2,
    "neutral": 2,
    "partial_coverage": 3
}

ID_MAP_4CLASS = {
    0: "contradiction",
    1: "full_coverage",
    2: "no_coverage",
    3: "partial_coverage"
}

# 3-Class Mapping (for backwards compatibility / ablation)
LABEL_MAP_3CLASS = {
    "contradiction": 0,
    "full_coverage": 1,
    "entailment": 1,
    "no_coverage": 2,
    "neutral": 2,
    "partial_coverage": 2  # Strict: non-full is non-entailment
}


class ArgumentNLIV4Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 256,
        num_classes: int = 4
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes

        # Extract Premise (LLM Argument) and Hypothesis (Human Perspective)
        if "llm_argument" in df.columns and "human_perspective" in df.columns:
            self.premises = df["llm_argument"].fillna("").astype(str).tolist()
            self.hypotheses = df["human_perspective"].fillna("").astype(str).tolist()
        elif "premise" in df.columns and "hypothesis" in df.columns:
            self.premises = df["premise"].fillna("").astype(str).tolist()
            self.hypotheses = df["hypothesis"].fillna("").astype(str).tolist()
        elif "text_a" in df.columns and "text_b" in df.columns:
            self.premises = df["text_a"].fillna("").astype(str).tolist()
            self.hypotheses = df["text_b"].fillna("").astype(str).tolist()
        else:
            raise ValueError(f"Could not find premise/hypothesis columns in DataFrame: {df.columns.tolist()}")

        # Extract labels
        label_col = "proposed_label" if "proposed_label" in df.columns else ("label" if "label" in df.columns else "label_id")
        lbl_mapping = LABEL_MAP_4CLASS if num_classes == 4 else LABEL_MAP_3CLASS

        self.labels = []
        for val in df[label_col]:
            if isinstance(val, (int, np.integer)):
                self.labels.append(int(val))
            else:
                norm_str = str(val).strip().lower()
                self.labels.append(lbl_mapping.get(norm_str, 2))

        self.subtypes = df["subtype"].tolist() if "subtype" in df.columns else [""] * len(df)
        self.claim_ids = df["claim_id"].tolist() if "claim_id" in df.columns else [0] * len(df)

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
            "labels": torch.tensor(label, dtype=torch.long),
            "claim_id": self.claim_ids[idx]
        }


def evaluate_split_v4(model, dataloader, device, num_classes: int = 4) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
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
            all_probs.extend(probs)

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    class_ids = list(range(num_classes))
    prec, rec, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=class_ids, zero_division=0)

    id_map = ID_MAP_4CLASS if num_classes == 4 else {0: "contradiction", 1: "entailment", 2: "neutral"}
    class_metrics = {}
    for i in class_ids:
        c_name = id_map.get(i, f"class_{i}")
        class_metrics[c_name] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    # Binary perspective coverage evaluation (full_coverage vs non_coverage)
    # Binary positive = 1 (full_coverage)
    bin_labels = [1 if y == 1 else 0 for y in all_labels]
    bin_preds = [1 if y == 1 else 0 for y in all_preds]
    bin_acc = accuracy_score(bin_labels, bin_preds)
    b_prec, b_rec, b_f1, _ = precision_recall_fscore_support(bin_labels, bin_preds, average="binary", zero_division=0)

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "class_metrics": class_metrics,
        "binary_coverage_metrics": {
            "precision": round(float(b_prec), 4),
            "recall": round(float(b_rec), 4),
            "f1": round(float(b_f1), 4),
            "accuracy": round(float(bin_acc), 4)
        }
    }


def train_v4_model(
    train_csv: str = "data/nli_training_v4/train.csv",
    val_csv: str = "data/nli_training_v4/val.csv",
    test_csv: Optional[str] = None,
    output_dir: str = "models/argument_nli_deberta_v3_v4",
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    num_classes: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 4,
    max_length: int = 256,
    use_contrastive_loss: bool = False,
    contrastive_weight: float = 0.3,
    margin: float = 0.2,
    dry_run: bool = False
) -> Dict[str, Any]:
    train_path = resolve(train_csv)
    val_path = resolve(val_csv)
    output_dir_resolved = resolve(output_dir)
    os.makedirs(output_dir_resolved, exist_ok=True)

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing train or val CSV: {train_path}, {val_path}")

    train_df = load_csv(train_path)
    val_df = load_csv(val_path)
    test_df = load_csv(resolve(test_csv)) if test_csv and os.path.exists(resolve(test_csv)) else pd.DataFrame()

    if dry_run:
        logger.info("DRY-RUN mode enabled: Truncating datasets for verification.")
        train_df = train_df.head(32)
        val_df = val_df.head(16)
        num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training V4 on device: {device} | num_classes={num_classes} | contrastive={use_contrastive_loss}")
    logger.info(f"Loaded: Train={len(train_df)} pairs, Val={len(val_df)} pairs")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    train_dataset = ArgumentNLIV4Dataset(train_df, tokenizer, max_length=max_length, num_classes=num_classes)
    val_dataset = ArgumentNLIV4Dataset(val_df, tokenizer, max_length=max_length, num_classes=num_classes)
    test_dataset = ArgumentNLIV4Dataset(test_df, tokenizer, max_length=max_length, num_classes=num_classes) if not test_df.empty else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    criterion_ce = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    training_history = []

    logger.info("Starting V4 training loop...")
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

            # Optional Contrastive Margin Regularization:
            # Penalizes model when score for full_coverage (class 1) does not exceed no_coverage (class 2)
            # on matched batches
            if use_contrastive_loss:
                probs = torch.softmax(logits, dim=1)
                p_full = probs[:, 1]
                p_no_cov = probs[:, 2]
                pos_mask = (labels == 1)
                neg_mask = (labels == 2)
                if pos_mask.any() and neg_mask.any():
                    mean_pos = p_full[pos_mask].mean()
                    mean_neg = p_full[neg_mask].mean()
                    margin_loss = torch.clamp(margin - (mean_pos - mean_neg), min=0.0)
                    loss = loss + contrastive_weight * margin_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        epoch_loss = running_loss / max(1, len(train_loader))
        val_res = evaluate_split_v4(model, val_loader, device, num_classes=num_classes)
        elapsed = time.time() - start_t

        logger.info(
            f"Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) - "
            f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_res['loss']:.4f} | "
            f"Val Macro-F1: {val_res['macro_f1']:.4f} | Val Acc: {val_res['accuracy']:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "val_loss": val_res["loss"],
            "val_macro_f1": val_res["macro_f1"],
            "val_accuracy": val_res["accuracy"],
            "val_class_metrics": val_res["class_metrics"],
            "val_binary_coverage": val_res["binary_coverage_metrics"]
        }
        training_history.append(epoch_record)

        if val_res["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_res["macro_f1"]
            logger.info(f"New best validation Macro-F1: {best_macro_f1:.4f}. Saving checkpoint to {output_dir_resolved}...")
            model.save_pretrained(output_dir_resolved)
            tokenizer.save_pretrained(output_dir_resolved)

    # Optional test evaluation
    test_metrics = {}
    if test_loader:
        logger.info("Evaluating best model on test split...")
        best_model = AutoModelForSequenceClassification.from_pretrained(output_dir_resolved).to(device)
        test_metrics = evaluate_split_v4(best_model, test_loader, device, num_classes=num_classes)

    final_report = {
        "version": "v4",
        "taxonomy": "4-class" if num_classes == 4 else "3-class",
        "base_model": model_name,
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": max_length,
            "use_contrastive_loss": use_contrastive_loss,
            "contrastive_weight": contrastive_weight,
            "margin": margin
        },
        "training_history": training_history,
        "best_val_macro_f1": round(best_macro_f1, 4),
        "test_metrics": test_metrics,
        "output_directory": output_dir_resolved
    }

    report_path = os.path.join(output_dir_resolved, "training_report.json")
    save_json(final_report, report_path)
    logger.info(f"Training completed successfully. Report saved to {report_path}")
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Argument-NLI V4 (4-Class Coverage Taxonomy)")
    parser.add_argument("--train_csv", type=str, default="data/nli_training_v4/train.csv", help="Train CSV path")
    parser.add_argument("--val_csv", type=str, default="data/nli_training_v4/val.csv", help="Val CSV path")
    parser.add_argument("--test_csv", type=str, default=None, help="Optional test CSV path")
    parser.add_argument("--output_dir", type=str, default="models/argument_nli_deberta_v3_v4", help="Output directory")
    parser.add_argument("--model_name", type=str, default="cross-encoder/nli-deberta-v3-small", help="Pretrained model")
    parser.add_argument("--num_classes", type=int, default=4, choices=[3, 4], help="4 for V4 taxonomy, 3 for standard NLI")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--use_contrastive_loss", action="store_true", help="Enable margin contrastive loss")
    parser.add_argument("--contrastive_weight", type=float, default=0.3, help="Weight for margin contrastive loss")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for contrastive loss")
    parser.add_argument("--dry_run", action="store_true", help="Quick run with truncated batches")
    args = parser.parse_args()

    train_v4_model(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        use_contrastive_loss=args.use_contrastive_loss,
        contrastive_weight=args.contrastive_weight,
        margin=args.margin,
        dry_run=args.dry_run
    )

