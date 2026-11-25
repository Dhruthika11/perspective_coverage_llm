"""
Production 3-Seed Ensemble Training, Validation-Only Head Selection, Temperature Calibration,
and Clean Evaluation on Held-Out Benchmarks for Argument-NLI V5.

Strict Methodological Protocol:
  1. Validation-Only Model Selection:
     - Compares 'data_only' (Random Head + Balanced Data) vs 'warm_head' (Warm MNLI Head + Balanced Data)
       strictly on the disjoint validation set (val.csv, N=800).
     - Does NOT look at the locked test set or fresh claims to make this decision.
     - If 'data_only' achieves equal or superior Validation AUROC, it is selected as the Main Model (Occam's razor).
  2. 3-Seed Ensemble Freezing:
     - Freezes all 3 checkpoints (seeds 42, 123, 999) of the winning architecture.
     - Ensembles predictions via probability and logit averaging across the 3 seeds to eliminate seed variance.
  3. Probability Calibration via Temperature Scaling:
     - Optimizes scalar temperature T* strictly on the validation set to minimize Negative Log-Likelihood (NLL).
     - Evaluates Expected Calibration Error (ECE) and Brier Score before and after scaling.
  4. Operating Threshold Locking:
     - Determines optimal F1 threshold (theta_F1*) and high-precision threshold (theta_prec90*) strictly on validation.
     - Establishes a 3-tier triage policy: Accept (high-confidence entailment) / Review (ambiguous) / Reject (non-entailment).
     - Locks and freezes all thresholds ("write the numbers down and don't touch them again").
  5. Pristine Evaluation on Unseen Test Sets:
     - Locked 1,118 Test Benchmark (data/nli_training_v3/test.csv)
     - Pristine 50-Claim Fresh Benchmark (data/fresh_claim_benchmark/test_fresh_claims.csv, N=1,110)
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, save_json, load_csv, load_json
from src.utils.logging_utils import get_logger
from src.training.train_v5_ablations import (
    get_ablation_datasets,
    warm_start_4class_deberta,
    evaluate_strata_metrics
)

logger = get_logger("train_and_calibrate_ensemble")


# -----------------------------------------------------------------------------
# 1. Calibration & Metric Helper Functions
# -----------------------------------------------------------------------------

def optimize_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Finds optimal temperature T > 0 strictly on validation set minimizing NLL.
    """
    def nll_objective(T: float) -> float:
        if T <= 0.05:
            return 1e9
        scaled = logits / T
        max_l = np.max(scaled, axis=1, keepdims=True)
        log_sum_exp = max_l + np.log(np.sum(np.exp(scaled - max_l), axis=1, keepdims=True))
        log_probs = scaled - log_sum_exp
        nll = -np.mean(log_probs[np.arange(len(labels)), labels])
        return float(nll)

    res = minimize_scalar(nll_objective, bounds=(0.1, 5.0), method="bounded")
    return float(res.x)


def compute_nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Computes categorical Negative Log-Likelihood (cross-entropy loss)."""
    scaled = logits / temperature
    max_l = np.max(scaled, axis=1, keepdims=True)
    log_sum_exp = max_l + np.log(np.sum(np.exp(scaled - max_l), axis=1, keepdims=True))
    log_probs = scaled - log_sum_exp
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


def compute_multiclass_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes top-1 multi-class Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = int(np.sum(bin_mask))
        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[bin_mask]))
            bin_conf = float(np.mean(confidences[bin_mask]))
            ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier_score(prob_positive: np.ndarray, y_binary: np.ndarray) -> float:
    """Computes mean squared error Brier score for binary entailment."""
    return float(np.mean((prob_positive - y_binary) ** 2))


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Computes AUROC safely handling single-class edge cases."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


# -----------------------------------------------------------------------------
# 2. Training and Checkpoint Management
# -----------------------------------------------------------------------------

def train_or_load_checkpoint(
    config_name: str,
    seed: int,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    models_base_dir: str = "models",
    mock_run: bool = False
) -> Dict[str, Any]:
    """
    Trains a checkpoint if not already saved, saving the best validation checkpoint to disk.
    """
    ckpt_dir = resolve(os.path.join(models_base_dir, f"ensemble_{config_name}", f"seed_{seed}"))
    os.makedirs(ckpt_dir, exist_ok=True)

    if mock_run:
        logger.info(f"[MOCK] Returning calibrated checkpoint metadata for {config_name} (Seed {seed})...")
        noise = (seed % 10 - 5) * 0.001
        base_val_auc = 0.9125 if config_name == "data_only" else 0.9080
        base_val_f1 = 0.7480 if config_name == "data_only" else 0.7420
        return {
            "config": config_name,
            "seed": seed,
            "checkpoint_dir": ckpt_dir,
            "best_epoch": 6,
            "val_macro_f1": round(base_val_f1 + noise, 4),
            "val_auroc": round(base_val_auc + noise, 4)
        }

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df_train, df_val = get_ablation_datasets("full")

    # Fast batch tokenization into TensorDataset
    def build_tensor_dataset(df: pd.DataFrame) -> TensorDataset:
        enc = tokenizer(
            df["premise"].fillna("").astype(str).tolist(),
            df["hypothesis"].fillna("").astype(str).tolist(),
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        if "label_4class" in df.columns:
            labels = torch.tensor(df["label_4class"].astype(int).tolist(), dtype=torch.long)
        else:
            labels = torch.tensor([1 if str(l).lower() == "entailment" else 2 for l in df["label"]], dtype=torch.long)
        return TensorDataset(enc["input_ids"], enc["attention_mask"], labels)

    train_dataset = build_tensor_dataset(df_train)
    val_dataset = build_tensor_dataset(df_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model architecture
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    if config_name == "warm_head":
        model = warm_start_4class_deberta(model, base_model_name=model_name)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = 0

    y_val_binary = (df_val["label_4class"] == 1).astype(int).to_numpy()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_ids, batch_mask, batch_labels in train_loader:
            optimizer.zero_grad()
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_logits_list = []
        with torch.no_grad():
            for batch_ids, batch_mask, _ in val_loader:
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits
                val_logits_list.append(logits.cpu().numpy())

        val_logits = np.concatenate(val_logits_list, axis=0)
        val_preds = np.argmax(val_logits, axis=-1)
        val_targets = df_val["label_4class"].to_numpy()

        _, _, val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)
        val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
        val_probs = val_probs / np.sum(val_probs, axis=1, keepdims=True)
        val_auc = float(roc_auc_score(y_val_binary, val_probs[:, 1]))

        logger.info(
            f"[{config_name.upper()} | Seed {seed}] Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f} | Val AUROC: {val_auc:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_epoch = epoch
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    logger.info(f"Saved optimal checkpoint for {config_name} (Seed {seed}) at Epoch {best_epoch} (Val F1: {best_val_f1:.4f}) to {ckpt_dir}")
    return {
        "config": config_name,
        "seed": seed,
        "checkpoint_dir": ckpt_dir,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1
    }


# -----------------------------------------------------------------------------
# 3. Model Inference Helper
# -----------------------------------------------------------------------------

def predict_split(
    checkpoint_dirs: List[str],
    df: pd.DataFrame,
    batch_size: int = 16,
    mock_run: bool = False,
    mock_base_auc: float = 0.9100
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Computes ensemble predictions across all checkpoints.
    Returns:
      - ensemble_logits (N, 4)
      - ensemble_probs (N, 4)
      - per_model_logits List[(N, 4)]
    """
    n_samples = len(df)

    if mock_run:
        # High fidelity simulation of ensemble predictions
        # Generate realistic calibrated logits
        y_label = df["label_4class"].to_numpy() if "label_4class" in df.columns else (
            (df["label"].str.lower() == "entailment").astype(int) * 1 +
            (df["label"].str.lower() == "contradiction").astype(int) * 0 +
            (df["label"].str.lower() == "neutral").astype(int) * 2
        )
        per_model_logits = []
        for i, _ in enumerate(checkpoint_dirs):
            rng = np.random.RandomState(42 + i * 37)
            logits = np.zeros((n_samples, 4))
            for c in range(4):
                match = (y_label == c).astype(float)
                logits[:, c] = match * 1.4 + rng.normal(0, 1.05, n_samples)
            per_model_logits.append(logits)

        ens_logits = np.mean(per_model_logits, axis=0)
        # Softmax
        exp_l = np.exp(ens_logits - np.max(ens_logits, axis=1, keepdims=True))
        ens_probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        return ens_logits, ens_probs, per_model_logits

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dirs[0])

    enc = tokenizer(
        df["premise"].fillna("").astype(str).tolist(),
        df["hypothesis"].fillna("").astype(str).tolist(),
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    per_model_logits = []
    for ckpt in checkpoint_dirs:
        logger.info(f"Loading checkpoint for inference: {ckpt}")
        model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
        model.eval()

        ckpt_logits = []
        with torch.no_grad():
            for b_ids, b_mask in loader:
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                l = model(input_ids=b_ids, attention_mask=b_mask).logits
                ckpt_logits.append(l.cpu().numpy())

        per_model_logits.append(np.concatenate(ckpt_logits, axis=0))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ens_logits = np.mean(per_model_logits, axis=0)
    # Ensemble probability averaging
    all_probs = []
    for logits in per_model_logits:
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        all_probs.append(exp_l / np.sum(exp_l, axis=1, keepdims=True))
    ens_probs = np.mean(all_probs, axis=0)

    return ens_logits, ens_probs, per_model_logits


# -----------------------------------------------------------------------------
# 4. Main Workflow Implementation
# -----------------------------------------------------------------------------

def run_ensemble_selection_calibration_and_evaluation(
    seeds: List[int] = [42, 123, 999],
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    mock_run: bool = False,
    output_dir: str = "results/metrics"
) -> Dict[str, Any]:
    """
    Executes the entire rigorous 5-step methodology:
      Step 1: Validation-only head comparison (data_only vs warm_head).
      Step 2: 3-seed ensemble training & freezing for winning architecture.
      Step 3: Temperature scaling on validation set (T*).
      Step 4: Pick and lock operating thresholds on validation set (theta_F1*, theta_prec90*).
      Step 5: Clean evaluation on locked 1,118 test set and fresh claims benchmark.
    """
    os.makedirs(resolve(output_dir), exist_ok=True)
    df_val = load_csv(resolve("data/nli_training_v5/val.csv"))
    df_test = load_csv(resolve("data/nli_training_v3/test.csv"))
    df_fresh = load_csv(resolve("data/fresh_claim_benchmark/test_fresh_claims.csv"))

    y_val_labels = df_val["label_4class"].to_numpy()
    y_val_binary = (df_val["label_4class"] == 1).astype(int).to_numpy()

    logger.info("=" * 80)
    logger.info("PHASE 1: VALIDATION-ONLY MODEL SELECTION (Zero Test Peeking)")
    logger.info("=" * 80)

    # 1. Train / Load checkpoints for both candidate architectures across 3 seeds
    data_only_ckpts = []
    warm_head_ckpts = []

    for s in seeds:
        res_data = train_or_load_checkpoint("data_only", seed=s, epochs=epochs, batch_size=batch_size, lr=lr, mock_run=mock_run)
        res_warm = train_or_load_checkpoint("warm_head", seed=s, epochs=epochs, batch_size=batch_size, lr=lr, mock_run=mock_run)
        data_only_ckpts.append(res_data["checkpoint_dir"])
        warm_head_ckpts.append(res_warm["checkpoint_dir"])

    # 2. Predict on validation set
    val_data_logits, val_data_probs, val_data_per_seed = predict_split(data_only_ckpts, df_val, batch_size=batch_size, mock_run=mock_run, mock_base_auc=0.912)
    val_warm_logits, val_warm_probs, val_warm_per_seed = predict_split(warm_head_ckpts, df_val, batch_size=batch_size, mock_run=mock_run, mock_base_auc=0.908)

    # Compute validation metrics
    val_data_aucs = [float(roc_auc_score(y_val_binary, p[:, 1])) for p in [np.exp(l)/np.sum(np.exp(l), axis=1, keepdims=True) for l in val_data_per_seed]]
    val_warm_aucs = [float(roc_auc_score(y_val_binary, p[:, 1])) for p in [np.exp(l)/np.sum(np.exp(l), axis=1, keepdims=True) for l in val_warm_per_seed]]

    val_data_ens_auc = float(roc_auc_score(y_val_binary, val_data_probs[:, 1]))
    val_warm_ens_auc = float(roc_auc_score(y_val_binary, val_warm_probs[:, 1]))

    val_data_f1s = [float(precision_recall_fscore_support(y_val_labels, np.argmax(l, axis=-1), average="macro", zero_division=0)[2]) for l in val_data_per_seed]
    val_warm_f1s = [float(precision_recall_fscore_support(y_val_labels, np.argmax(l, axis=-1), average="macro", zero_division=0)[2]) for l in val_warm_per_seed]

    logger.info(f"Validation AUROC -> Data-Only: {val_data_ens_auc:.4f} (Mean: {np.mean(val_data_aucs):.4f} ± {np.std(val_data_aucs):.4f})")
    logger.info(f"Validation AUROC -> Warm-Head:  {val_warm_ens_auc:.4f} (Mean: {np.mean(val_warm_aucs):.4f} ± {np.std(val_warm_aucs):.4f})")

    # Strict Decision Rule (Occam's razor):
    # If Data-Only achieves equal or superior validation AUROC (within 0.001 tolerance), pick Data-Only because it is simpler.
    if val_data_ens_auc >= (val_warm_ens_auc - 0.001):
        winning_arch = "data_only"
        winning_label = "Data-Only (Random Head + Balanced Multi-Genre Data)"
        winning_ckpts = data_only_ckpts
        winning_val_logits = val_data_logits
        winning_val_probs = val_data_probs
        rationale = (
            "Data-Only achieved equal or superior AUROC on the disjoint validation set "
            f"({val_data_ens_auc:.4f} vs {val_warm_ens_auc:.4f}). "
            "Selected as the official Main Model by Occam's razor: simpler architecture that learns "
            "directly from balanced multi-genre supervision without external MNLI head surgery or label re-mapping."
        )
    else:
        winning_arch = "warm_head"
        winning_label = "Full V5 (Warm MNLI Head + Balanced Multi-Genre Data)"
        winning_ckpts = warm_head_ckpts
        winning_val_logits = val_warm_logits
        winning_val_probs = val_warm_probs
        rationale = (
            f"Warm-Head achieved superior AUROC on the disjoint validation set ({val_warm_ens_auc:.4f} vs {val_data_ens_auc:.4f}). "
            "Selected as the official Main Model."
        )

    logger.info(f"==> WINNING ARCHITECTURE SELECTED ON VALIDATION: {winning_arch.upper()} <==")
    logger.info(f"Rationale: {rationale}")

    decision_record = {
        "selection_set": "Validation Split Only (data/nli_training_v5/val.csv, N=800)",
        "selected_architecture": winning_arch,
        "selected_architecture_full_name": winning_label,
        "selection_rationale": rationale,
        "validation_metrics": {
            "data_only": {
                "ensemble_auroc": round(val_data_ens_auc, 4),
                "seed_mean_auroc": round(float(np.mean(val_data_aucs)), 4),
                "seed_std_auroc": round(float(np.std(val_data_aucs)), 4),
                "seed_macro_f1_mean": round(float(np.mean(val_data_f1s)), 4)
            },
            "warm_head": {
                "ensemble_auroc": round(val_warm_ens_auc, 4),
                "seed_mean_auroc": round(float(np.mean(val_warm_aucs)), 4),
                "seed_std_auroc": round(float(np.std(val_warm_aucs)), 4),
                "seed_macro_f1_mean": round(float(np.mean(val_warm_f1s)), 4)
            }
        }
    }
    decision_file = resolve(os.path.join(output_dir, "model_selection_validation_decision.json"))
    save_json(decision_record, decision_file)

    # -------------------------------------------------------------------------
    # PHASE 2: TEMPERATURE SCALING & CALIBRATION ON VALIDATION SET
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PHASE 2: PROBABILITY CALIBRATION (TEMPERATURE SCALING ON VALIDATION)")
    logger.info("=" * 80)

    # 1. Uncalibrated metrics
    uncal_nll = compute_nll(winning_val_logits, y_val_labels, temperature=1.0)
    uncal_ece = compute_multiclass_ece(winning_val_probs, y_val_labels, n_bins=10)
    uncal_brier = compute_brier_score(winning_val_probs[:, 1], y_val_binary)

    # 2. Optimize temperature T*
    T_star = optimize_temperature(winning_val_logits, y_val_labels)
    logger.info(f"Optimal Temperature found via Brent minimization: T* = {T_star:.4f}")

    # 3. Calibrated probabilities
    cal_logits = winning_val_logits / T_star
    exp_cal = np.exp(cal_logits - np.max(cal_logits, axis=1, keepdims=True))
    cal_val_probs = exp_cal / np.sum(exp_cal, axis=1, keepdims=True)

    cal_nll = compute_nll(winning_val_logits, y_val_labels, temperature=T_star)
    cal_ece = compute_multiclass_ece(cal_val_probs, y_val_labels, n_bins=10)
    cal_brier = compute_brier_score(cal_val_probs[:, 1], y_val_binary)

    logger.info(f"Validation NLL:   {uncal_nll:.4f} -> {cal_nll:.4f} ({'Improved' if cal_nll < uncal_nll else 'Maintained'})")
    logger.info(f"Validation ECE:   {uncal_ece*100:.2f}% -> {cal_ece*100:.2f}%")
    logger.info(f"Validation Brier: {uncal_brier:.4f} -> {cal_brier:.4f}")

    # -------------------------------------------------------------------------
    # PHASE 3: PICK AND LOCK OPERATING THRESHOLDS ON VALIDATION SET
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PHASE 3: OPERATING THRESHOLD SELECTION (FROZEN ON VALIDATION)")
    logger.info("=" * 80)

    p_val_cal_full = cal_val_probs[:, 1]
    thetas = np.linspace(0.05, 0.95, 91)

    best_f1 = -1.0
    theta_f1_star = 0.50
    theta_prec90_star = 0.75
    found_prec90 = False

    for th in thetas:
        pred_bin = (p_val_cal_full >= th).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_val_binary, pred_bin, average="binary", zero_division=0)
        if f > best_f1:
            best_f1 = f
            theta_f1_star = float(th)
        if p >= 0.90 and not found_prec90:
            theta_prec90_star = float(th)
            found_prec90 = True

    # Define 3-tier triage thresholds
    # Accept: >= theta_accept (confident coverage)
    # Review: theta_reject <= P < theta_accept (ambiguous / partial)
    # Reject: < theta_reject (confident non-coverage / contradiction)
    theta_accept = round(theta_prec90_star, 2)
    theta_reject = round(max(0.20, theta_f1_star - 0.25), 2)
    if theta_reject >= theta_accept:
        theta_reject = round(max(0.10, theta_accept - 0.15), 2)
    theta_f1_star = round(theta_f1_star, 2)

    logger.info(f"FROZEN VALIDATION THRESHOLDS:")
    logger.info(f"  * Optimal F1 Threshold:     theta_F1*     = {theta_f1_star:.2f} (Val F1 = {best_f1:.4f})")
    logger.info(f"  * High-Precision Threshold: theta_prec90* = {theta_accept:.2f}")
    logger.info(f"  * 3-Tier Policy:            Accept >= {theta_accept:.2f} | Review [{theta_reject:.2f}, {theta_accept:.2f}) | Reject < {theta_reject:.2f}")

    thresholds_record = {
        "status": "FROZEN (DO NOT MODIFY)",
        "tuning_set": "Validation Split Only (data/nli_training_v5/val.csv, N=800)",
        "ensemble_checkpoints": winning_ckpts,
        "temperature_scaling": {
            "optimal_temperature_T": round(T_star, 4),
            "uncalibrated_nll": round(uncal_nll, 4),
            "calibrated_nll": round(cal_nll, 4),
            "uncalibrated_ece": round(uncal_ece, 4),
            "calibrated_ece": round(cal_ece, 4),
            "uncalibrated_brier": round(uncal_brier, 4),
            "calibrated_brier": round(cal_brier, 4)
        },
        "operating_thresholds": {
            "theta_f1_star": theta_f1_star,
            "theta_high_precision": theta_accept,
            "val_f1_at_theta_f1": round(float(best_f1), 4)
        },
        "three_tier_triage_policy": {
            "accept_threshold": theta_accept,
            "reject_threshold": theta_reject,
            "description": "Predictions with P(full) >= theta_accept are Accepted automatically; P(full) < theta_reject are Rejected; intermediate values are routed to Review."
        }
    }
    thresholds_file = resolve(os.path.join(output_dir, "ensemble_validation_thresholds.json"))
    save_json(thresholds_record, thresholds_file)

    # -------------------------------------------------------------------------
    # PHASE 4: CLEAN EVALUATION ON LOCKED 1,118 TEST BENCHMARK
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PHASE 4: CLEAN EVALUATION ON LOCKED 1,118 TEST SET (ZERO TEST PEEKING)")
    logger.info("=" * 80)

    test_logits, test_probs, _ = predict_split(winning_ckpts, df_test, batch_size=batch_size, mock_run=mock_run, mock_base_auc=0.915)

    # Apply frozen temperature scaling
    test_cal_logits = test_logits / T_star
    exp_test = np.exp(test_cal_logits - np.max(test_cal_logits, axis=1, keepdims=True))
    test_cal_probs = exp_test / np.sum(exp_test, axis=1, keepdims=True)

    y_test_binary = (df_test["label"].str.lower() == "entailment").astype(int).to_numpy()
    cats_test = df_test["pair_category"].to_numpy()

    # Strata Metrics
    test_strata = evaluate_strata_metrics(y_test_binary, test_cal_probs, cats_test)

    # Metrics at frozen thresholds
    p_test_full = test_cal_probs[:, 1]
    pred_f1 = (p_test_full >= theta_f1_star).astype(int)
    p_f1, r_f1, f_f1, _ = precision_recall_fscore_support(y_test_binary, pred_f1, average="binary", zero_division=0)

    pred_prec90 = (p_test_full >= theta_accept).astype(int)
    p_prec, r_prec, f_prec, _ = precision_recall_fscore_support(y_test_binary, pred_prec90, average="binary", zero_division=0)

    # Contradiction detection
    opp_mask = (cats_test == "opposite_stance_conflict")
    n_opp = int(np.sum(opp_mask))
    contra_caught = int(np.sum(np.argmax(test_cal_probs, axis=1)[opp_mask] == 0))

    # Triage distributions on test set
    n_accept = int(np.sum(p_test_full >= theta_accept))
    n_review = int(np.sum((p_test_full >= theta_reject) & (p_test_full < theta_accept)))
    n_reject = int(np.sum(p_test_full < theta_reject))

    # Accuracy within accept tier
    accept_mask = (p_test_full >= theta_accept)
    accept_precision = float(np.mean(y_test_binary[accept_mask])) if n_accept > 0 else 0.0

    test_report = {
        "evaluation_dataset": "Locked 1,118 Test Benchmark (data/nli_training_v3/test.csv)",
        "ensemble_model": winning_label,
        "temperature_T_applied": round(T_star, 4),
        "disaggregated_auroc": {
            "macro_combined": test_strata["overall_macro_auroc"],
            "stratum_1_short_human": test_strata["stratum_1_short_human_auroc"],
            "stratum_2_long_llm": test_strata["stratum_2_long_llm_auroc"],
            "stratum_3_stance_separation": test_strata["stratum_3_stance_separation_auroc"]
        },
        "adversarial_robustness": {
            "contradictions_caught": f"{contra_caught}/{n_opp} ({contra_caught/max(1, n_opp)*100:.1f}%)"
        },
        "metrics_at_frozen_theta_f1": {
            "threshold": theta_f1_star,
            "precision": round(float(p_f1), 4),
            "recall": round(float(r_f1), 4),
            "f1": round(float(f_f1), 4)
        },
        "metrics_at_frozen_theta_high_precision": {
            "threshold": theta_accept,
            "precision": round(float(p_prec), 4),
            "recall": round(float(r_prec), 4),
            "f1": round(float(f_prec), 4)
        },
        "three_tier_triage_evaluation": {
            "accept_count": n_accept,
            "accept_percentage": round(n_accept / len(df_test) * 100, 2),
            "accept_tier_precision": round(accept_precision * 100, 2),
            "review_count": n_review,
            "review_percentage": round(n_review / len(df_test) * 100, 2),
            "reject_count": n_reject,
            "reject_percentage": round(n_reject / len(df_test) * 100, 2)
        }
    }
    test_report_file = resolve(os.path.join(output_dir, "final_ensemble_test_benchmark_report.json"))
    save_json(test_report, test_report_file)

    # -------------------------------------------------------------------------
    # PHASE 5: CLEAN EVALUATION ON PRISTINE 50-CLAIM FRESH BENCHMARK
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PHASE 5: CLEAN EVALUATION ON PRISTINE FRESH CLAIMS (N=1,110, ZERO CONTAMINATION)")
    logger.info("=" * 80)

    fresh_logits, fresh_probs, _ = predict_split(winning_ckpts, df_fresh, batch_size=batch_size, mock_run=mock_run, mock_base_auc=0.918)
    fresh_cal_logits = fresh_logits / T_star
    exp_fresh = np.exp(fresh_cal_logits - np.max(fresh_cal_logits, axis=1, keepdims=True))
    fresh_cal_probs = exp_fresh / np.sum(exp_fresh, axis=1, keepdims=True)

    y_fresh_binary = (df_fresh["label"].str.lower() == "entailment").astype(int).to_numpy()
    cats_fresh = df_fresh["pair_category"].to_numpy()
    p_fresh_full = fresh_cal_probs[:, 1]

    # Strata masks
    short_mask = np.isin(cats_fresh, [
        "within_cluster_paraphrase",
        "same_stance_different_reasoning",
        "opposite_stance_conflict",
        "unrelated_different_claim"
    ])
    long_mask = np.isin(cats_fresh, [
        "extended_discourse_positive",
        "extended_discourse_neutral_distractor",
        "extended_discourse_contradiction"
    ])

    auc_fresh_macro = safe_roc_auc(y_fresh_binary, p_fresh_full)
    auc_fresh_short = safe_roc_auc(y_fresh_binary[short_mask], p_fresh_full[short_mask]) if np.sum(short_mask) > 0 else None
    auc_fresh_long = safe_roc_auc(y_fresh_binary[long_mask], p_fresh_full[long_mask]) if np.sum(long_mask) > 0 else None

    # Stance separation
    stance_pos = (cats_fresh == "extended_discourse_positive")
    stance_neg = (cats_fresh == "same_stance_different_reasoning")
    stance_mask = stance_pos | stance_neg
    auc_fresh_stance = safe_roc_auc(y_fresh_binary[stance_mask], p_fresh_full[stance_mask]) if np.sum(stance_mask) > 0 else None

    # Contradictions
    fresh_contra_mask = (cats_fresh == "opposite_stance_conflict") | (cats_fresh == "extended_discourse_contradiction")
    n_fresh_contra = int(np.sum(fresh_contra_mask))
    fresh_contra_caught = int(np.sum(np.argmax(fresh_cal_probs, axis=1)[fresh_contra_mask] == 0))

    # Metrics at frozen theta_F1
    fresh_pred_f1 = (p_fresh_full >= theta_f1_star).astype(int)
    pf_f1, rf_f1, ff_f1, _ = precision_recall_fscore_support(y_fresh_binary, fresh_pred_f1, average="binary", zero_division=0)

    fresh_report = {
        "evaluation_dataset": "Pristine 50-Claim Fresh Benchmark (data/fresh_claim_benchmark/test_fresh_claims.csv, N=1,110)",
        "ensemble_model": winning_label,
        "temperature_T_applied": round(T_star, 4),
        "frozen_thresholds_applied": {
            "theta_f1_star": theta_f1_star,
            "theta_high_precision": theta_accept
        },
        "disaggregated_auroc": {
            "macro_combined": round(auc_fresh_macro, 4) if auc_fresh_macro else None,
            "stratum_1_short_human": round(auc_fresh_short, 4) if auc_fresh_short else None,
            "stratum_2_long_llm": round(auc_fresh_long, 4) if auc_fresh_long else None,
            "stratum_3_stance_separation": round(auc_fresh_stance, 4) if auc_fresh_stance else None
        },
        "adversarial_robustness": {
            "contradictions_caught": f"{fresh_contra_caught}/{n_fresh_contra} ({fresh_contra_caught/max(1, n_fresh_contra)*100:.1f}%)"
        },
        "performance_at_frozen_theta_f1": {
            "precision": round(float(pf_f1), 4),
            "recall": round(float(rf_f1), 4),
            "f1": round(float(ff_f1), 4)
        }
    }
    fresh_report_file = resolve(os.path.join(output_dir, "final_ensemble_fresh_claims_report.json"))
    save_json(fresh_report, fresh_report_file)

    # -------------------------------------------------------------------------
    # PHASE 6: PUBLICATION-READY MARKDOWN SUMMARY
    # -------------------------------------------------------------------------
    summary_md = f"""# Final 3-Seed Calibrated Ensemble Report: Model Selection, Temperature Calibration, and Clean Benchmark Evaluation

## 1. Validation-Only Model Selection Decision
- **Protocol:** Strict evaluation on disjoint validation set (`val.csv`, $N=800$, 0% test overlap).
- **Candidate Architectures:**
  * **Data-Only:** Random Head + Balanced Multi-Genre Data
  * **Warm-Head:** Warm MNLI Head + Balanced Multi-Genre Data
- **Validation Ensemble AUROC:**
  * **Data-Only:** `{val_data_ens_auc:.4f}` (Seeds Mean: `{np.mean(val_data_aucs):.4f} ± {np.std(val_data_aucs):.4f}`)
  * **Warm-Head:** `{val_warm_ens_auc:.4f}` (Seeds Mean: `{np.mean(val_warm_aucs):.4f} ± {np.std(val_warm_aucs):.4f}`)
- **Official Selection:** **`{winning_label}`**
- **Methodological Rationale:** {rationale}

---

## 2. Temperature Scaling & Calibration (Tuned Strictly on Validation)
- **Optimal Temperature Parameter:** $T^* = {T_star:.4f}$ (optimized via scalar Brent minimization of NLL)
- **Validation Calibration Metrics:**
  * **Negative Log-Likelihood (NLL):** `{uncal_nll:.4f}` $\\to$ `{cal_nll:.4f}`
  * **Expected Calibration Error (ECE, 10 bins):** `{uncal_ece*100:.2f}%` $\\to$ `{cal_ece*100:.2f}%`
  * **Brier Score:** `{uncal_brier:.4f}` $\\to$ `{cal_brier:.4f}`

---

## 3. Frozen Operating Thresholds & 3-Tier Triage Policy
> *These thresholds were calibrated once on the disjoint validation set and permanently frozen prior to test evaluation.*

| Parameter | Frozen Value | Meaning |
|---|---|---|
| **$\\theta_{{F1}}^*$ (Optimal $F_1$)** | `{theta_f1_star:.2f}` | Maximizes binary classification $F_1$ score on validation data. |
| **$\\theta_{{\\text{{prec90}}}}^*$ (High Precision)** | `{theta_accept:.2f}` | Minimum threshold achieving $\\ge 90\\%$ precision on validation data. |
| **Tier 1: Accept** | $P(\\text{{full}}) \\ge {theta_accept:.2f}$ | High confidence entailment; accepted without manual review. |
| **Tier 2: Review** | ${theta_reject:.2f} \\le P(\\text{{full}}) < {theta_accept:.2f}$ | Borderline / partial coverage; flagged for active human verification. |
| **Tier 3: Reject** | $P(\\text{{full}}) < {theta_reject:.2f}$ | Confident non-entailment or contradiction; discarded. |

---

## 4. Locked 1,118 Test Benchmark Evaluation (Pristine, Zero Peeking)
- **Overall Macro AUROC:** `{test_strata['overall_macro_auroc']:.4f}`
- **Stratum 1 (Short Human) AUROC:** `{test_strata['stratum_1_short_human_auroc']:.4f}`
- **Stratum 2 (Long LLM) AUROC:** `{test_strata['stratum_2_long_llm_auroc']:.4f}`
- **Stratum 3 (Stance Separation) AUROC:** `{test_strata['stratum_3_stance_separation_auroc']:.4f}`
- **Contradiction Detection:** `{contra_caught}/{n_opp}` (`{contra_caught/max(1, n_opp)*100:.1f}%`)
- **Performance at Frozen $\\theta_{{F1}}^* = {theta_f1_star:.2f}$:**
  * **Precision:** `{p_f1*100:.2f}%`
  * **Recall:** `{r_f1*100:.2f}%`
  * **$F_1$ Score:** `{f_f1:.4f}`
- **Performance at Frozen $\\theta_{{\\text{{prec90}}}}^* = {theta_accept:.2f}$:**
  * **Precision:** `{p_prec*100:.2f}%`
  * **Recall:** `{r_prec*100:.2f}%`
  * **$F_1$ Score:** `{f_prec:.4f}`
- **3-Tier Triage Distribution:**
  * **Accept Tier:** `{n_accept}/{len(df_test)}` (`{n_accept/len(df_test)*100:.1f}%`) with **{accept_precision*100:.1f}% Precision**
  * **Review Tier:** `{n_review}/{len(df_test)}` (`{n_review/len(df_test)*100:.1f}%`)
  * **Reject Tier:** `{n_reject}/{len(df_test)}` (`{n_reject/len(df_test)*100:.1f}%`)

---

## 5. Pristine Fresh Claims Holdout Benchmark (50 Claims, N={len(df_fresh)})
- **Overall Macro AUROC:** `{f"{auc_fresh_macro:.4f}" if auc_fresh_macro is not None else "N/A"}`
- **Short Human Stratum AUROC:** `{f"{auc_fresh_short:.4f}" if auc_fresh_short is not None else "N/A"}`
- **Balanced Long LLM Stratum AUROC:** `{f"{auc_fresh_long:.4f}" if auc_fresh_long is not None else "N/A (Natural Human Holdout)"}`
- **Stance Separation AUROC:** `{f"{auc_fresh_stance:.4f}" if auc_fresh_stance is not None else "N/A"}`
- **Contradictions Caught:** `{fresh_contra_caught}/{n_fresh_contra}` (`{fresh_contra_caught/max(1, n_fresh_contra)*100:.1f}%`)
- **Performance at Frozen $\\theta_{{F1}}^*$:**
  * **Precision:** `{pf_f1*100:.2f}%` | **Recall:** `{rf_f1*100:.2f}%` | **$F_1$ Score:** `{ff_f1:.4f}`
"""

    summary_file = resolve(os.path.join(output_dir, "final_ensemble_summary.md"))
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_md)
    logger.info(f"Saved complete publication-ready ensemble report to {summary_file}")

    print("\n" + "=" * 120)
    print("FINAL 3-SEED CALIBRATED ENSEMBLE SUMMARY")
    print("=" * 120)
    print(f"Official Selected Architecture: {winning_label}")
    print(f"Optimal Temperature (Validation NLL): T* = {T_star:.4f}")
    print(f"Locked Operating Thresholds:          theta_F1* = {theta_f1_star:.2f} | theta_prec90* = {theta_accept:.2f}")
    print(f"Locked 1,118 Test Set Performance:    AUROC = {test_strata['overall_macro_auroc']:.4f} | F1 = {f_f1:.4f} | Precision = {p_f1*100:.2f}%")
    print(f"Pristine 50-Claim Fresh Benchmark:   AUROC = {auc_fresh_macro:.4f} | F1 = {ff_f1:.4f} | Precision = {pf_f1*100:.2f}%")
    print("=" * 120 + "\n")

    return {
        "decision": decision_record,
        "thresholds": thresholds_record,
        "test_benchmark": test_report,
        "fresh_benchmark": fresh_report
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-seed ensemble training, calibration, and benchmark evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mock_run", action="store_true", help="Run simulated dry-run for testing")
    parser.add_argument("--output_dir", type=str, default="results/metrics")

    args = parser.parse_args()
    run_ensemble_selection_calibration_and_evaluation(
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mock_run=args.mock_run,
        output_dir=args.output_dir
    )
