import tarfile
import os
import glob

archive_name = "hpc_sync_package.tar.gz"
print(f"Creating comprehensive HPC sync package: {archive_name}...")

# Directories and files to sync to HPC
sync_targets = [
    # Core configuration & root runner
    "configs/config.yaml",
    "run_pipeline.py",
    "run_hpc_loop.sh",
    "run_secondary_models.sh",
    "run_olmo2_matrix.sh",
    "run_unseen_models.sh",
    "train_v3_on_hpc.sh",
    "run_jury_hpc.sh",
    ".env",
    "run_167_benchmark.py",
    "run_ensemble_benchmark.py",
    "run_twostage_ensemble_benchmark.py",
    "run_calibrated_twostage_benchmark.py",
    "run_sentence_max_pooling_benchmark.py",
    "run_claim_split_calibration.py",
    "run_single_pair_audit.py",
    "paper_draft.tex",
    "paper_draft.md",
    
    # Complete source code tree
    "src",
    "scripts",
    
    # Complete test suite
    "tests",
    
    # Calibration datasets & benchmarks
    "data/calibration",
    "data/audit",
    "data/processed/perspectives.csv",
    "data/nli_training_v2",
    "data/nli_training_v3",
    "compute_llm_stage1_recall.py",
    "compute_paired_bootstrap.py",
    
    # Paper tables & results summaries
    "results/paper_tables",
    "results/calibration",
]

count = 0
total_bytes = 0

with tarfile.open(archive_name, "w:gz") as tar:
    for target in sync_targets:
        if os.path.exists(target):
            if os.path.isdir(target):
                # Walk directory, excluding __pycache__ and checkpoint files
                for root, dirs, files in os.walk(target):
                    dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
                    for f in files:
                        if f.endswith(".pyc") or f.endswith(".pyo"):
                            continue
                        fpath = os.path.join(root, f)
                        arcname = os.path.relpath(fpath, ".").replace("\\", "/")
                        tar.add(fpath, arcname=arcname)
                        size = os.path.getsize(fpath)
                        total_bytes += size
                        count += 1
                        print(f"  + Added: {arcname} ({size / 1024:.1f} KB)")
            else:
                arcname = target.replace("\\", "/")
                tar.add(target, arcname=arcname)
                size = os.path.getsize(target)
                total_bytes += size
                count += 1
                print(f"  + Added: {arcname} ({size / 1024:.1f} KB)")
        else:
            print(f"  - Skipped (not found): {target}")

total_mb = os.path.getsize(archive_name) / (1024 * 1024)
print(f"\n[SUCCESS] Archive created: {archive_name} with {count} files ({total_mb:.2f} MB compressed).")
print("Ready to transfer to HPC via scp or sftp.")

