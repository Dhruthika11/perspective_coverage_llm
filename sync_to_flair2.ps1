# PowerShell script to sync new V5 extended experiment files to Flair2 GPU cluster
$REMOTE_HOST = "srujan.sai@Flair2"
$REMOTE_DIR = "~/perspective_coverage"

Write-Host "Syncing V5 experiment scripts to ${REMOTE_HOST}:${REMOTE_DIR}..." -ForegroundColor Cyan

# 1. Training & data preparation scripts
Write-Host "--> Uploading training & prep scripts..." -ForegroundColor Yellow
scp src/data_prep/prepare_multimodel_processed_generations.py "${REMOTE_HOST}:${REMOTE_DIR}/src/data_prep/"
scp src/training/train_v5_ablations.py "${REMOTE_HOST}:${REMOTE_DIR}/src/training/"
scp src/training/run_multiseed_ablations.py "${REMOTE_HOST}:${REMOTE_DIR}/src/training/"
scp src/training/train_and_calibrate_ensemble.py "${REMOTE_HOST}:${REMOTE_DIR}/src/training/"
scp src/training/train_leave_one_family_out.py "${REMOTE_HOST}:${REMOTE_DIR}/src/training/"
scp src/training/prepare_balanced_logfo_benchmark.py "${REMOTE_HOST}:${REMOTE_DIR}/src/training/"
scp src/training/prepare_fresh_claim_benchmark.py "${REMOTE_HOST}:${REMOTE_DIR}/src/training/"
scp src/training/prepare_mistral_benchmark.py "${REMOTE_HOST}:${REMOTE_DIR}/src/training/"
scp src/training/prepare_clean_logfo_benchmark.py "${REMOTE_HOST}:${REMOTE_DIR}/src/training/"

# 2. Analysis & evaluation scripts
Write-Host "--> Uploading analysis & embedding scripts..." -ForegroundColor Yellow
scp src/embeddings/embed_model.py "${REMOTE_HOST}:${REMOTE_DIR}/src/embeddings/"
scp src/analysis/run_in_family_two_stage_coverage.py "${REMOTE_HOST}:${REMOTE_DIR}/src/analysis/"
scp src/analysis/leak_check_audit.py "${REMOTE_HOST}:${REMOTE_DIR}/src/analysis/"
scp src/analysis/calibrate_and_bootstrap_v5.py "${REMOTE_HOST}:${REMOTE_DIR}/src/analysis/"
scp src/analysis/audit_v5_false_positives.py "${REMOTE_HOST}:${REMOTE_DIR}/src/analysis/"
scp src/analysis/evaluate_surface_and_zeroshot_baselines.py "${REMOTE_HOST}:${REMOTE_DIR}/src/analysis/"
scp src/analysis/evaluate_family_specific_zeroshot_and_logfo.py "${REMOTE_HOST}:${REMOTE_DIR}/src/analysis/"
scp src/analysis/evaluate_rebalanced_logfo_and_zeroshot.py "${REMOTE_HOST}:${REMOTE_DIR}/src/analysis/"
scp src/analysis/evaluate_zeroshot_per_family.py "${REMOTE_HOST}:${REMOTE_DIR}/src/analysis/"
scp src/analysis/evaluate_v5_disaggregated_benchmark.py "${REMOTE_HOST}:${REMOTE_DIR}/src/analysis/"
scp src/analysis/evaluate_v5_on_mistral.py "${REMOTE_HOST}:${REMOTE_DIR}/src/analysis/"

# 3. Shell runner scripts
Write-Host "--> Uploading shell runners..." -ForegroundColor Yellow
scp scripts/run_v5_in_family_coverage_on_hpc.sh "${REMOTE_HOST}:${REMOTE_DIR}/scripts/"
scp scripts/run_mistral_evaluation_on_hpc.sh "${REMOTE_HOST}:${REMOTE_DIR}/scripts/"
scp scripts/run_clean_logfo_evaluation_on_hpc.sh "${REMOTE_HOST}:${REMOTE_DIR}/scripts/"
ssh $REMOTE_HOST "chmod +x ${REMOTE_DIR}/scripts/*.sh"

# 4. Test suites
Write-Host "--> Uploading test suite..." -ForegroundColor Yellow
scp tests/test_v5_extended_experiments.py "${REMOTE_HOST}:${REMOTE_DIR}/tests/"

# 5. Fresh claims, Rebalanced LOGFO, and Mistral benchmark datasets
Write-Host "--> Uploading benchmark datasets..." -ForegroundColor Yellow
ssh $REMOTE_HOST "mkdir -p ${REMOTE_DIR}/data/fresh_claim_benchmark ${REMOTE_DIR}/data/logfo_benchmark ${REMOTE_DIR}/data/mistral_benchmark"
scp data/fresh_claim_benchmark/test_fresh_claims.csv "${REMOTE_HOST}:${REMOTE_DIR}/data/fresh_claim_benchmark/"
scp data/fresh_claim_benchmark/fresh_claims_manifest.json "${REMOTE_HOST}:${REMOTE_DIR}/data/fresh_claim_benchmark/"
scp data/logfo_benchmark/balanced_multifamily_benchmark.csv "${REMOTE_HOST}:${REMOTE_DIR}/data/logfo_benchmark/"
scp data/logfo_benchmark/balanced_multifamily_manifest.json "${REMOTE_HOST}:${REMOTE_DIR}/data/logfo_benchmark/"
scp data/logfo_benchmark/clean_balanced_logfo_benchmark.csv "${REMOTE_HOST}:${REMOTE_DIR}/data/logfo_benchmark/"
scp data/logfo_benchmark/clean_balanced_logfo_manifest.json "${REMOTE_HOST}:${REMOTE_DIR}/data/logfo_benchmark/"
scp data/mistral_benchmark/mistral_adjudicated_benchmark.csv "${REMOTE_HOST}:${REMOTE_DIR}/data/mistral_benchmark/"
scp data/mistral_benchmark/mistral_benchmark_manifest.json "${REMOTE_HOST}:${REMOTE_DIR}/data/mistral_benchmark/"

Write-Host "Sync complete! All V5 extended scripts are updated on Flair2." -ForegroundColor Green
