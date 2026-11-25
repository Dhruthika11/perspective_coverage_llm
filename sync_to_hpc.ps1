Write-Host "Syncing updates to IITGN HPC Server (10.0.62.150:2022)..." -ForegroundColor Cyan
scp -P 2022 hpc_sync_package.tar.gz thoutam.dhruthika@10.0.62.150:/iitgn/home/thoutam.dhruthika/perspective_coverage/
Write-Host "Extracting on HPC..." -ForegroundColor Cyan
ssh -p 2022 thoutam.dhruthika@10.0.62.150 "cd /iitgn/home/thoutam.dhruthika/perspective_coverage ; rm -f '*\\*' 2>/dev/null ; tar -xzvf hpc_sync_package.tar.gz"
Write-Host "HPC Sync Complete!" -ForegroundColor Green
