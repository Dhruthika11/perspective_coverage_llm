@echo off
echo ========================================================
echo Syncing updates to IITGN HPC Server (10.0.62.150:2022)
echo ========================================================
scp -P 2022 hpc_sync_package.tar.gz thoutam.dhruthika@10.0.62.150:/iitgn/home/thoutam.dhruthika/perspective_coverage/
echo.
echo Upload complete! Now extracting on HPC...
ssh -p 2022 thoutam.dhruthika@10.0.62.150 "cd /iitgn/home/thoutam.dhruthika/perspective_coverage ; tar -xzvf hpc_sync_package.tar.gz"
echo.
echo ========================================================
echo HPC Sync Completed Successfully!
echo ========================================================
pause
