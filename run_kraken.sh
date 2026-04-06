#!/bin/bash
# SOVEREIGN KRAKEN 8 IGNITION SCRIPT (V4.2)
# =========================================
cd /var/www/html/ML/kat
pkill -9 -f train.py
pkill -9 -f auto_run.py
sleep 1
rm -f kraken_mission.log
nohup env PYTHONUNBUFFERED=1 ./venv/bin/python auto_run.py train --model hydra --epochs 300 --batch 32 --candles 30000 --timeframe 1m > kraken_mission.log 2>&1 &
echo "🚀 SOVEREIGN KRAKEN 8: IGNITION SUCCESSFUL."
