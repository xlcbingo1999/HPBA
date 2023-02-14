#!/bin/bash
mkdir ./log_20230214/
git fetch
git pull
nohup python DL_worker.py > ./log_20230214/DL_worker_02141017.log 2>&1 &