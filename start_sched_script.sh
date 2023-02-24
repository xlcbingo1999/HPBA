mkdir ./log_20230214/
git fetch
git pull
nohup python DL_sched.py > ./log_20230214/DL_sched_02141017.log 2>&1 &
