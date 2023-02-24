mkdir ./log_20230214/
git fetch
git pull
sleep 4
nohup python DL_dispatcher.py > ./log_20230214/DL_dispatcher_02141017.log 2>&1 &
