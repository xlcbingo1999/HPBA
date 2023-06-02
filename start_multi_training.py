# while true; do python test.py --epsilon 0.5; sleep 1; done &
# while true; do python test.py --epsilon 0.5; sleep 1; done &
# while true; do python test.py --epsilon 0.5; sleep 1; done &
# while true; do python test.py --epsilon 0.5; sleep 1; done &
# while true; do python test.py --epsilon 0.5; sleep 1; done &
# while true; do python test.py --epsilon 0.5; sleep 1; done &
# while true; do python test.py --epsilon 0.5; sleep 1; done &
# while true; do python test.py --epsilon 0.5; sleep 1; done &
# while true; do python test.py --epsilon 0.5; sleep 1; done &
# while true; do python test.py --epsilon 0.5; sleep 1; done &

import os
import multiprocessing

def do_action(index, all_command):
    print(f"job {index} start!")
    os.system(all_command)
    print(f"job {index} end!")

def get_command(epsilon, model, device_index, MAX_PHYSICAL_BATCH_SIZE):
    commands_str = [
        f"temp=10",
        f"while [ $temp -ge 0 ]",
        f"do python DL_simple_train_one_job.py --epsilon {epsilon} --model {model} --device_index {device_index} --MAX_PHYSICAL_BATCH_SIZE {MAX_PHYSICAL_BATCH_SIZE}",
        f"sleep 1",
        f"temp=$(( $temp - 1 ))",
        f'echo "remain temp $temp"',
        f"done"
    ]
    all_command = "; ".join(commands_str)
    print(all_command)
    return all_command
all_num = 10

with multiprocessing.Pool(processes=all_num) as pool:
    index_arr = list(range(all_num))
    epsilon_arr = [0.5] * int(all_num/2) + [1.0] * (all_num - int(all_num/2))
    # model_arr = ["CNN"] * all_num
    # MAX_PHYSICAL_BATCH_SIZE_arr = [256] * all_num
    model_arr = ["FF"] * all_num
    MAX_PHYSICAL_BATCH_SIZE_arr = [96] * all_num
    
    device_index_arr = [2] * int(all_num/2) + [3] * (all_num - int(all_num/2)) 
    command_arr = [get_command(epsilon_arr[index], model_arr[index], device_index_arr[index], MAX_PHYSICAL_BATCH_SIZE_arr[index]) for index in range(all_num)]
    
    args_zip = zip(index_arr, command_arr)
    pool.starmap(do_action, args_zip)