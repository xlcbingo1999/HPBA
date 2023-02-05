import pynvml
import distutils
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import time
import argparse

def get_df_config():
    parser = argparse.ArgumentParser(
                description='Sweep through lambda values')
    parser.add_argument('--local_ip', type=str, default="172.18.162.3")
    args = parser.parse_args()
    return args

def get_current_gpu_status(device_id):
    pynvml.nvmlInit() # 初始化
    target_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    typeinfo = pynvml.nvmlDeviceGetName(target_handle)
    uuid = pynvml.nvmlDeviceGetUUID(target_handle)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(target_handle)
    used = meminfo.used / (1024 ** 2.)
    free = meminfo.free / (1024 ** 2.)
    total = meminfo.total / (1024 ** 2.)
    pynvml.nvmlShutdown() # 最后要关闭管理工具
    results = {
        "type": bytes.decode(typeinfo),
        "UUID": bytes.decode(uuid),
        "server_id": device_id,
        "used_mem": used,
        "free_mem": free,
        "total_mem": total
    }
    return results

def write_gpu_status(args):
    local_ip = args[0]
    device_id = args[1]
    gpu_status_path = "/mnt/linuxidc_client/gpu_states/{}-{}.json".format(local_ip, device_id)
    current_gpu_status = get_current_gpu_status(device_id)
    
    with open(gpu_status_path, 'w') as f:
        json.dump(current_gpu_status, f)

def read_scheduler_status():
    scheduler_status_path = "/mnt/linuxidc_client/gpu_states/scheduler_status.json"
    scheduler_status = {}
    with open(scheduler_status_path, 'r') as f:
        scheduler_status = json.load(f)
    return scheduler_status

def timely_update_gpu_status(local_ip, device_ids):
    update_time = 0.5
    
    while True:
        scheduler_status = read_scheduler_status()
        if not scheduler_status['scheduler_status']:
            return
        write_args = [
            (local_ip, device_ids[index]) for index in range(len(device_ids))
        ]
        with ThreadPoolExecutor(max_workers=len(write_args)) as pool:
            pool.map(write_gpu_status, write_args)
            
        time.sleep(update_time)


if __name__ == "__main__":
    args = get_df_config()
    local_ip = args.local_ip
    device_ids = [0, 1, 2, 3]
    print("begin timely update GPU status...")
    print("local_ip: {}".format(local_ip))
    timely_update_gpu_status(local_ip, device_ids)