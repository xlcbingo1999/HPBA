import zerorpc
import time
from utils.global_variable import SCHE_IP, SCHE_PORT, MAX_EPSILON

def dispatch_jobs(client, sched_ip, sched_port, global_job_id):
    jobs_detail = [
        [global_job_id, {
            'target_func': 'opacus_split_review',
            'model_name': 'FF-split',
            'early_stopping': None,
            'device': 0,
            'LR': 1e-3,
            'EPSILON': MAX_EPSILON,
            'EPOCH_SET_EPSILON': False,
            'DELTA': 1e-5,
            'MAX_GRAD_NORM': 1.2,
            'BATCH_SIZE': 128,
            'MAX_PHYSICAL_BATCH_SIZE': 128/4,
            'EPOCHS': 4,
        }]
    ]
    client.add_jobs(jobs_detail)

def sched_dispatch(client):
    client.sched_dispatch()

def sched_initial_all_workers_dataset(client, keep_origin_dataset):
    fetch_dataset_origin_info = {
        "DATASET_NAME": "MIX3_sentiment",
        "LABEL_TYPE": "sentiment",
        "VALID_SIZE": 0.1,
        "SEQUENCE_LENGTH": 50,
        "SPLIT_NUM": 4,
        "same_capacity": True
    }
    client.initial_all_workers_dataset(fetch_dataset_origin_info, keep_origin_dataset)

if __name__ == '__main__':
    global_job_id = 0
    sched_ip = SCHE_IP
    sched_port = SCHE_PORT
    
    client = zerorpc.Client()
    client.connect("tcp://{}:{}".format(sched_ip, sched_port))

    origin_time = time.time()
    temp_time = time.time()

    keep_origin_dataset = False
    sched_initial_all_workers_dataset(client, keep_origin_dataset) # 测试成功
    
    # while True:
    #     if time.time() - origin_time >= 4:
    #         print("over")
    #         break
    #     if time.time() - temp_time >= 2:
    print("sched_dispatch")
    dispatch_jobs(client, sched_ip, sched_port, global_job_id)
    sched_dispatch(client)
    global_job_id = global_job_id + 1

    
    
