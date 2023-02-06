import zerorpc
import time
from utils.global_variable import SCHE_IP, SCHE_PORT, MAX_EPSILON

def dispatch_jobs(sched_ip, sched_port, global_job_id):
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
    client = get_zerorpc_client(sched_ip, sched_port)
    client.add_jobs(jobs_detail)

def sched_dispatch(sched_ip, sched_port):
    client = get_zerorpc_client(sched_ip, sched_port)
    client.sched_dispatch()

def sched_initial_all_workers_dataset(sched_ip, sched_port, keep_origin_dataset):
    fetch_dataset_origin_info = {
        "DATASET_NAME": "MIX3_sentiment",
        "LABEL_TYPE": "sentiment",
        "VALID_SIZE": 0.1,
        "SEQUENCE_LENGTH": 50,
        "SPLIT_NUM": 4,
        "same_capacity": True
    }
    client = get_zerorpc_client(sched_ip, sched_port, timeout=60)
    client.initial_sched_and_all_workers_dataset(fetch_dataset_origin_info, keep_origin_dataset)

def get_zerorpc_client(ip, port, timeout=30):
    tcp_ip_port = "tcp://{}:{}".format(ip, port)
    client = zerorpc.Client(timeout=timeout)
    client.connect(tcp_ip_port)
    return client

def sched_clear_all_jobs(ip, port):
    client = get_zerorpc_client(ip, port)
    client.clear_all_jobs()


if __name__ == '__main__':
    global_job_id = 0
    sched_ip = SCHE_IP
    sched_port = SCHE_PORT
    
    # client = zerorpc.Client()
    # client.connect("tcp://{}:{}".format(sched_ip, sched_port))

    origin_time = time.time()
    temp_time = time.time()

    keep_origin_dataset = False
    sched_initial_all_workers_dataset(sched_ip, sched_port, keep_origin_dataset)
    
    # while True:
    #     if time.time() - origin_time >= 4:
    #         print("over")
    #         break
    #     if time.time() - temp_time >= 2:
    # print("sched_dispatch")
    # sched_clear_all_jobs(sched_ip, sched_port)
    dispatch_jobs(sched_ip, sched_port, global_job_id)
    sched_dispatch(sched_ip, sched_port)
    # global_job_id = global_job_id + 1

    
    
