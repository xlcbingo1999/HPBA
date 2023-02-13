import zerorpc
import time
from utils.global_variable import SCHE_IP, SCHE_PORT, SUB_TRAIN_DATASET_CONFIG_PATH, TEST_DATASET_CONFIG_PATH, INIT_WORKERIDENTIFIERS
import threading


# class Dispatcher(object):
#     def __init__(self, jobs_list) -> None:
#         self.jobs_list = jobs_list
#         self.all_finished = False
    
        
def dispatch_jobs(sched_ip, sched_port, global_job_id):
    jobs_list = [{
        'target_func': 'opacus_split_review',
        'model_name': 'FF-split',
        'dataset_name': 'Home_and_Kitchen',
        'label_type': 'sentiment',
        'select_num': 1,
        'early_stopping': None,
        'device': 0,
        'LR': 1e-3,
        'EPSILON': 5.0,
        'EPOCH_SET_EPSILON': False,
        'DELTA': 1e-5,
        'MAX_GRAD_NORM': 1.2,
        'BATCH_SIZE': 128,
        'MAX_PHYSICAL_BATCH_SIZE': 128/4,
        'EPOCHS': 4,
        'train_configs': {
            'hidden_size': [150, 110],
            'embedding_size': 100,
            'sequence_length': 50,
        },
    # }, {
    #     'target_func': 'opacus_split_review',
    #     'model_name': 'LSTM-split',
    #     'dataset_name': 'Home_and_Kitchen',
    #     'label_type': 'sentiment',
    #     'select_num': 1,
    #     'early_stopping': None,
    #     'device': 0,
    #     'LR': 1e-3,
    #     'EPSILON': 5.0,
    #     'EPOCH_SET_EPSILON': False,
    #     'DELTA': 1e-5,
    #     'MAX_GRAD_NORM': 1.2,
    #     'BATCH_SIZE': 128,
    #     'MAX_PHYSICAL_BATCH_SIZE': 128/4,
    #     'EPOCHS': 4,
    #     'train_configs': {
    #         'n_layer': 2,
    #         'hidden_size': 40,
    #         'embedding_size': 100,
    #         'sequence_length': 50,
    #     },
    # }, {
    #     'target_func': 'opacus_split_review',
    #     'model_name': 'LSTM-split',
    #     'dataset_name': 'Home_and_Kitchen',
    #     'label_type': 'sentiment',
    #     'select_num': 1,
    #     'early_stopping': None,
    #     'device': 0,
    #     'LR': 1e-3,
    #     'EPSILON': 5.0,
    #     'EPOCH_SET_EPSILON': False,
    #     'DELTA': 1e-5,
    #     'MAX_GRAD_NORM': 1.2,
    #     'BATCH_SIZE': 128,
    #     'MAX_PHYSICAL_BATCH_SIZE': 128/4,
    #     'EPOCHS': 4,
    #     'train_configs': {
    #         'n_layer': 2,
    #         'hidden_size': 40,
    #         'embedding_size': 100,
    #         'sequence_length': 50,
    #     },
    }, {
        'target_func': 'opacus_split_review',
        'model_name': 'FF-split',
        'dataset_name': 'Home_and_Kitchen',
        'label_type': 'sentiment',
        'select_num': 1,
        'early_stopping': None,
        'device': 0,
        'LR': 1e-3,
        'EPSILON': 5.0,
        'EPOCH_SET_EPSILON': False,
        'DELTA': 1e-5,
        'MAX_GRAD_NORM': 1.2,
        'BATCH_SIZE': 128,
        'MAX_PHYSICAL_BATCH_SIZE': 128/4,
        'EPOCHS': 4,
        'train_configs': {
            'hidden_size': [150, 110],
            'embedding_size': 100,
            'sequence_length': 50,
        },
    }, {
        'target_func': 'opacus_split_review',
        'model_name': 'FF-split',
        'dataset_name': 'Home_and_Kitchen',
        'label_type': 'sentiment',
        'select_num': 1,
        'early_stopping': None,
        'device': 0,
        'LR': 1e-3,
        'EPSILON': 5.0,
        'EPOCH_SET_EPSILON': False,
        'DELTA': 1e-5,
        'MAX_GRAD_NORM': 1.2,
        'BATCH_SIZE': 128,
        'MAX_PHYSICAL_BATCH_SIZE': 128/4,
        'EPOCHS': 4,
        'train_configs': {
            'hidden_size': [150, 110],
            'embedding_size': 100,
            'sequence_length': 50,
        },
    }]
    jobs_id_list = [x for x in range(global_job_id, global_job_id+len(jobs_list))]
    next_job_id = jobs_id_list[-1] + 1 if len(jobs_id_list) > 0 else global_job_id
    client = get_zerorpc_client(sched_ip, sched_port)
    jobs_detail = list(map(lambda x: [x[0], x[1]], zip(jobs_id_list, jobs_list)))
    client.add_jobs(jobs_detail)
    return next_job_id

def sched_dispatch(sched_ip, sched_port):
    client = get_zerorpc_client(sched_ip, sched_port)
    client.sched_dispatch()

def sched_update_dataset(sched_ip, sched_port):
    sub_train_dataset_config_path = SUB_TRAIN_DATASET_CONFIG_PATH
    test_dataset_config_path = TEST_DATASET_CONFIG_PATH
    init_datasetidentifiers = {
        "Home_and_Kitchen": [
            "train_sub_0",
            "train_sub_1"
        ]
    }
    
    init_datasetidentifier_2_epsilon_capacity = {
        "Home_and_Kitchen": {
            "train_sub_0": 500.0,
            "train_sub_1": 500.0
        }
    }

    client = get_zerorpc_client(sched_ip, sched_port, timeout=60)
    client.update_dataset(init_datasetidentifiers, sub_train_dataset_config_path, init_datasetidentifier_2_epsilon_capacity, test_dataset_config_path)

def sched_update_gpu_status(sched_ip, sched_port, init_gpuidentifiers, sleep_time):
    def thread_func_timely_update(sched_ip, sched_port, init_gpuidentifiers):
        while True:
            client = get_zerorpc_client(sched_ip, sched_port)
            client.update_gpu(init_gpuidentifiers)
            time.sleep(sleep_time)
    
    p = threading.Thread(target=thread_func_timely_update, args=(sched_ip, sched_port, init_gpuidentifiers), daemon=True)
    p.start()
    return p

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
    init_gpuidentifiers = INIT_WORKERIDENTIFIERS
    sleep_time = 10
    
    # client = zerorpc.Client()
    # client.connect("tcp://{}:{}".format(sched_ip, sched_port))

    origin_time = time.time()
    temp_time = time.time()

    p = None
    try:
        p = sched_update_gpu_status(sched_ip, sched_port, init_gpuidentifiers, sleep_time)
        sched_update_dataset(sched_ip, sched_port)
        # sched_clear_all_jobs(sched_ip, sched_port)
        global_job_id = dispatch_jobs(sched_ip, sched_port, global_job_id)
        sched_dispatch(sched_ip, sched_port)
    except Exception as e:
        p.join()
    finally:
        p.join()
    # global_job_id = global_job_id + 1

    # while True:
    #     if time.time() - origin_time >= 4:
    #         print("over")
    #         break
    #     if time.time() - temp_time >= 2:
    # print("sched_dispatch")