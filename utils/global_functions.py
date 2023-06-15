from enum import Enum
import json
import numpy as np
import zerorpc
from contextlib import contextmanager

def normal_counter(origin_counter):
    sum_value = sum(origin_counter.values(), 0)
    new_counter = {}
    for key in origin_counter:
        key = str(key)
        new_counter[key] = origin_counter[key] / sum_value
    return new_counter

def add_2_map(A, B):
    C = {}
    for key in list(set(A) | set(B)):
        if A.get(key) and B.get(key):
            C.update({key: A.get(key) + B.get(key)})
        else:
            C.update({key: A.get(key) or B.get(key)})
    return C

def get_types(var):
    var_type = type(var).__name__
    if var_type in ('list', 'tuple', 'set'):
        sub_types = set()
        for sub_var in var:
            sub_types.add(get_types(sub_var))
        return var_type + '[' + ', '.join(sub_types) + ']'
    elif var_type == 'dict':
        sub_types = set()
        for key, value in var.items():
            sub_types.add(get_types(key) + ': ' + get_types(value))
        return var_type + '[' + ', '.join(sub_types) + ']'
    else:
        return var_type

def convert_types(var):
    if isinstance(var, np.int64):
        return int(var)
    elif isinstance(var, np.float64):
        return float(var)
    elif isinstance(var, (list, tuple)):
        return [convert_types(sub_var) for sub_var in var]
    elif isinstance(var, dict):
        return {convert_types(key): convert_types(value) for key, value in var.items()}
    else:
        return var

def print_console_file(content, fileHandler=None):
    if fileHandler is not None:
        print(content, file=fileHandler)
    print(content)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

@contextmanager
def get_zerorpc_client(ip, port, timeout=500, heartbeat=None):
    try:
        tcp_ip_port = "tcp://{}:{}".format(ip, port)
        client = zerorpc.Client(timeout=timeout, heartbeat=heartbeat)
        client.connect(tcp_ip_port)
        yield client
    except Exception as e:
        import traceback
        raise Exception(traceback.format_exc())
    finally:
        client.close()

class FAILED_RESULT_KEY(Enum):
    WORKER_NO_READY = 1
    JOB_FAILED = 2
    JOB_TYPE_ERROR = 3
    SCHED_FAILED = 4
    TRY_MAX_TIME = 5
    RUNNING_FAILED = 6

class JOB_STATUS_KEY(Enum):
    # DONE_GPU_SCHED = 102
    # DONE_DATASET_SCHED = 103
    NO_SCHE = 101
    DONE_ALL_SCHED = 104
    RUNNING = 105
    DONE_SIGNIFICANCE_CAL = 107

    WAITING = 201

    SIMULATION_NO_SUMBIT = 301

    FINISHED = 888
    FAILED = 999


    

class JOB_STATUS_UPDATE_PATH(Enum):
    # NOSCHED_2_GPUSCHED = 0
    # NOSCHED_2_DATASETSCHED = 1
    NOSCHED_2_SIGNIFICANCE = 2
    SIGNIFICANCE_2_ALLSCHED = 3

    ALLSCHED_2_RECOMING = 101
    RUNNING_2_RECOMING = 102
    SIGNIFICANCE_2_RECOMING = 103
    NOSCHED_2_RECOMING = 104

    NOSCHED_2_FINISHED = 201
    SIGNIFICANCE_2_FINISHED = 202
    ALLSCHED_2_FINISHED = 203
    RUNNING_2_FINISHED = 204
    
    WAITING_2_ALLSCHED = 301
    SIMULATION_NOSUBMIT_2_NOSHCED = 401

    NOSCHED_2_FAILED = 991
    ALLSCHED_2_FAILED = 992
    SIGNIFICANCE_2_FAILED = 993
    RUNNING_2_FAILED = 994
    WAITING_2_FAILED = 995

    # GPUSCHED_2_ALLSCHED = 3
    # DATASETSCHED_2_ALLSCHED = 4

class DATASET_STATUS_KEY(Enum):
    NO_SUBMIT = 0
    SUBMITED = 1
    EXHAUST = 2

class EVENT_KEY(Enum):
    JOB_SUBMIT = 100
    JOB_SUCCESS = 101
    JOB_FAILED = 102
    JOB_SIGCAL_SCHED_RUNNING = 103

    WORKER_ADD = 200
    WORKER_REMOVE = 201

    DATABLOCK_ADD = 300
    DATABLOCK_REMOVE = 301

    HISTORY_JOB_SUBMIT = 400

    TEST_START = 500
    MAX_TIME = 100000
    