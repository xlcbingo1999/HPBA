from enum import Enum


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


class FAILED_RESULT_KEY(Enum):
    WORKER_NO_READY = 1
    JOB_FAILED = 2
    JOB_TYPE_ERROR = 3

class JOB_STATUS_KEY(Enum):
    # DONE_GPU_SCHED = 102
    # DONE_DATASET_SCHED = 103
    NO_SCHE = 101
    DONE_ALL_SCHED = 104
    RUNNING = 105
    DONE_SIGNIFICANCE_CAL = 107

    RECOMING = 201

    FINISHED = 301
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
    
    RECOMING_2_NOSCHED = 301

    NOSCHED_2_FINISHED = 201
    SIGNIFICANCE_2_FINISHED = 202
    ALLSCHED_2_FINISHED = 203
    RUNNING_2_FINISHED = 204

    NOSCHED_2_FAILED = 991
    ALLSCHED_2_FAILED = 992
    SIGNIFICANCE_2_FAILED = 993
    RUNNING_2_FAILED = 994

    # GPUSCHED_2_ALLSCHED = 3
    # DATASETSCHED_2_ALLSCHED = 4

class DATASET_STATUS_KEY(Enum):
    NO_SUBMIT = 0
    SUBMITED = 1
    EXHAUST = 2

    