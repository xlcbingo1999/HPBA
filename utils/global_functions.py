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
    NO_SCHE = 0
    DONE_GPU_SCHED = 1 # 无效
    DONE_DATASET_SCHED = 2 # 无效
    DONE_ALL_SCHED = 3
    RUNNING = 4
    FINISHED = 5
    FAILED = 999
    DONE_SIGINIFICANCE_CAL = 7

class JOB_STATUS_UPDATE_PATH(Enum):
    NOSCHED_2_GPUSCHED = 0
    NOSCHED_2_DATASETSCHED = 1
    NOSCHED_2_SIGNIFICANCE = 2
    SIGNIFICANCE_2_ALLSCHED = 3

    NOSCHED_2_FAILED = 997
    ALLSCHED_2_FAILED = 998
    SIGNIFICANCE_2_FAILED = 999

    # GPUSCHED_2_ALLSCHED = 3
    # DATASETSCHED_2_ALLSCHED = 4

class DATASET_STATUS_KEY(Enum):
    NO_SUBMIT = 0
    SUBMITED = 1
    EXHAUST = 2

    