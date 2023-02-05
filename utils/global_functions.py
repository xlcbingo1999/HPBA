from enum import Enum


def normal_counter(origin_counter):
    sum_value = sum(origin_counter.values(), 0)
    new_counter = {}
    for key in origin_counter:
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
    DONE_SCHED = 1
    RUNNING = 2
    FINISHED = 3
    FAILED = 4