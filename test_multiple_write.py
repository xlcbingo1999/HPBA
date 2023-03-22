import fcntl
import json
import time

significance_trace_path = "/home/netlab/DL_lab/opacus_testbed/test_multiple_write.json"
significance_trace = {}
with open(significance_trace_path, "r+") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    significance_trace = json.load(f)
    print(significance_trace)
    fcntl.flock(f, fcntl.LOCK_UN)

with open(significance_trace_path, "w+") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    significance_trace["a"] = 8
    json.dump(significance_trace, f)
    fcntl.flock(f, fcntl.LOCK_UN)