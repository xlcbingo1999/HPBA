import torch
import time
import zerorpc
import sys

device_0 = torch.device('cuda:0')  # 使用GPU
dtype = torch.float
ip = "172.18.162.4"
port = 16042
try:
    print("start test_1 used")
    # tensor_size = (10000, 10000)
    # tensor = torch.ones(tensor_size, device=device_0, dtype=dtype)
    # num_copies = 18  # 复制的次数（根据需要修改）
    # tensors = [tensor.clone() for _ in range(num_copies)]
    time.sleep(2.5)
    print("finished test_1 used")
    dispatcher_client = zerorpc.Client()
    dispatcher_client.connect("tcp://{}:{}".format(ip, port))
    re = dispatcher_client.handle_finished("test_1")
    print("re: ", re)
except Exception as e:
    dispatcher_client = zerorpc.Client()
    dispatcher_client.connect("tcp://{}:{}".format(ip, port))
    dispatcher_client.handle_error(f"test_1: {e}")
finally:
    sys.exit(0)