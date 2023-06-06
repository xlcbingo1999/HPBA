import torch
import time
import zerorpc
import sys

device_0 = torch.device('cuda:1')  # 使用GPU
dtype = torch.float
ip = "172.18.162.4"
port = 11111
try:
    print("start second used")
    tensor_size = (10000, 10000)
    tensor = torch.ones(tensor_size, device=device_0, dtype=dtype)
    num_copies = 18  # 复制的次数（根据需要修改）
    tensors = [tensor.clone() for _ in range(num_copies)]
    time.sleep(2)
    dispatcher_client = zerorpc.Client()
    dispatcher_client.connect("tcp://{}:{}".format(ip, port))
    dispatcher_client.handle_finished("test_2")
except Exception as e:
    dispatcher_client = zerorpc.Client()
    dispatcher_client.connect("tcp://{}:{}".format(ip, port))
    dispatcher_client.handle_error(f"test_2: {e}")
