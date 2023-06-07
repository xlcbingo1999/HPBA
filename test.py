'''
import argparse
from utils.global_variable import WORKER_LOCAL_IP, WORKER_LOCAL_PORT
from significance_policies.OTDD import OTDDPolicy

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--train_dataset_name", type=str, default="EMNIST")
    parser.add_argument("--test_dataset_name", type=str, default="EMNIST_MNIST-1000_1000") # MNIST-2000, EMNIST-2000
    parser.add_argument("--sub_train_key_ids", type=str, default="train_sub_0")
    parser.add_argument("--sub_test_key_ids", type=str, default="test_sub_0")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    return args

args = get_df_config()
ot = OTDDPolicy(args.batch_size)
signficance_state = {
    "train_dataset_name": args.train_dataset_name,
    "test_dataset_name": args.test_dataset_name,
    "sub_train_key_ids": [args.sub_train_key_ids],
    "sub_test_key_ids": [args.sub_test_key_ids],
}
ot.get_job_datablock_significance_async(signficance_state, args.device_index)
'''

'''
import os
path = '/mnt/linuxidc_client/models_save'
subpath = path + '/d20230324'
file = subpath + '/d.txt'

if not os.path.exists(file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        f.write("File created successfully!")
else:
    print("File already exists!")
'''

'''
import json

data = [{'key': 1}, {'key': 2}]

with open('a.json', 'w') as f:
    json.dump(data, f)
'''

'''
import numpy as np
import random

def poisson_arrival_times(last_arrival_time, lambdas):
    # n: 总任务数
    # lambdas: 每个任务的到达率
    arrival_time = last_arrival_time + np.random.exponential(scale=1/lambdas)
    return arrival_time

jiange = 100 # 100, 500, 1000
current_lambda = 1 / jiange
last_arrival_time = 0.0
n = 50
for i in range(n):
    last_arrival_time = poisson_arrival_times(last_arrival_time, current_lambda)
    print("i: {}; time: {}".format(i, last_arrival_time))
'''

'''
import cvxpy as cp
import numpy as np

# Define problem parameters
n = 2  # Number of items
m = 3  # Number of bins
p = np.array(
    [[1, 2, 3],
     [4, 5, 6]]
)  # Probability matrix
c = np.array(
    [2, 4]
)  # Capacity of each item
x = cp.Variable((n, m), boolean=True)  # Binary variables

# Define optimization problem
# objective = cp.Maximize(cp.sum(cp.multiply(p, x) / c))
objective = cp.Maximize(cp.sum((cp.sum(cp.multiply(p, x), axis=1) / c)))
print(objective)

constraints = [
    cp.sum(x[i, :]) <= 1 for i in range(n)  # Each item can only be in one bin
]
prob = cp.Problem(objective, constraints)

# Solve optimization problem
result = prob.solve()

# Print optimal objective value and binary variables
print("Optimal objective value:", result)
print("Binary variables:\n", x.value)
'''
'''
import cvxpy as cp
import numpy as np
import time

solver = cp.ECOS

job_num, datablock_num = 250, 100
sign_matrix = np.random.random((job_num, datablock_num))
job_privacy_budget_consume_list = np.random.random_sample(size=(job_num, ))
datablock_privacy_budget_capacity_list = [10.0] * datablock_num
job_target_datablock_selected_num_list = [1] * job_num
job_privacy_budget_consume_list = np.array(job_privacy_budget_consume_list)[np.newaxis, :]
datablock_privacy_budget_capacity_list = np.array(datablock_privacy_budget_capacity_list)[np.newaxis, :]
job_target_datablock_selected_num_list = np.array(job_target_datablock_selected_num_list)

print("sign_matrix: ", sign_matrix.shape)
print("job_privacy_budget_consume_list: ", job_privacy_budget_consume_list.shape)
print("datablock_privacy_budget_capacity_list: ", datablock_privacy_budget_capacity_list.shape)
print("job_target_datablock_selected_num_list: ", job_target_datablock_selected_num_list.shape)

begin_time = time.time()
matrix_X = cp.Variable((job_num, datablock_num), nonneg=True)
print("cp.sum(matrix_X, axis=1): ", cp.sum(matrix_X, axis=1).shape)

objective = cp.Maximize(
    cp.sum(cp.multiply(sign_matrix, matrix_X))
)

constraints = [
    matrix_X >= 0,
    matrix_X <= 1,
    cp.sum(matrix_X, axis=1) <= job_target_datablock_selected_num_list,
    (job_privacy_budget_consume_list @ matrix_X) <= datablock_privacy_budget_capacity_list
]

print("check job_target_datablock_selected_num_list: {}".format(job_target_datablock_selected_num_list))
print("check datablock_privacy_budget_capacity_list: {}".format(datablock_privacy_budget_capacity_list))
print("check sum of job_privacy_budget_consume_list: {}".format(np.sum(job_privacy_budget_consume_list * job_target_datablock_selected_num_list)))

cvxprob = cp.Problem(objective, constraints)
result = cvxprob.solve(solver)
print(matrix_X.value)
if cvxprob.status != "optimal":
    print('WARNING: Allocation returned by policy not optimal!')
print("cal time: {} s".format(time.time() - begin_time))
'''

'''
import numpy as np
import time

a = np.random.random(size=(1000,))
b = np.random.choice(range(1000), size=10)

begin = time.time()
result = a[b]
print(time.time() - begin)

a_list = list(a)
begin = time.time()
result = [a[i] for i in b] + [a[i] for i in b]
print(time.time() - begin)
'''

'''
import os
import re
import numpy as np
from utils.global_variable import RESULT_PATH

current_test_all_dir = "schedule-review-simulation-05-08-12-38-15"

def result_read_func(trace_save_path):
    all_need_iter_paths = []
    for file_dir in os.listdir(trace_save_path):
        if "DL_dispatcher" in file_dir:
            result_read_file_dir = os.path.join(trace_save_path, file_dir)
            # print(file_dir)
            all_need_iter_paths.append(result_read_file_dir)
    success_fail_num_pattern = r'current_success_num:\s*(?P<success>\d+);\s+current_failed_num:\s*(?P<failed>\d+);\s+current_no_submit_num:\s*(?P<no_submit>\d+);\s+current_no_sche_num:\s*(?P<no_sche>\d+);'
    all_final_significance_pattern = r'all_final_significance:\s*(?P<all_final_significance>\d+\.\d+)'
    success_final_significance_pattern = r'success_final_significance:\s*(?P<success_final_significance>\d+\.\d+)'

    success_num = []
    failed_num = []
    all_final_significance_arr = []
    success_final_significance_arr = []
    for file_path in all_need_iter_paths:
        with open(file_path, "r+") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if "current_success_num" in line:
                    match = re.search(success_fail_num_pattern, line)
                    if match:
                        success = int(match.group('success'))
                        failed = int(match.group('failed'))
                        success_num.append(success)
                        failed_num.append(failed)
                    else:
                        print('No match')
                if "all_final_significance" in line:
                    match = re.search(all_final_significance_pattern, line)
                    if match:
                        all_final_significance = float(match.group('all_final_significance'))
                        all_final_significance_arr.append(all_final_significance)
                    else:
                        print('No match')
                if "success_final_significance" in line:
                    match = re.search(success_final_significance_pattern, line)
                    if match:
                        success_final_significance = float(match.group('success_final_significance'))
                        success_final_significance_arr.append(success_final_significance)
                    else:
                        print('No match')
    return success_num, failed_num, all_final_significance_arr, success_final_significance_arr
    

# success_num, failed_num, all_final_significance_arr, success_final_significance_arr = result_read_func(current_test_all_dir)

def final_operate_data(current_test_all_dir):
    trace_save_path = "{}/{}".format(RESULT_PATH, current_test_all_dir)
    success_num_arr, failed_num_arr, all_final_significance_arr, success_final_significance_arr = result_read_func(trace_save_path)
    
    # 新建一个全新的log进行保存
    all_result_path = "{}/all_result.log".format(trace_save_path)
    print("all_result_path: ", all_result_path)
    with open(all_result_path, "w+") as f:
        print("success_num_mean: {}; success_num_min: {}; success_num_max: {}".format(
            np.mean(success_num_arr), min(success_num_arr), max(success_num_arr)
        ), file=f)
        print("failed_num_mean: {}; failed_num_min: {}; failed_num_max: {}".format(
            np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr)
        ), file=f)
        print("all_final_significance_mean: {}; all_final_significance_min: {}; all_final_significance_max: {}".format(
            np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr)
        ), file=f)
        print("success_final_significance_mean: {}; success_final_significance_min: {}; success_final_significance_max: {}".format(
            np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr)
        ), file=f)

final_operate_data(current_test_all_dir)
'''

'''
import numpy as np

# 定义区间和参数
def sample(xmin, xmax, size):
    # 生成符合条件的随机数列表
    data = np.random.uniform(xmin, xmax, size=size)
    samples = np.random.choice(data, size=size, replace=True)

    return samples

num = 100000
L = 0.02
R = 0.8
all_results = sample_28(L, R, num)

# 计算分布情况
hist, bin_edges = np.histogram(all_results, bins=5)

# 输出结果
print("元素分布情况：")
for i in range(len(hist)):
    print("[{:.2f}, {:.2f}): {}".format(bin_edges[i], bin_edges[i+1], hist[i]))
'''

'''
import numpy as np
li = [
    7.6519257353104955,
    7.798196379215965,
    7.590180827383737,
    7.489382511445238,
    7.4520007703541715
]
print("mean: ", np.mean(li))
print("min: ", min(li))
print("max: ", max(li))
'''

'''
import cvxpy as cp
import numpy as np

def get_LP_result(sign_matrix, datablock_privacy_budget_capacity_list, job_target_datablock_selected_num_list, job_privacy_budget_consume_list, solver=cp.CBC):
    job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
    print(f"job_num: {job_num}; datablock_num: {datablock_num}")
    job_target_datablock_selected_num_list = np.array(job_target_datablock_selected_num_list)[np.newaxis, :]
    job_privacy_budget_consume_list = np.array(job_privacy_budget_consume_list)[np.newaxis, :]
    datablock_privacy_budget_capacity_list = np.array(datablock_privacy_budget_capacity_list)[np.newaxis, :]

    print(f"job_target_datablock_selected_num_list: {job_target_datablock_selected_num_list}")
    print(f"job_privacy_budget_consume_list: {job_privacy_budget_consume_list}")
    print(f"datablock_privacy_budget_capacity_list: {datablock_privacy_budget_capacity_list}")

    matrix_X = cp.Variable((job_num, datablock_num), nonneg=True)
    objective = cp.Maximize(
        cp.sum(cp.multiply(sign_matrix, matrix_X))
    )

    # print(f"========= multiply ========")
    # print(np.sum(np.multiply(sign_matrix, matrix_X)))
    print(f"========= sign_matrix.value [{sign_matrix.shape}]=========")
    print(sign_matrix)
    print(f"========= job_target_datablock_selected_num_list.value [{job_target_datablock_selected_num_list.shape}] [{np.squeeze(job_target_datablock_selected_num_list).shape}]=========")
    print(job_target_datablock_selected_num_list)
    print(f"========= job_privacy_budget_consume_list.value [{job_privacy_budget_consume_list.shape}]=========")
    print(job_privacy_budget_consume_list)
    print(f"========= datablock_privacy_budget_capacity_list.value [{datablock_privacy_budget_capacity_list.shape}]=========")
    print(datablock_privacy_budget_capacity_list)
    print(f"========= job_privacy_budget_consume_list @ matrix_X [{(job_privacy_budget_consume_list @ matrix_X).shape}]=========")

    constraints = [
        matrix_X >= 0,
        matrix_X <= 1,
        cp.sum(matrix_X, axis=1) <= np.squeeze(job_target_datablock_selected_num_list),
        (job_privacy_budget_consume_list @ matrix_X) <= datablock_privacy_budget_capacity_list
    ]

    cvxprob = cp.Problem(objective, constraints)
    result = cvxprob.solve(solver, verbose=False)
    # print(matrix_X.value)
    if cvxprob.status != "optimal":
        print('WARNING: Allocation returned by policy not optimal!')

    print("========= matrix_X.value =========")
    print(matrix_X.value)
    if matrix_X.value is None:
        result = np.zeros(shape=(job_num, datablock_num))
    else:
        result = matrix_X.value
    for job_index in range(job_num):
        print(np.sum(result[job_index]))
    return result

sign_matrix = np.array([[0.71558818, 0.7281234 , 0.64447522, 0.71558818, 0.70829598, 0.7281234,
                        0.64447522, 0.70829598, 0.71558818, 0.69991515, 0.57958984, 0.7281234, 
                        0.71558818, 0.69991515, 0.7281234 , 0.57958984, 0.71558818, 0.57958984, 
                        0.69991515, 0.64447522, 0.71558818],
                        [0.71558818, 0.7281234 , 0.64447522, 0.71558818, 0.70829598, 0.7281234, 
                        0.64447522, 0.70829598, 0.71558818, 0.69991515, 0.57958984, 0.7281234, 
                        0.71558818, 0.69991515, 0.7281234 , 0.57958984, 0.71558818, 0.57958984, 
                        0.69991515, 0.64447522, 0.71558818]])
datablock_privacy_budget_capacity_list = np.array([10., 10., 10., 10., 10., 10., 
                                                    10., 10., 10., 10., 10., 10., 
                                                    10., 10., 10., 10., 10., 10., 
                                                    10., 10., 10.])
job_target_datablock_selected_num_list = np.array([8, 7])
job_privacy_budget_consume_list = np.array([0.2423561,  0.10008523])

get_LP_result(sign_matrix, datablock_privacy_budget_capacity_list, job_target_datablock_selected_num_list, job_privacy_budget_consume_list)
'''

'''
import gurobipy


def assignment(cost_matrix):
	# 保存行列标签
	index = cost_matrix.index
	columns = cost_matrix.columns

	# 创建模型
	model = gurobipy.Model('Assignment')
	x = model.addVars(index, columns, vtype=gurobipy.GRB.BINARY)
	model.update()

	# 设置目标函数
	model.setObjective(gurobipy.quicksum(x[i, j] * cost_matrix.at[i, j] for i in index for j in columns))

	# 添加约束条件
	model.addConstr(gurobipy.quicksum(x[i, j] for i in index for j in columns) == min([len(index), len(columns)]))
	model.addConstrs(gurobipy.quicksum(x[i, j] for j in columns) <= 1 for i in index)
	model.addConstrs(gurobipy.quicksum(x[i, j] for i in index) <= 1 for j in columns)

	# 执行最优化
	model.optimize()

	# 输出信息
	result = cost_matrix * 0
	if model.status == gurobipy.GRB.Status.OPTIMAL:
		solution = [k for k, v in model.getAttr('x', x).items() if v == 1]
		for i, j in solution:
			print(f"{i} -> {j}: {cost_matrix.at[i,j]}")
			result.at[i, j] = 1
	return result


if __name__ == '__main__':
	import pandas as pd

	cost_matrix = pd.DataFrame(
			[[4, 8, 7, 15, 12], [7, 9, 17, 14, 10], [6, 9, 12, 8, 7], [6, 7, 14, 6, 10], [6, 9, 12, 10, 6],
				[5, 8, 13, 11, 10]],
			index=['A1', 'A2', 'A3', 'A4', 'A5', 'A6'], columns=['B1', 'B2', 'B3', 'B4', 'B5'])

	assignment(cost_matrix)
'''

import re
import ast

# string = "2023-05-16(Tue)10:38:31 [INFO] at [process_id: 2159268] DL_sched.py,397: dispatcher init job_all_seq_num: 1000"

# match = re.search(r'dispatcher init job_all_seq_num: (?P<test_job_num>\d+)', string)

# if match:
#     map_string = match.group('test_job_num')
#     print(map_string)
    # job_map = ast.literal_eval(map_string)
    # print(job_map['job_11'])
# ('job_10', 'train_sub_42'), ('job_10', 'train_sub_72'), ('job_10', 'train_sub_70'), ('job_10', 'train_sub_49'), ('job_10', 'train_sub_37'), ('job_10', 'train_sub_84'), ('job_10', 'train_sub_96'), ('job_10', 'train_sub_27'), ('job_10', 'train_sub_61'), ('job_10', 'train_sub_1'), ('job_10', 'train_sub_83'), ('job_10', 'train_sub_32'), ('job_10', 'train_sub_47')

'''
string = "from policy [HISwithOrderProVersionPolicy] selected_datablock_identifiers: []"

policy_match = re.search(r"from policy (\[(?P<policy_name>(.*?))\]) selected_datablock_identifiers: (?P<selected_list>\[.*?\])", string)

if policy_match:
    policy = policy_match.group('policy_name')
    datablock_identifiers = policy_match.group('selected_list')
    datablock_identifiers_list = ast.literal_eval(datablock_identifiers)

    print("Policy:", policy)
    print("Datablock Identifiers:", datablock_identifiers_list)
    for item in datablock_identifiers_list:
        print(f"datablock_identifier_item: {item}")
'''

'''
import concurrent.futures

def process_data(arg1, arg2):
    # 执行操作
    result = arg1 + arg2
    return result

# 创建一个Executor
executor = concurrent.futures.ThreadPoolExecutor()

# 定义要传递的多个参数
args1 = [1, 2, 3, 4]
args2 = [5, 6, 7, 8]

# 使用zip函数将多个参数打包为一个可迭代对象
args_combined = zip(args1, args2)

# 使用map函数传递多个参数
results = executor.map(process_data, args1, args2)

# 处理结果
for result in results:
    print(result)
'''

'''
from multiprocessing import Pool

# 定义一个需要并行执行的函数
def my_function(arg1, arg2, arg3):
    # 在这里编写你的函数逻辑
    # 这里只是简单示例，将传入的参数相加并返回结果
    result = arg1 + arg2 + arg3
    return result

# 创建一个Pool对象
pool = Pool()

# 定义要传递给函数的多个参数
arg1_list = [1, 2, 3]
arg2_list = [4, 5, 6]
arg3_list = [7, 8, 9]

# 使用zip()函数将多个参数打包为元组
args_list = zip(arg1_list, arg2_list, arg3_list)

# 使用map()方法调用函数并传递参数
results = pool.starmap(my_function, args_list)

# 打印结果
print(results)

# 关闭Pool
pool.close()
pool.join()
'''

'''
import multiprocessing
import time

def process_data(a, b, pool):
    try:
        time.sleep(b)
        result = a / b
        print(f"result: {result}")
        return result
    except Exception as e:
        # 处理异常的代码
        print(f"Exception in task: {e}")
        pool.terminate()
        return None

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        data_1 = [1, 3, 4]
        data_2 = [2, 0, 1]
        pool_list = [pool] * 3
        args_zip = zip(data_1, data_2, pool_list)
        results = pool.starmap(process_data, args_zip)
'''

'''
import multiprocessing
import datetime
import time
import os
import subprocess
from subprocess import PIPE
import pdb
 
#子进程中某个进程发生异常，则结束整个进程池
all_process = []

def work_process(i):
    time.sleep(1)
    raise Exception(multiprocessing.current_process().name, all_process)
    return 'a' + str(i)
 
 
def throw_exception(name, all_process):
    print('子进程%s发生异常,进程号为%s'%(name, os.getpid()))
    for p in all_process:
        os.system(f"kill -9 {p}")
    time.sleep(2)
 
 
 
if __name__ == '__main__':
    res = []
    with multiprocessing.Pool(processes=10) as pool:
        for i in range(10):  # 遍历所有的文件
            start_time_fp = datetime.datetime.now()
            r = pool.apply_async(func=work_process,args= (i,),error_callback=throw_exception)
            over_time_fp = datetime.datetime.now()
            total_time = (over_time_fp - start_time_fp).total_seconds()
            print('启动单个程序%s完成共计%s秒' % (i,total_time))
        pool.close()  # 关闭进程池，不再接受请求
        pool.join()  # 等待进程池中的任务执行完毕
        print(res)#打印异步结果
'''


import os
import zerorpc
import threading
import time
import torch

def do_sth():
    print("start test_1.py")
    os.system("python -u test_1.py")
    print("wakeup test_1.py")

class Scheduler_server(object):
    def __init__(self, sched_ip, sched_port):
        self.sched_ip = sched_ip
        self.sched_port = sched_port

    def handle_error(self, log_content):
        time.sleep(2)
        print(f"log_content: {log_content}")

    def handle_finished(self, job_id):
        time.sleep(2)
        print(f"job_id: {job_id}")
        return 1

    def start_test(self):
        p = threading.Thread(target=do_sth, daemon=True)
        p.start()

def scheduler_listener_func(scheduler_server_item):
    def sched_func_timely(scheduler_server_item):
        s = zerorpc.Server(scheduler_server_item)
        ip_port = "tcp://0.0.0.0:{}".format(scheduler_server_item.sched_port)
        s.bind(ip_port)
        print("DL_server running in {}".format(ip_port))
        s.run()
        print("self.sched_logger.info sth...")
    p = threading.Thread(target=sched_func_timely, args=(scheduler_server_item, ), daemon=True)
    p.start()
    return p

if __name__ == "__main__":
    sched_ip = "172.18.162.4"
    sched_port = 16042

    scheduler_server_item = Scheduler_server(sched_ip, sched_port)
    sched_p = scheduler_listener_func(scheduler_server_item)
    # scheduler_server_item.start_test()
    flag = True
    scheduler_server_item.start_test()
    # device_0 = torch.device('cuda:1')
    # tensor_size = (10000, 10000)
    # tensor = torch.ones(tensor_size, device=device_0)
    # num_copies = 18  # 复制的次数（根据需要修改）
    # tensors = [tensor.clone() for _ in range(num_copies)]
    
    while flag:
        time.sleep(6)

