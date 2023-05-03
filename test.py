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
