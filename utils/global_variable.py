GLOBAL_PATH = "/home/netlab/DL_lab/opacus_testbed"
GPU_PATH = "/mnt/linuxidc_client/gpu_states"
DATASET_PATH = "/mnt/linuxidc_client/dataset"
DATASET_NAME = "EMNIST"
SUB_TRAIN_DATASET_CONFIG_PATH = DATASET_PATH + "/{}/subtrain_12_split_1.0_dirichlet.json".format(DATASET_NAME)
TEST_DATASET_CONFIG_PATH = DATASET_PATH + "/{}/subtest.json".format(DATASET_NAME)
SIGNIFICANCE_TRACE_PREFIX_PATH = DATASET_PATH + "/traces"
RECONSTRUCT_TRACE_PREFIX_PATH = DATASET_PATH + "/traces/reconstruct_traces"
ALIBABA_DP_TRACE_PATH = DATASET_PATH + "/PRIVACY/alibaba-dp-workload/sample_output"

LOGGING_DATE = "20230323"
RESULT_PATH = "/mnt/linuxidc_client/opacus_testbed_result"

# WORKER_LOCAL_IP = "172.18.162.6"
# WORKER_LOCAL_PORT = 16206
# ALL_WORKER_IPS_2_PORTS = {"172.18.162.3": 16203, "172.18.162.6": 16206}
# ALL_WORKERIDENTIFIERS = [
#     "172.18.162.2-0", "172.18.162.2-1", "172.18.162.2-2", "172.18.162.2-3",
#     "172.18.162.3-0", "172.18.162.3-1", "172.18.162.3-2", "172.18.162.3-3",
#     "172.18.162.4-0", "172.18.162.4-1", "172.18.162.4-2", "172.18.162.4-3",
#     "172.18.162.5-0", "172.18.162.5-1", "172.18.162.5-2", "172.18.162.5-3",
#     "172.18.162.6-0", "172.18.162.6-1", "172.18.162.6-2", "172.18.162.6-3"
# ]
# ALL_WORKERIP_2_PORTS = {
#     "172.18.162.2": 16202, 
#     "172.18.162.3": 16203,
#     "172.18.162.4": 16204, 
#     "172.18.162.5": 16205,
#     "172.18.162.6": 16206,
# }


# SCHE_IP = "172.18.162.6"
# SCHE_PORT = 16200

# DISPATCHER_IP = "172.18.162.6"
# DISPATCHER_PORT = 16100
