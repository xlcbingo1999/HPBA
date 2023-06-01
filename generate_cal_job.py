import time
import os
import zerorpc
import threading
import numpy as np
import sys

class Worker_server(object):
    def __init__(self, local_ip, local_port, max_success_time, sub_train_num, 
                device_index, model_name, train_dataset_name, test_dataset_name, sub_test_key_id,
                SITON_RUN_EPOCH_NUM, EPSILON_one_siton):
        self.local_ip = local_ip
        self.local_port = local_port
        self.need_next = True
        self.success_time = 0
        self.max_success_time = max_success_time

        self.sub_train_num = sub_train_num
        self.current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
        self.all_finished = False

        self.device_index = device_index
        self.model_name = model_name
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        self.sub_test_key_id = sub_test_key_id

        self.SITON_RUN_EPOCH_NUM = SITON_RUN_EPOCH_NUM
        self.EPSILON_one_siton = EPSILON_one_siton

    def finished_job_callback(self, job_id, result, real_duration_time):
        print(f"job_id: {job_id}")
        print(f"result: {result}")
        print(f"real_duration_time: {real_duration_time}")
        
        self.success_time += 1
        if self.success_time > self.max_success_time:
            self.need_next = False

        if self.need_next:
            self.do_cal(self.device_index, self.model_name, self.train_dataset_name, self.test_dataset_name, self.sub_test_key_id, 
                        self.SITON_RUN_EPOCH_NUM, self.EPSILON_one_siton)
        else:
            self.all_finished = True

    def do_cal(self, device_index, model_name, train_dataset_name, test_dataset_name, sub_test_key_id,
                SITON_RUN_EPOCH_NUM, EPSILON_one_siton):
        job_id = 0
        details = get_specific_model_config(model_name)
        
        prefix_dir = f"/home/netlab/DL_lab/opacus_testbed/log_temp_store/result_{self.current_time}/"
        if not os.path.exists(prefix_dir):
            os.makedirs(os.path.dirname(prefix_dir), exist_ok=True)
        
        summary_writer_path = prefix_dir
        summary_writer_key = f"{job_id}"
        logging_file_path = os.path.join(prefix_dir, f"{job_id}.log") 
        model_save_path = os.path.join(prefix_dir, f"{job_id}.pt")
        
        LR = details["LR"]
        DELTA = details["DELTA"]
        MAX_GRAD_NORM = details["MAX_GRAD_NORM"]
        BATCH_SIZE = details["BATCH_SIZE"]
        MAX_PHYSICAL_BATCH_SIZE = details["MAX_PHYSICAL_BATCH_SIZE"]
        TAGRET_ACC = details["TAGRET_ACC"]
        SELECT_BLOCK_NUM = details["SELECT_BLOCK_NUM"]

        simulation_flag = False

        begin_epoch_num = self.success_time * SITON_RUN_EPOCH_NUM
        final_significance = 0.1
        sub_train_key_ids = [f"train_sub_{index}" for index in np.random.choice(range(self.sub_train_num), size=SELECT_BLOCK_NUM)]
        sub_train_key_ids_str = ":".join(sub_train_key_ids)

        print("======= cal_job =======")
        cal_job_cmds = []
        cal_job_cmds.append(f"python -u DL_do_calculate.py")
        cal_job_cmds.append(f"--worker_ip {worker_ip}")
        cal_job_cmds.append(f"--worker_port {worker_port}")

        cal_job_cmds.append(f"--job_id {job_id}")
        cal_job_cmds.append(f"--model_name {model_name}")
        cal_job_cmds.append(f"--train_dataset_name {train_dataset_name}")
        cal_job_cmds.append(f"--test_dataset_name {test_dataset_name}")
        cal_job_cmds.append(f"--sub_train_key_ids {sub_train_key_ids_str}")
        cal_job_cmds.append(f"--sub_test_key_id {sub_test_key_id}")

        cal_job_cmds.append(f"--device_index {device_index}")
        if len(logging_file_path) > 0:
            cal_job_cmds.append("--logging_file_path {}".format(logging_file_path))
        if len(summary_writer_path) > 0 and len(summary_writer_key) > 0:
            cal_job_cmds.append("--summary_writer_path {}".format(summary_writer_path))
            cal_job_cmds.append("--summary_writer_key {}".format(summary_writer_key))
        if len(model_save_path) > 0:
            cal_job_cmds.append("--model_save_path {}".format(model_save_path))

        cal_job_cmds.append("--LR {}".format(LR))
        cal_job_cmds.append("--EPSILON_one_siton {}".format(EPSILON_one_siton))
        cal_job_cmds.append("--DELTA {}".format(DELTA))
        cal_job_cmds.append("--MAX_GRAD_NORM {}".format(MAX_GRAD_NORM))

        cal_job_cmds.append("--BATCH_SIZE {}".format(BATCH_SIZE))
        cal_job_cmds.append("--MAX_PHYSICAL_BATCH_SIZE {}".format(MAX_PHYSICAL_BATCH_SIZE))
        cal_job_cmds.append("--begin_epoch_num {}".format(begin_epoch_num))
        cal_job_cmds.append("--siton_run_epoch_num {}".format(SITON_RUN_EPOCH_NUM))
        cal_job_cmds.append("--final_significance {}".format(final_significance))
        if simulation_flag:
            cal_job_cmds.append("--simulation_flag")

        finally_execute_cmd = " ".join(cal_job_cmds)
        print(finally_execute_cmd)
        print("======= =======")

        os.system(finally_execute_cmd)

def worker_listener_func(worker_server_item):
    def work_func_timely(worker_server_item):
        s = zerorpc.Server(worker_server_item)
        ip_port = "tcp://0.0.0.0:{}".format(worker_server_item.local_port)
        s.bind(ip_port)
        print("DL_server running in {}".format(ip_port))
        s.run()
    p = threading.Thread(target=work_func_timely, args=(worker_server_item, ), daemon=True)
    p.start()
    return p

def get_specific_model_config(model_name):
    if model_name == "CNN":
        return {
            "BATCH_SIZE": 1024,
            "MAX_PHYSICAL_BATCH_SIZE": 512,
            "TARGET_EPOCHS": 50,
            "TAGRET_ACC": 0.7,
            "DELTA": 1e-8,
            "MAX_GRAD_NORM": 1.2,
            "LR": 1e-3,
            "SELECT_BLOCK_NUM": 2
        }
    elif model_name == "FF":
        return {
            "BATCH_SIZE": 1024,
            "MAX_PHYSICAL_BATCH_SIZE": 512,
            "TARGET_EPOCHS": 50,
            "TAGRET_ACC": 0.6,
            "DELTA": 1e-8,
            "MAX_GRAD_NORM": 1.2,
            "LR": 1e-3,
            "SELECT_BLOCK_NUM": 2
        }

if __name__ == "__main__":
    worker_ip = "172.18.162.5"
    worker_port = 162538

    max_success_time = 100
    sub_train_num = 144
    device_index = 2
    model_name = "FF"
    train_dataset_name = "EMNIST"
    test_dataset_name = "EMNIST_MNIST-1000_1000" # EMNIST-2000 MNIST-2000 EMNIST_MNIST-1000_1000
    sub_test_key_id = "test_sub_0"

    SITON_RUN_EPOCH_NUM = 5
    EPSILON_one_siton = 0.25 # 0.1/5, 0.5/5

    worker_server_item = Worker_server(worker_ip, worker_port, max_success_time, sub_train_num,
                        device_index, model_name, train_dataset_name, test_dataset_name, sub_test_key_id,
                        SITON_RUN_EPOCH_NUM, EPSILON_one_siton)
    p = worker_listener_func(worker_server_item)

    worker_server_item.do_cal(device_index, model_name, train_dataset_name, test_dataset_name, sub_test_key_id,
                            SITON_RUN_EPOCH_NUM, EPSILON_one_siton)
    while not worker_server_item.all_finished:
        time.sleep(10)
    print("DL sched finished!!")

    sys.exit(0)