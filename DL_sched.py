import zerorpc
from concurrent.futures import ThreadPoolExecutor
import threading

from utils.data_loader import fetch_new_dataset
from utils.get_profiler_significance import get_profiler_significance_result
from utils.global_functions import FAILED_RESULT_KEY, JOB_STATUS_KEY, JOB_STATUS_UPDATE_PATH
from utils.global_variable import SCHE_IP, SCHE_PORT, INIT_WORKERIDENTIFIERS, INIT_WORKERIP_2_PORTS

def DL_server_do_jobs(args):
    job_id, origin_info, worker_ip, worker_port, worker_gpu_id, worker_dataset_config = args
    client = zerorpc.Client()
    client.connect("tcp://{}:{}".format(worker_ip, worker_port))
    
    client.begin_job(job_id, worker_gpu_id, worker_dataset_config, origin_info)

def init_worker_dataset(client, fetch_dataset_origin_info, keep_origin_dataset):
    # print("fetch_dataset_origin_info: {}".format(fetch_dataset_origin_info))
    # print("keep_origin_dataset: ", keep_origin_dataset)
    client.initial_dataset(fetch_dataset_origin_info, keep_origin_dataset)

class Scheduler_server(object):
    def __init__(self, sched_ip, sched_port, init_workeridentifiers, init_workerip_2_ports):
        self.sched_ip = sched_ip
        self.sched_port = sched_port
        self.workeridentifier_2_gpustatus = {key:False for key in init_workeridentifiers}

        self.workerip_2_datasetstatus = {key:False for key in init_workerip_2_ports}
        self.workerip_2_ports = init_workerip_2_ports
        
        self.train_all_dataset = None
        self.sub_train_datasets = None
        self.valid_dataset = None
        self.output_size = None
        self.vocab_size = None


        self.jobid_2_status = {} # 0: no sche; 1: sched target decide; 2: runnning; 3: success finished; 4: failed;
        self.status_2_jobid = {JOB_STATUS_KEY.NO_SCHE: [], 
                                JOB_STATUS_KEY.DONE_GPU_SCHED: [], 
                                JOB_STATUS_KEY.DONE_DATASET_SCHED: [], 
                                JOB_STATUS_KEY.DONE_ALL_SCHED: [],
                                JOB_STATUS_KEY.RUNNING: [], 
                                JOB_STATUS_KEY.FINISHED: [],
                                JOB_STATUS_KEY.FAILED: []}
        self.jobid_2_results = {}
        self.jobid_2_origininfo = {}
        self.jobid_2_gputarget = {}
        self.jobid_2_datasettargetconfig = {}
        self.jobid_2_trainconfig = {}

    def get_worker_zerorpc_client(self, worker_ip, worker_port):
        tcp_ip_port = "tcp://{}:{}".format(worker_ip, worker_port)
        client = zerorpc.Client()
        client.connect(tcp_ip_port)
        return client

    def get_worker_identifier_detail(self, worker_identifier):
        worker_ip, worker_gpu_id = worker_identifier.split('-')
        return worker_ip, int(worker_gpu_id)

    def initial_dataset(self, fetch_dataset_origin_info):
        DATASET_NAME = fetch_dataset_origin_info['DATASET_NAME']
        LABEL_TYPE = fetch_dataset_origin_info['LABEL_TYPE']
        VALID_SIZE = fetch_dataset_origin_info['VALID_SIZE']
        SEQUENCE_LENGTH = fetch_dataset_origin_info['SEQUENCE_LENGTH']
        # BATCH_SIZE = fetch_dataset_origin_info[] # TODO(xlc): 这里明显有问题, 不应该把BATCH_SIZE这种内容在这里传
        SPLIT_NUM = fetch_dataset_origin_info['SPLIT_NUM']
        # ALPHA = fetch_dataset_origin_info[]
        same_capacity = fetch_dataset_origin_info['same_capacity']
        # plot_path = fetch_dataset_origin_info[]
        train_all_dataset, sub_train_datasets, valid_dataset, \
            output_size, vocab_size = fetch_new_dataset(DATASET_NAME, LABEL_TYPE, VALID_SIZE, SEQUENCE_LENGTH, SPLIT_NUM, same_capacity)
        self.train_all_dataset = train_all_dataset
        self.sub_train_datasets = sub_train_datasets
        self.valid_dataset = valid_dataset
        self.output_size = output_size
        self.vocab_size = vocab_size

    def initial_sched_and_all_workers_dataset(self, fetch_dataset_origin_info, keep_origin_dataset):
        # 初始化sched的数据集
        self.initial_dataset(fetch_dataset_origin_info)
        # 初始化worker的数据集
        # args = []
        for worker_ip in self.workerip_2_datasetstatus:
            worker_port = self.workerip_2_ports[worker_ip]
            client = self.get_worker_zerorpc_client(worker_ip, worker_port)
            # args.append([client, fetch_dataset_origin_info, keep_origin_dataset])
            init_worker_dataset(client, fetch_dataset_origin_info, keep_origin_dataset)
        # args = tuple(args)
        # print(args)
        # with ThreadPoolExecutor(max_workers=len(args)) as pool:
        #     pool.map(init_worker_dataset, args)

    def add_jobs(self, jobs_detail):
        for id, origin_info in jobs_detail:
            if id in self.jobid_2_status:
                print("Waring: job {} has existed!".format(id))
                continue
            else:
                print("success add new job {}".format(id))
                self.jobid_2_status[id] = 0
                self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE].append(id)
                self.jobid_2_results[id] = None
                self.jobid_2_origininfo[id] = origin_info
                self.jobid_2_gputarget[id] = None
                self.jobid_2_datasettargetconfig[id] = None
                self.jobid_2_trainconfig[id] = None
        # self.report_sched_status("after add all jobs")

    def worker_finished_job_callback(self, job_id, origin_info, result):
        print("=========  Scheduler: Job Finished! ===========")
        print("job_id: {}".format(job_id))
        print("origin_info: {}".format(origin_info))
        print("result: {}".format(result))
        print("====================")
        self.sche_update_job_status(job_id, JOB_STATUS_KEY.FINISHED)
        self.sche_reflash_job_status([job_id], JOB_STATUS_KEY.RUNNING, JOB_STATUS_KEY.FINISHED)
        self.jobid_2_results[job_id] = result

    def worker_failed_job_callback(self, job_id, failed_result_key):
        print("=========  Scheduler: Job Failed! ===========")
        print("job_id: {}".format(job_id))
        print("failed_result_key: {}".format(failed_result_key))
        print("====================")

    def worker_dataset_status_callback(self, worker_ip, new_status):
        # 应该是一个单点通信, 单个worker直接和调度器通信即可
        self.workerip_2_datasetstatus[worker_ip] = new_status
        print("Scheduler: Worker [{}]'s dataset Status Update to {}".format(worker_ip, new_status))

    def worker_gpu_status_callback(self, worker_identifier, new_status):
        # 应该是一个单点通信, 单个worker直接和调度器通信即可
        self.workeridentifier_2_gpustatus[worker_identifier] = new_status
        print("Scheduler: Worker's gpu [{}] Status Update to {}".format(worker_identifier, new_status))

    def sche_update_job_status(self, job_id, new_status):
        self.jobid_2_status[job_id] = new_status

    def sche_reflash_job_status(self, job_ids, origin_status, new_status):
        for job_id in job_ids:
            self.status_2_jobid[origin_status].remove(job_id)
            self.status_2_jobid[new_status].append(job_id)

    def report_sched_status(self, location):
        print("======== Scheduler Status in {} ========".format(location))
        print("self.jobid_2_status: {}".format(self.jobid_2_status))
        print("self.status_2_jobid: {}".format(self.status_2_jobid))
        print("self.jobid_2_gputarget: {}".format(self.jobid_2_gputarget))
        print("==================================")

    def get_target_job_status_update_path_and_status(self, job_id, operator):
        origin_status = self.jobid_2_status[job_id]
        if operator == 'gpu':
            if origin_status == JOB_STATUS_KEY.NO_SCHE:
                return JOB_STATUS_UPDATE_PATH.NOSCHED_2_GPUSCHED, JOB_STATUS_KEY.DONE_GPU_SCHED
            elif origin_status == JOB_STATUS_KEY.DONE_DATASET_SCHED:
                return JOB_STATUS_UPDATE_PATH.DATASETSCHED_2_ALLSCHED, JOB_STATUS_KEY.DONE_ALL_SCHED
        elif operator == 'dataset':
            if origin_status == JOB_STATUS_KEY.NO_SCHE:
                return JOB_STATUS_UPDATE_PATH.NOSCHED_2_DATASETSCHED, JOB_STATUS_KEY.DONE_DATASET_SCHED
            elif origin_status == JOB_STATUS_KEY.DONE_GPU_SCHED:
                return JOB_STATUS_UPDATE_PATH.GPUSCHED_2_ALLSCHED, JOB_STATUS_KEY.DONE_ALL_SCHED

    def get_job_status_update_origin_target(self, status_update_path):
        if status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_GPUSCHED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_GPU_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_DATASETSCHED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_DATASET_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.GPUSCHED_2_ALLSCHED:
            origin_status = JOB_STATUS_KEY.DONE_GPU_SCHED
            target_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.DATASETSCHED_2_ALLSCHED:
            origin_status = JOB_STATUS_KEY.DONE_DATASET_SCHED
            target_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        return origin_status, target_status

    def get_scheduling_datablock_result(self):
        probe_model = None
        device = None
        TARGET_SELECT_NUM = 1
        selected_datablock_ids, not_selected_datablock_ids, \
            final_scores, origin_sub_label_distributions = \
            get_profiler_significance_result(self.train_all_dataset, self.sub_train_datasets, 
                                            probe_model, device, TARGET_SELECT_NUM)
        del probe_model
        return selected_datablock_ids, not_selected_datablock_ids, final_scores, origin_sub_label_distributions

    def sched_dispatch(self):
        # 未调度先调度
        to_reflash_job_ids = {
            JOB_STATUS_UPDATE_PATH.NOSCHED_2_DATASETSCHED: [],
            JOB_STATUS_UPDATE_PATH.GPUSCHED_2_ALLSCHED: []
        }
        to_sched_dataset_jobids = self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE] + self.status_2_jobid[JOB_STATUS_KEY.DONE_GPU_SCHED]
        for job_id in to_sched_dataset_jobids:
            # 需要使用复杂一点的调度策略了
            selected_datablock_ids, not_selected_datablock_ids, final_scores, origin_sub_label_distributions = self.get_scheduling_datablock_result()
            self.jobid_2_datasettargetconfig[job_id]['selected_datablock_ids'] = selected_datablock_ids
            self.jobid_2_datasettargetconfig[job_id]['not_selected_datablock_ids'] = not_selected_datablock_ids
            self.jobid_2_datasettargetconfig[job_id]['final_scores'] = final_scores
            self.jobid_2_datasettargetconfig[job_id]['origin_sub_label_distributions'] = origin_sub_label_distributions
            self.jobid_2_datasettargetconfig[job_id]['is_select'] = True

            status_update_path, target_status = self.get_target_job_status_update_path_and_status(job_id, 'dataset')
            to_reflash_job_ids[status_update_path].append(job_id)

        for status_path in to_reflash_job_ids:
            origin_status, target_status = self.get_job_status_update_origin_target(status_path)
            for job_id in to_reflash_job_ids[status_path]:
                self.sche_update_job_status(job_id, target_status)
            self.sche_reflash_job_status(to_reflash_job_ids[status_path], origin_status, target_status)
        
        to_reflash_job_ids = {
            JOB_STATUS_UPDATE_PATH.NOSCHED_2_GPUSCHED: [],
            JOB_STATUS_UPDATE_PATH.DATASETSCHED_2_ALLSCHED: []
        }
        to_sched_gpu_jobids = self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE] + self.status_2_jobid[JOB_STATUS_KEY.DONE_DATASET_SCHED]
        for job_id in to_sched_gpu_jobids:
            all_workers_list = list(self.workeridentifier_2_gpustatus.keys())
            target_worker_id = job_id % len(all_workers_list) # 决定worker_gpu
            target_worker_identifier = all_workers_list[target_worker_id]
            target_worker_ip, _ = self.get_worker_identifier_detail(target_worker_identifier)
            
            if self.workeridentifier_2_gpustatus[target_worker_identifier] and self.workerip_2_datasetstatus[target_worker_ip]: # 判断GPU的状态
                self.jobid_2_gputarget[job_id] = target_worker_identifier
                status_update_path, target_status = self.get_target_job_status_update_path_and_status(job_id, 'gpu')
                to_reflash_job_ids[status_update_path].append(job_id)
            
        for status_path in to_reflash_job_ids:
            origin_status, target_status = self.get_job_status_update_origin_target(status_path)
            for job_id in to_reflash_job_ids[status_path]:
                self.sche_update_job_status(job_id, target_status)
            self.sche_reflash_job_status(to_reflash_job_ids[status_path], origin_status, target_status)
        
        self.report_sched_status("after sched_dispatch all jobs")
        self.placement_dispatch()

    def placement_dispatch(self):
        # 放置任务
        args = []
        to_reflash_job_ids = []
        
        for job_id in self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED]:
            origin_info = self.jobid_2_origininfo[job_id]
            worker_dataset_config = self.jobid_2_datasettargetconfig[job_id]
            worker_identifier = self.jobid_2_gputarget[job_id]
            worker_ip, worker_gpu_id = self.get_worker_identifier_detail(worker_identifier)
            worker_port = self.workerip_2_ports[worker_ip]
            args.append([job_id, origin_info, worker_ip, worker_port, worker_gpu_id, worker_dataset_config])
            self.sche_update_job_status(job_id, JOB_STATUS_KEY.RUNNING)
            to_reflash_job_ids.append(job_id)
        self.sche_reflash_job_status(to_reflash_job_ids, JOB_STATUS_KEY.DONE_ALL_SCHED, JOB_STATUS_KEY.RUNNING)
        print("check args: {}".format(args))
        if len(args) > 0:
            self.report_sched_status("after placement_dispatch all jobs")
            args = tuple(args)
            print("args: {} (len: {})".format(args, len(args)))
            
            with ThreadPoolExecutor(max_workers=len(args)) as pool:
                pool.map(DL_server_do_jobs, args)

def scheduler_listener_func(scheduler_server_item):
    s = zerorpc.Server(scheduler_server_item)
    ip_port = "tcp://0.0.0.0:{}".format(scheduler_server_item.sched_port)
    s.bind(ip_port)
    print("DL_server running in {}".format(ip_port))
    s.run()
    print("print sth...")

if __name__ == "__main__":
    sched_ip = SCHE_IP
    sched_port = SCHE_PORT
    init_workeridentifiers = INIT_WORKERIDENTIFIERS
    init_workerip_2_ports = INIT_WORKERIP_2_PORTS

    scheduler_server_item = Scheduler_server(sched_ip, sched_port, init_workeridentifiers, init_workerip_2_ports)
    scheduler_listener_func(scheduler_server_item)