from utils.opacus_scheduler_job import do_calculate_func
import argparse
import json
import zerorpc

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--worker_ip", type=str, required=True)
    parser.add_argument("--worker_port", type=str, required=True)
    
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_dataset_raw_paths", type=str, required=True) # : 用这个进行split
    parser.add_argument("--test_dataset_raw_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--label_type", type=str, required=True)
    parser.add_argument("--selected_datablock_identifiers", type=str, required=True) # : 用这个进行split
    parser.add_argument("--not_selected_datablock_identifiers", type=str, required=True) # : 用这个进行split
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--summary_writer_path", type=str, default="")
    parser.add_argument("--LR", type=float, required=True)
    parser.add_argument("--EPSILON", type=float, required=True)
    parser.add_argument("--EPOCH_SET_EPSILON", action="store_true")
    parser.add_argument("--DELTA", type=float, required=True)
    parser.add_argument("--MAX_GRAD_NORM", type=float, required=True)

    parser.add_argument("--BATCH_SIZE", type=int, required=True)
    parser.add_argument("--MAX_PHYSICAL_BATCH_SIZE", type=int, required=True)
    parser.add_argument("--EPOCHS", type=int, required=True)
    
    parser.add_argument("--label_distributions", type=str, required=True) # 需要从str转成json
    parser.add_argument("--train_configs", type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_df_config()

    worker_ip = args.worker_ip 
    worker_port = args.worker_port 
    job_id = args.job_id 
    model_name = args.model_name 
    train_dataset_raw_paths = args.train_dataset_raw_paths
    train_dataset_raw_paths = train_dataset_raw_paths.split(":")
    test_dataset_raw_path = args.test_dataset_raw_path
    dataset_name = args.dataset_name
    label_type = args.label_type
    selected_datablock_identifiers = args.selected_datablock_identifiers
    selected_datablock_identifiers = selected_datablock_identifiers.split(":")
    not_selected_datablock_identifiers = args.not_selected_datablock_identifiers
    not_selected_datablock_identifiers = not_selected_datablock_identifiers.split(":")
    device = args.device
    early_stopping = args.early_stopping
    summary_writer_path = args.summary_writer_path
    LR = args.LR 
    EPSILON = args.EPSILON
    EPOCH_SET_EPSILON = args.EPOCH_SET_EPSILON 
    DELTA = args.DELTA
    MAX_GRAD_NORM = args.MAX_GRAD_NORM 
    BATCH_SIZE = args.BATCH_SIZE
    MAX_PHYSICAL_BATCH_SIZE = args.MAX_PHYSICAL_BATCH_SIZE
    EPOCHS = args.EPOCHS
    label_distributions = args.label_distributions
    label_distributions = json.loads(label_distributions)
    
    train_configs = args.train_configs
    train_configs = json.loads(train_configs)

    job_id, all_results, real_duration_time = do_calculate_func(job_id, model_name, train_dataset_raw_paths, test_dataset_raw_path,
                    dataset_name, label_type, selected_datablock_identifiers, not_selected_datablock_identifiers,
                    device, summary_writer_path,
                    LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, 
                    BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPOCHS,
                    label_distributions, train_configs)
    
    tcp_ip_port = "tcp://{}:{}".format(worker_ip, worker_port)
    client = zerorpc.Client()
    client.connect(tcp_ip_port)
    client.finished_job_callback(job_id, all_results, real_duration_time)