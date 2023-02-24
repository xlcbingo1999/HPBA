from utils.opacus_scheduler_job import do_calculate_func
import argparse

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--EPSILON", type=float, default=15.0)
    parser.add_argument("--sub_train_datablock", type=str, default="sub_train_3_split_10")
    parser.add_argument("--device_index", type=int, default=0)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_df_config()
    job_id = 0
    model_name = "FF-split"
    sub_train_datablock = args.sub_train_datablock
    train_dataset_raw_paths = [
        "/mnt/linuxidc_client/dataset/Amazon_Review_split/class_2_1.8MTrain_20kTest/{}.csv".format(sub_train_datablock)
    ]
    test_dataset_raw_path = "/mnt/linuxidc_client/dataset/Amazon_Review_split/class_2_1.8MTrain_20kTest/test_split_10.csv"
    dataset_name = "class_2_1.8MTrain_20kTest"
    label_type = "sentiment"
    selected_datablock_identifiers = [3]
    not_selected_datablock_identifiers = [0, 1, 2]
    device_index = args.device_index
    summary_writer_path = "/home/netlab/DL_lab/opacus_testbed/tensorboard_20230225"
    loss_func = "CrossEntropyLoss"
    LR = 1e-3
    EPSILON = args.EPSILON
    EPOCH_SET_EPSILON = False
    DELTA = 1e-5
    MAX_GRAD_NORM = 1.2
    BATCH_SIZE = 256
    MAX_PHYSICAL_BATCH_SIZE = 16
    EPOCHS = 40
    label_distributions = {}
    train_configs = {
        "hidden_size": [150, 110],
        "embedding_size": 100,
        "sequence_length": 50
    }

    do_calculate_func(job_id, model_name, train_dataset_raw_paths, test_dataset_raw_path,
                        dataset_name, label_type, selected_datablock_identifiers, not_selected_datablock_identifiers,
                        loss_func, device_index, summary_writer_path,
                        LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, 
                        BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPOCHS,
                        label_distributions, train_configs)