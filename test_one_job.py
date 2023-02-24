from utils.opacus_scheduler_job import do_calculate_func

job_id = 0
model_name = "FF-split"
train_dataset_raw_paths = ["/mnt/linuxidc_client/dataset/Amazon_Review_split/class_2_1.8MTrain_20kTest/sub_train_3_split_10.csv"]
test_dataset_raw_path = "/mnt/linuxidc_client/dataset/Amazon_Review_split/class_2_1.8MTrain_20kTest/test_split_10.csv"
dataset_name = "class_2_1.8MTrain_20kTest"
label_type = "sentiment"
selected_datablock_identifiers = [3]
not_selected_datablock_identifiers = [0, 1, 2]
device = 1
summary_writer_path = ""
loss_func = "CrossEntropyLoss"
LR = 1e-3
EPSILON = 0.0
EPOCH_SET_EPSILON = False
DELTA = 1e-5
MAX_GRAD_NORM = 1.2
BATCH_SIZE = 64
MAX_PHYSICAL_BATCH_SIZE = 32
EPOCHS = 20
label_distributions = {
    "1": 306158,
    "0": 53841
}
train_configs = {
    "hidden_size": [150, 110],
    "embedding_size": 100,
    "sequence_length": 50
}

do_calculate_func(job_id, model_name, train_dataset_raw_paths, test_dataset_raw_path,
                    dataset_name, label_type, selected_datablock_identifiers, not_selected_datablock_identifiers,
                    loss_func, device, summary_writer_path,
                    LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, 
                    BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPOCHS,
                    label_distributions, train_configs)