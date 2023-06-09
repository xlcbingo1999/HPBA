
import json
all_cmds = "python -u DL_do_calculate.py \
    --worker_ip 172.18.162.5 --worker_port 16205 \
    --job_id job_68 --model_name FF \
    --train_dataset_name EMNIST --test_dataset_name EMNIST-2000 \
    --sub_train_key_ids train_sub_15:train_sub_19:train_sub_7:train_sub_12 \
    --sub_test_key_id test_sub_0 \
    --sub_train_dataset_config_path /mnt/linuxidc_client/dataset/EMNIST/subtrain_144_split_1.0_dirichlet.json \
    --test_dataset_config_path /mnt/linuxidc_client/dataset/EMNIST/subtest.json \
    --device_index 1 \
    --summary_writer_path /mnt/linuxidc_client/opacus_testbed_result/schedule-review-testbed-06-08-16-06-09 \
    --summary_writer_key job_68 \
    --logging_file_path /mnt/linuxidc_client/opacus_testbed_result/schedule-review-testbed-06-08-16-06-09/job_68.log \
    --model_save_path /mnt/linuxidc_client/opacus_testbed_result/schedule-review-testbed-06-08-16-06-09/job_68.pt \
    --LR 0.001 --EPSILON_one_siton 0.006899650899611198 --DELTA 0.00025 --MAX_GRAD_NORM 1.2 --BATCH_SIZE 1024 \
    --MAX_PHYSICAL_BATCH_SIZE 96 --begin_epoch_num 0 --siton_run_epoch_num 10 --final_significance 0.12042031075827023"

# print(all_cmds)
cmd_sub_str_arr = all_cmds.split(" ")
cmd_sub_str_arr = [s for s in cmd_sub_str_arr if len(s) > 0]
print(json.dumps(cmd_sub_str_arr))