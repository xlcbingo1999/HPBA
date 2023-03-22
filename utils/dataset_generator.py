import random

    

def _generate_epsilon_capacity(rng):
    epsilon = 50.0
    r = rng.uniform(0, 1)
    if 0.7 <= r <= 0.8:
        epsilon = 20.0
    elif 0.8 <= r <= 0.95:
        epsilon = 10.0
    elif 0.95 <= r:
        epsilon = 5.0
    return epsilon

def _generate_delta_capacity(rng):
    return 4e-6

def generate_subtrain_datablocks(dataset_name, block_num,
                                epsilon_capacity_generator=_generate_epsilon_capacity,
                                delta_capacity_generator=_generate_delta_capacity):
    rng = random.Random()
    
    all_blocks = {}
    min_epsilon_capacity = float("inf")
    for index in range(block_num):
        block_name = "train_sub_{}".format(index)
        all_blocks[block_name] = {
            "train_type": index,
            "dataset_name": dataset_name,
            "epsilon_capacity": epsilon_capacity_generator(rng),
            "delta_capacity": delta_capacity_generator(rng),
        }
        min_epsilon_capacity = min(min_epsilon_capacity, all_blocks[block_name]["epsilon_capacity"])
    sub_train_result = all_blocks
    test_result = {}
    return sub_train_result, test_result, min_epsilon_capacity

def generate_all_subtrain_datablocks(dataset_name_2_block_num):
    subtrain_datasets_map = {}
    test_datasets_map = {}
    min_epsilon_capacity = float("inf")
    for dataset_name, block_num in dataset_name_2_block_num.items():
        sub_train_result, test_result, dataset_min_epsilon_capacity = generate_subtrain_datablocks(dataset_name, block_num)
        subtrain_datasets_map[dataset_name] = sub_train_result
        test_datasets_map[dataset_name] = test_result
        min_epsilon_capacity = min(min_epsilon_capacity, dataset_min_epsilon_capacity)
    return subtrain_datasets_map, test_datasets_map, min_epsilon_capacity

