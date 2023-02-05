import warnings
warnings.simplefilter("ignore")
import torch
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
from functools import reduce
from utils.global_functions import normal_counter



def get_u2_multi_det_S_q(selected_datablock_ids, S_matrix, sub_dataset_us):
    if len(selected_datablock_ids) == 0:
        return 0.0
    else:
        subset_us = [sub_dataset_us[id] for id in selected_datablock_ids]
        subset_us = [np.float_power(u, 2) for u in subset_us]
        multi_us = reduce(lambda x, y: x*y, subset_us)
        sub_S_matrix = S_matrix[selected_datablock_ids, :][:, selected_datablock_ids]
        # print("sub_S_matrix: {}".format(sub_S_matrix))
        det_sub_S_matrix = np.linalg.det(sub_S_matrix) # 这个非常小
        result = multi_us * np.abs(det_sub_S_matrix)
        print("subset_us: {} with multi_us: {} and det_sub_S_matrix: {}".format(subset_us, multi_us, det_sub_S_matrix))
        print("calculate score one side: {}".format(result))
        return result


def get_profiler_significance_result(train_all_dataset, sub_train_datasets, embedding_model, device, target_select_num):
    # 先得到原始的分布
    origin_label_distribution = Counter(train_all_dataset.get_subset_targets())
    print("check originl label distribution: {}".format(origin_label_distribution))
    origin_label_distribution = normal_counter(origin_label_distribution)
    print("check norm label distribution: {}".format(origin_label_distribution))
    origin_keys = set(origin_label_distribution.keys())
    
    
    # sub_label_intersections = []
    # sub_label_complementarys = []
    sub_dataset_us = []
    origin_sub_label_distributions = []
    for sub_dataset in sub_train_datasets:
        origin_sub_label_distribution = Counter(sub_dataset.get_subset_targets())
        origin_sub_label_distributions.append(origin_sub_label_distribution)
        print("check originl sub label distribution: {}".format(origin_sub_label_distribution))
        sub_label_distribution = normal_counter(origin_sub_label_distribution)
        print("check norm sub label distribution: {}".format(sub_label_distribution))
        

        sub_dataset_label = set(sub_label_distribution.keys())
        inter_with_origin = origin_keys & sub_dataset_label
        comple_with_origin = origin_keys - sub_dataset_label
        print("inter_with_origin: {}".format(inter_with_origin))
        print("comple_with_origin: {}".format(comple_with_origin))
        # sub_label_intersections.append(inter_with_origin)
        # sub_label_complementarys.append(comple_with_origin)
        
        all_labels_distribution_diff = 0.0
        for inter_label in inter_with_origin:
            all_labels_distribution_diff += pow(sub_label_distribution[inter_label] - origin_label_distribution[inter_label], 2)
        for comple_lable in comple_with_origin: # 有区别吗?
            all_labels_distribution_diff += pow(sub_label_distribution[comple_lable] - origin_label_distribution[comple_lable], 2)
        # print("all_labels_distribution_diff: {}".format(all_labels_distribution_diff))
        # print("sqrt: {}".format(np.sqrt(all_labels_distribution_diff)))
        sub_dataset_u = 2 - np.sqrt(all_labels_distribution_diff)
        sub_dataset_us.append(sub_dataset_u)
    print("sub_dataset_us: {}".format(sub_dataset_us))
    
    # diversity driven datablock selection
    if embedding_model is None:
        S_matrix = np.ones(shape=(len(sub_train_datasets), len(sub_train_datasets)))
    else:
        embedding_model.eval()
        sub_dataset_Hs = []
        for sub_dataset in sub_train_datasets:
            # 首先封装成DataLoader
            probe_batch_size = int(len(sub_dataset) / 16)
            sub_dataloder = DataLoader(sub_dataset, batch_size=probe_batch_size)
            sub_dataloader_output_matrix = None
            for i, (images, target) in enumerate(sub_dataloder):
                images = images.to(device)
                target = target.to(device)

                # compute output
                output = embedding_model(images)
                if sub_dataloader_output_matrix is None:
                    sub_dataloader_output_matrix = output.detach().cpu().numpy()
                else:
                    sub_dataloader_output_matrix = np.r_[sub_dataloader_output_matrix, output.detach().cpu().numpy()]
            # 是否需要做行归一化?
            
            sub_dataset_H = np.mean(sub_dataloader_output_matrix, axis=0)
            sub_dataset_Hs.append(sub_dataset_H)
        S_matrix = np.zeros(shape=(len(sub_train_datasets), len(sub_train_datasets)))
        for i in range(len(sub_dataset_Hs)):
            for j in range(i, len(sub_dataset_Hs)):
                S_matrix[i, j] = np.dot(sub_dataset_Hs[i], sub_dataset_Hs[j]) / (np.linalg.norm(sub_dataset_Hs[i]) * np.linalg.norm(sub_dataset_Hs[j]))
                S_matrix[j, i] = S_matrix[i, j]
                print("check dot: {} norm1: {} norm2: {}".format(
                    np.dot(sub_dataset_Hs[i], sub_dataset_Hs[j]),
                    np.linalg.norm(sub_dataset_Hs[i]),
                    np.linalg.norm(sub_dataset_Hs[j]),
                ))
        print("S_matrix: {}".format(S_matrix))
    selected_datablock_id = []
    final_scores = []
    valid_datablock_id = list(range(len(sub_dataset_us)))
    while len(selected_datablock_id) < target_select_num:
        origin_score = get_u2_multi_det_S_q(selected_datablock_id, S_matrix, sub_dataset_us)
        final_score_list = {}
        for vid in valid_datablock_id:
            new_datablock_ids = []
            for id in selected_datablock_id:
                new_datablock_ids.append(id)
            new_datablock_ids.append(vid)
            # new_datablock_ids.sort()
            print("selected_datablock_id: {}".format(new_datablock_ids))
            new_score = get_u2_multi_det_S_q(new_datablock_ids, S_matrix, sub_dataset_us)
            final_score = new_score - origin_score
            final_score_list[vid] = final_score
        final_scores.append(final_score_list)
        print("final_score_list: {}".format(final_score_list))
        select_id = max(final_score_list, key=lambda x: final_score_list[x])
        selected_datablock_id.append(select_id)
        valid_datablock_id.remove(select_id)
        print("check select_id: {}".format(select_id))
    return selected_datablock_id, valid_datablock_id, final_scores, origin_sub_label_distributions

'''
def get_profiler_result(train_loaders,
                        model, optimizer, criterion, device, 
                        privacy_engine, EPSILON, DELTA, MAX_GRAD_NORM):
    significant_acc_list = []
    significant_loss_list = []
    epsilon_cesume_list = []
    
    origin_model_state_dict = copy.deepcopy(model.state_dict())
    origin_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
    for i, train_loader in enumerate(train_loaders):
        model.train()
        optimizer.zero_grad()

        model, optimizer, train_loader = get_privacy_dataloader(privacy_engine, model, optimizer, train_loader, 1, EPSILON, DELTA, MAX_GRAD_NORM)
        train_acc, train_loss, epsilon_consume = image_classification_train(model, train_loader, optimizer, criterion, -1, 
                                                                            device, privacy_engine,
                                                                            MAX_PHYSICAL_BATCH_SIZE, DELTA)
        significant_acc_list.append(train_acc)
        significant_loss_list.append(train_loss)
        epsilon_cesume_list.append(epsilon_consume)
        model.load_state_dict(origin_model_state_dict, strict=False)
        optimizer.load_state_dict(origin_optimizer_state_dict)
    print("check significant_acc_list: {}".format(significant_acc_list))
    print("check significant_loss_list: {}".format(significant_loss_list))
    print("check epsilon_cesume_list: {}".format(epsilon_cesume_list))
    max_significant_loss_index = significant_loss_list.index(min(significant_loss_list))
    target_train_loader = train_loaders[max_significant_loss_index]
    print("target {} / {}".format(max_significant_loss_index, len(significant_acc_list)))
    print("Finishded Profiler!")
    return target_train_loader
'''