import torch
from torch.utils.data import DataLoader, TensorDataset
import time
from utils.data_loader import extract_data, pad_input
from utils.training_functions import privacy_model_train_valid
import pandas as pd
import numpy as np
import nltk
from collections import Counter

def do_calculate_func(job_id, model_name, train_dataset_raw_paths, test_dataset_raw_path,
                    dataset_name, label_type, selected_datablock_identifiers, not_selected_datablock_identifiers,
                    device, summary_writer_path,
                    LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, 
                    BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPOCHS,
                    label_distributions, train_configs):
    begin_time = int(time.time())
    all_results = {}
    
    train_raw_dataset = pd.concat([pd.read_csv(path) for path in train_dataset_raw_paths])
    train_raw_dataset = train_raw_dataset.sort_values(by=['year', 'month', 'day']).reset_index(drop=True)
    test_raw_dataset = pd.read_csv(test_dataset_raw_path)

    # nltk.download('punkt') # Tokenizer
    words = Counter()

    assert label_type == 'sentiment'
    train_sentences, train_sentiment, _, _, _, _ = extract_data(train_raw_dataset)
    test_sentences, test_sentiment, _, _, _, _  = extract_data(test_raw_dataset)

    train_sentences = list(filter(None, train_sentences))
    test_sentences = list(filter(None, test_sentences))
    train_sentences = [sen for sen in train_sentences if sen is not np.nan]
    test_sentences = [sen for sen in test_sentences if sen is not np.nan]
    # print(f'Valid Training Sentences: {len(train_sentences)}')
    # print(f'Valid Test Sentences: {len(test_sentences)}')

    for i, sentence in enumerate(train_sentences):
        try:
            #tokens = nltk.word_tokenize(sentence)
            tokens = nltk.regexp_tokenize(sentence, pattern="\s+", gaps = True)
            train_sentences[i] = []
            for word in tokens: # Tokenize the words
                words.update([word.lower()]) # To Lower Case
                train_sentences[i].append(word)
        except:
            print(sentence)
    # print(f"tokenize 100% done")

    # Remove infrequent words (i.e. words that only appear once)
    words = {k:v for k,v in words.items() if v>1}
    # Sort the words according to frequency, descending order
    words = sorted(words, key=words.get, reverse=True)
    # Add padding & unknown to corpus
    words = ['_PAD','_UNK'] + words

    # Dictionaries for fast mappings
    word2idx = {w:i for i,w in enumerate(words)}
    # Convert word to indices
    for i, sentence in enumerate(train_sentences):
        train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]
        
    for i, sentence in enumerate(test_sentences):
        # For valid sentences, we have to tokenize the sentences as well
        test_sentences[i] = [word2idx[word.lower()] if word in word2idx else 0 for word in nltk.regexp_tokenize(sentence, pattern="\s+", gaps = True)]
    # print(f"word2idx 100% done")

    
    train_sentences = torch.from_numpy(pad_input(train_sentences, train_configs['sequence_length']))
    test_sentences = torch.from_numpy(pad_input(test_sentences, train_configs['sequence_length']))
    
    label_num = len(set(train_sentiment) | set(test_sentiment))
    train_sentiment = torch.from_numpy(np.array(train_sentiment))
    test_sentiment = torch.from_numpy(np.array(test_sentiment))
    all_train_dataset = TensorDataset(
        train_sentences,
        train_sentiment
    )
    test_dataset = TensorDataset(
        test_sentences,
        test_sentiment
    )
    print("load test dataset finished! len: {}".format(len(test_dataset)))
    print("load all dataset finished! len: {}".format(len(all_train_dataset)))

    target_train_loader = DataLoader(all_train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    selected_datablock_ids_str = '['
    for id in selected_datablock_identifiers:
        selected_datablock_ids_str += "{};".format(id)
    selected_datablock_ids_str += ']'
    select_log_str = "SELECT_{}".format(selected_datablock_ids_str)
    
    
    train_configs['vocab_size'] = len(word2idx) + 1
    train_configs['label_distributions'] = label_distributions
    
    summary_writer_keyword = "{}-{}-{}-{}".format(job_id, model_name, dataset_name, select_log_str)
    train_acc, train_loss, test_acc, test_loss, epsilon_consume = privacy_model_train_valid(
        model_name, select_log_str, target_train_loader, test_loader,
        device, label_num, summary_writer_path, summary_writer_keyword,
        LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, MAX_PHYSICAL_BATCH_SIZE, EPOCHS,
        train_configs
    )
    all_results = {
        'train_acc': train_acc,
        'train_loss': train_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'epsilon_consume': epsilon_consume,
    }
    real_duration_time = int(time.time()) - begin_time
    return job_id, all_results, real_duration_time
    # del all_train_dataset, test_dataset
    # torch.cuda.empty_cache() # 删除了一部分, 但是剩下的好像就是pytorch的上下文信息, 无法直接被删除, 估计要启动一个后台的script才可以