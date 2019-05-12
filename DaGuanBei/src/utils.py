import pandas as pd
import random
from collections import Counter
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def self_sample_by_class(train_file, ratio):
    train_index, val_index = [], []
    df = pd.read_csv(train_file)
    df.set_index('id', inplace = True)
    class_item = Counter(df['class'].tolist())
    
    for k in class_item:
        tmp = df[df['class'] == k].index.tolist()
        random.shuffle(tmp)
        n_length = len(tmp)
        for i in range(n_length):
            if i < int(ratio * n_length):
                train_index.append(tmp[i])
            else:
                val_index.append(tmp[i])
    train_set = df.loc[train_index]
    val_set = df.loc[val_index]

    return train_set, val_set

def transform_fasttext(data_set, fasttext_file, column):
    labels = (data_set['class'] - 1).tolist()
    texts = (data_set[column]).tolist()
    string_list = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        string = '__label__{} , {}\n'.format(label, text)
        string_list.append(string)
    write_data(''.join(string_list), fasttext_file)

def write_data(string_list, save_file):
    with open(save_file, 'w') as f:
        f.write(string_list)
