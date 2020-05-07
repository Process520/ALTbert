# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-02-12 12:57
import pandas as pd
import os
import codecs
import pickle


# create vocabulary of lables. label is sorted. 1 is high frequency, 2 is low frequency.
def create_vocabulary_label(voabulary_label=''):
    print("create_voabulary_label_sorted.started.training_data_path:", voabulary_label)
    cache_path ='cache_vocabulary_label_pik/' + "label_vocabulary.pik"
    if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
        print('cheng:')
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index_label, vocabulary_index2word_label = pickle.load(data_f)
            return vocabulary_word2index_label, vocabulary_index2word_label



# 读取txt文件
def read_txt_file(file_path):
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     content = [_.strip() for _ in f.readlines()]

    # labels, texts = [], []
    # for line in content:
    #     parts = line.split()
    #     label, text = parts[0], ''.join(parts[1:])
    #     labels.append(label)
    #     texts.append(text)

    # return labels, texts

    print("load_data.started...")
    file = codecs.open(file_path, 'r', 'utf8')
    lines = file.readlines()
    X = []
    Y = []

    for i, line in enumerate(lines):
        x, y = line.split('__label__')
        y = y.replace('\n', '')
        x = x.replace("\t", '。').replace(" ", '').strip('')
        if i < 5:
            print("x0:", x) #get raw x

        # x = x.split(" ")
        # x = [vocabulary_word2index.get(e, 0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        # if i < 5:
        #     print("x1:",x) #word to index
        # y = vocabulary_label2index[y]
        X.append(x)
        Y.append(y)

    print("load_data.ended...")
    print("dataset examples:", len(X))
    return Y, X

labels_path = 'data/allLables.txt'
file_path = 'data/train.txt'

vocabulary_label2index, vocabulary_index2label = create_vocabulary_label(labels_path)

labels, texts = read_txt_file(file_path)
labelsIndex = [vocabulary_label2index[lable] for lable in labels]
train_df = pd.DataFrame({'label': labelsIndex, 'text': texts})

file_path = 'data/test.txt'

labels, texts = read_txt_file(file_path)
labelsIndex = [vocabulary_label2index[lable] for lable in labels]
test_df = pd.DataFrame({'label': labelsIndex, 'text': texts})

file_path = 'data/dev.txt'

labels, texts = read_txt_file(file_path)
labelsIndex = [vocabulary_label2index[lable] for lable in labels]
dev_df = pd.DataFrame({'label': labelsIndex, 'text': texts})

print(train_df.head())
print(test_df.head())
print(dev_df.head())

train_df['text_len'] = train_df['text'].apply(lambda x: len(x))
# print('cheng:')
# print(train_df.head())
print(train_df.describe())

