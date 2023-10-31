import torch
from torch import nn
import pandas as pd 
import re
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import Counter


# read the raw data
def _read_data():
	# read in files
	train_data = pd.read_csv("training_raw_data.csv")
	test_data = pd.read_csv("test_raw_data.csv")

	return train_data, test_data


# data preprocessing
def _data_preprocessing(input_data):
    input_data['clean_text'] = input_data['Content'].map(lambda x: re.sub(r'[^\w\s]',' ',x))

    # make lower case
    input_data['clean_text'] = input_data["clean_text"].apply(lambda x: x.lower())

    return input_data


# stats of length of sentences
def _stats_seq_len(input_data):
    input_data['seq_words'] = input_data['clean_text'].apply(lambda x: x.split())
    input_data['seq_len'] = input_data['seq_words'].apply(lambda x: len(x))
    # input_data['seq_len'].hist()
    # plt.show()
    print(input_data['seq_len'].describe())
    
    # remove short and long tokens
    min_seq_len = 100
    max_seq_len = 600
    input_data = input_data[min_seq_len <= input_data['seq_len']]
    input_data = input_data[input_data['seq_len'] <= max_seq_len]

    return input_data



# convert 'positive and negative' to labels 1 and 0
def _convert_labels(input_data):

	# convert positive = 1, negative = 0
	input_data['Label'] = input_data["Label"].apply(lambda x: 0 if x == "neg" else 1)

	return input_data



# Tokenize: create Vocab to Index mapping dictionary
def _map_tokens2index(input_data,top_K = 500):
    words = input_data['seq_words'].tolist()
    tokens_list = []
    for l in words:
        tokens_list.extend(l)
    
    # count  frequency of words
    count_tokens = Counter(tokens_list)

    # dictionary = {words: count}
    sorted_tokens = count_tokens.most_common(len(tokens_list))

    # choose the top K tokens or all of them
    tokens_top = sorted_tokens[:top_K]
    
    # tokens to index staring from 2, index=0:<padding>, index=1:<unknown>
    tokens2index = {w:i+2 for i, (w,c) in enumerate(tokens_top)}

    # add index values for padding/unknown values
    tokens2index['<pad>'] = 0
    tokens2index['<unk>'] = 1
    
    # save tokens2index as json file
    with open('tokens2index.json', 'w') as outfile:
        json.dump(tokens2index, outfile,indent=4)
    
    return tokens2index



# Encode the words in sentences to index
def _encode_word2index(x,tokens2index):
	## unknown words: index=1
    input_tokens = [tokens2index.get(w,1) for w in x]

    return input_tokens


# Pad/truncate sequence data
def _pad_truncate_seq(x,seq_len):
    if len(x) >= seq_len:
        return x[:seq_len]
    else:
    	# add padding if not long enough
        return x + [0] * (seq_len - len(x))


def main():
	# read raw data from files [to do]
 # print("hi :3")
	train_data, test_data = _read_data()

	# clean data, make lower case
	train_data = _data_preprocessing(train_data)
	test_data = _data_preprocessing(test_data)

	# get stats of length and remove short and long sentences
	train_data = _stats_seq_len(train_data)
	test_data = _stats_seq_len(test_data)
	
	# convert to string labels to boolean 1 and 0
	train_data = _convert_labels(train_data)
	test_data = _convert_labels(test_data)
	# print("test", test_data['Label'][:20])

	
	# Tokenize: create Vocab to Index mapping dictionary [to do]
	top_K = 10000 ## try 8000-10000 words
	tokens2index = _map_tokens2index(train_data,top_K)
	print("num of tokens", len(tokens2index))

	
	# Encode the words in sentences to index
	train_data['input_x'] = train_data['seq_words'].apply(lambda x: _encode_word2index(x,tokens2index))
	test_data['input_x'] = test_data['seq_words'].apply(lambda x: _encode_word2index(x,tokens2index))
	print(test_data['input_x'][:10])
	
	
	# Pad/truncate sequence data, save results
	batch_seq_len = 150  ## recommend 150-300
	# pad the training data
	train_data['input_x'] = train_data['input_x'].apply(lambda x: _pad_truncate_seq(x,batch_seq_len))
	train_data.to_csv('training_data.csv', index=False)

	# pad test data
	test_data['input_x'] = test_data['input_x'].apply(lambda x: _pad_truncate_seq(x,batch_seq_len))
	test_data.to_csv('test_data.csv', index=False)
	
if __name__ == '__main__':
	  main()
    


