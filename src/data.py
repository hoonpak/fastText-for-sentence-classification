import re
import csv
import sys
import torch
import numpy as np

from tqdm import tqdm
from torch import LongTensor
from torch.utils.data import Dataset

csv.field_size_limit(sys.maxsize)

class GetTrainTestData():
    def __init__(self, file_path, max_length = 256, sogou = False, bigram = False):
        self.sogou = sogou
        
        train_path = file_path + "/train.csv"
        test_path = file_path + "/test.csv"
        
        self.word2id = {"<pad>":0}
        # self.id2word = {0:"<pad>"}
        self.id = 1
        
        self.max_length = max_length
        
        if bigram:
            self.train_x, self.train_bigram_x, self.train_y = self.get_csv_bigram(train_path, True)
            self.test_x, self.test_bigram_x, self.test_y = self.get_csv_bigram(test_path, False)
        else:
            self.train_x, self.train_y = self.get_csv(train_path, True)
            self.test_x, self.test_y = self.get_csv(test_path, False)
        
    def get_csv_bigram(self, file_path, train):
        inputs = list()
        bigrams = list()
        labels = list()
        
        desc = "Loading test data... "
        if train:
            desc = "Loading train data... "
            
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            csv_train = csv.reader(file)
            for row in tqdm(csv_train, desc=desc):
                if len(row) == 3:
                    label, header, main = row
                    input = " ".join([header, main])
                elif len(row) == 2:
                    label, input = row
                elif len(row) == 4:
                    label, temp1, temp2, temp3 = row
                    input = " ".join([temp1, temp2, temp3])
                else:
                    raise ValueError("length is out of range (2,3,4)")
                    
                input = self.clean_sentence(input)
                input = input[:self.max_length]
                bigram = self.get_bigram(input)
                    
                if train:    
                    for word in input:
                        if word not in self.word2id: 
                            self.word2id[word] = self.id
                            # self.id2word[self.id] = word
                            self.id += 1
                    inputs.append(list(map(lambda x:self.word2id[x], input)))
                    for bi in bigram:
                        if bi not in self.word2id:
                            self.word2id[bi] = self.id
                            # self.id2word[self.id] = bi
                            self.id += 1
                    bigrams.append(list(map(lambda x:self.word2id[x], bigram)))
                else:
                    inputs.append([self.word2id[x] for x in input if x in self.word2id])
                    bigrams.append([self.word2id[x] for x in bigram if x in self.word2id])
                
                labels.append(int(label))
            
        return inputs, bigrams, labels
    
    def get_csv(self, file_path, train):
        inputs = list()
        labels = list()
        
        desc = "Loading test data... "
        if train:
            desc = "Loading train data... "
            
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            csv_train = csv.reader(file)
            for row in tqdm(csv_train, desc=desc):
                if len(row) == 3:
                    label, header, main = row
                    input = " ".join([header, main])
                elif len(row) == 2:
                    label, input = row
                elif len(row) == 4:
                    label, temp1, temp2, temp3 = row
                    input = " ".join([temp1, temp2, temp3])
                else:
                    raise ValueError("length is out of range (2,3,4)")
                    
                input = self.clean_sentence(input)
                # if len(input) > self.max_length:
                #     self.max_length = len(input)
                input = input[:self.max_length]
                    
                if train:    
                    for word in input:
                        if word not in self.word2id: 
                            self.word2id[word] = self.id
                            # self.id2word[self.id] = word
                            self.id += 1
                    inputs.append(list(map(lambda x:self.word2id[x], input)))
                else:
                    inputs.append([self.word2id[x] for x in input if x in self.word2id])
                
                labels.append(int(label))
            
        return inputs, labels
    
    def clean_sentence(self, sentence):
        if self.sogou:
            sentence = ' ' + sentence
            sentence = re.sub(r"[^A-Za-z0-9(),.!?]", " ", sentence)     
            sentence = re.sub(r",", " , ", sentence) 
            sentence = re.sub(r"\.", " . ", sentence)
            sentence = re.sub(r"!", " ! ", sentence) 
            sentence = re.sub(r"\(", " ( ", sentence) 
            sentence = re.sub(r"\)", " ) ", sentence) 
            sentence = re.sub(r"\?", " ? ", sentence) 
            sentence = re.sub(r" \d+", " ", sentence)
            sentence = re.sub(r"\s{2,}", " ", sentence) 
        else:
            sentence = re.sub(r"[^A-Za-z()\.\,\!\?\"\']", " ", sentence)
            sentence = re.sub(r"\(", " ( ", sentence)
            sentence = re.sub(r"\)", " ) ", sentence)
            sentence = re.sub(r"\.", " . ", sentence)
            sentence = re.sub(r"\,", " , ", sentence)
            sentence = re.sub(r"\!", " ! ", sentence)
            sentence = re.sub(r"\?", " ? ", sentence)
            sentence = re.sub(r"\"", " \" ", sentence)
            sentence = re.sub(r"\'", " \' ", sentence)
            sentence = re.sub(r"\s{2,}", " ", sentence)
        return sentence.lower().split()
    
    def get_bigram(self, text):
        bigram = list()
        length = len(text)
        if length <= 1:
            return bigram
        for head in range(length-1):
            tail = head + 1
            bigram.append(text[head]+text[tail])
        return bigram

class ClassificationDataset(Dataset):
    def __init__(self, data_x, data_y, max_length, label_to_id, bigram_x = None):
        self.data_x = data_x
        self.data_y = data_y
        self.bigram_x = bigram_x
        self.max_length = max_length
        self.label_to_id = label_to_id
        
    def __len__(self):
        return len(self.data_y)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = self.data_x[idx]
        x = x+[0]*(self.max_length-len(x))
        if self.bigram_x != None:
            bigram_x = self.bigram_x[idx]
            bigram_x = bigram_x+[0]*(self.max_length-len(bigram_x))
            x += bigram_x
        y = self.data_y[idx]
        y = self.label_to_id[y]
        
        return LongTensor(x), y