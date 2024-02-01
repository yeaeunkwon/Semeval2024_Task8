import numpy as np
import pandas as pd
import torch
import spacy
import urllib
import re
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import TensorDataset,Dataset,DataLoader
import spacy

              
class Dataset(Dataset):

    def __init__(self,data,tokenizer):
        """
        Make Dataset for inputting data to the model
        :param file_path : the path of the data in local
        :tokenizer : the tokenizer to encode the data

        """
        super().__init__()
        self.input1=data['text1'].tolist()
        self.input2=data['text2'].tolist()
        self.label=data['label'].values
        self.class1=data['class1'].tolist()
        self.class2=data['class2'].tolist()
        
        self.tokenizer=tokenizer
        



    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        """
        Index a data tuple to encode it
        :param idx : index
        :return a dictionary that consists of encoded data, attention mask, and target value

        """
        text1=self.input1[idx]
        text2=self.input2[idx]
        ys=self.label[idx]
        target1=self.class1[idx]
        target2=self.class2[idx]
        encode1=self.tokenizer(text1, max_length=128,padding='max_length', truncation=True, return_tensors="pt")
        encode2=self.tokenizer(text2, max_length=128,padding='max_length', truncation=True, return_tensors="pt")
        

        return {'input_ids1': encode1['input_ids'].flatten(),
                'attention_mask1': encode1['attention_mask'].flatten(),
                'input_ids2': encode2['input_ids'].flatten(),
                'attention_mask2': encode2['attention_mask'].flatten(),
                'label':torch.tensor(ys),
                'target1': torch.tensor(target1),
                'target2': torch.tensor(target2)}
       