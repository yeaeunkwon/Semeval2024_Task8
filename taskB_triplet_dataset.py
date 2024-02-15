import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,Dataset,DataLoader

class Dataset(Dataset):

    def __init__(self,data,label,tokenizer):
        
        super().__init__()
        self.tokenizer=tokenizer
        self.x=data
        self.y=label

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
      

        xs=self.x[idx]

        ys=self.y[idx]
        encode=self.tokenizer(xs,max_length=128,padding='max_length',truncation=True,return_tensors='pt')
    
        return {'input_ids': encode['input_ids'].flatten(),
                'attention_mask': encode['attention_mask'].flatten(),
                'label':torch.tensor(ys,dtype=torch.long)}
        