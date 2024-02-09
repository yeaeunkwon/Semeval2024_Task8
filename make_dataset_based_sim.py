import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer,AdamW
import torch.nn as nn
from taskB_simSCE_dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pynndescent
import json


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

model_name="princeton-nlp/sup-simcse-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
 

TRAIN_PATH="/home/labuser/Semeval/Data/SubtaskB/subtaskB_train.jsonl"
VALID_PATH="/home/labuser/Semeval/Data/SubtaskB/subtaskB_dev.jsonl"

train_data=pd.read_json(TRAIN_PATH,lines=True)
valid_data=pd.read_json(VALID_PATH,lines=True)


n_labels=6

def make_pospair(dataset,n_labels): # the least similar pair
    pos_pair=[]
    text=dataset['text'].tolist()
    label=dataset['label'].values
    for i in range(n_labels):       
        idxs=np.where(label==i)
        idx_embedding=[]
        #print(f"indexs for label {i}: {idxs[0]}")
        for idx in idxs[0][:3000]:
            input=tokenizer(text[idx],max_length=128,padding='max_length', truncation=True, return_tensors="pt").to(device)
            embedding = model(**input, output_hidden_states=True, 
                        return_dict=True).pooler_output
            idx_embedding.append(embedding.detach().cpu().numpy())
        idx_embedding=torch.squeeze(torch.tensor(idx_embedding),1)
        print(idx_embedding.shape)
        index=pynndescent.NNDescent(idx_embedding,metric='cosine')
        index.prepare()
        neighbors=index.query(idx_embedding,k=2000)
    
        for j,neighbor in enumerate(neighbors[0]):
            row={'text1':[],'text2':[],'label':[],'class1':[],'class2':[]}
            row['text1']=text[idxs[0][j]]
            row['text2']=text[idxs[0][neighbor[-1]]]
            row['label']=1
            row['class1']=i
            row['class2']=i
            pos_pair.append(row)
    return pos_pair

def make_negpair(dataset,n_labels):
    neg_pair=[]
    text=dataset['text'].tolist()
    label=dataset['label'].values
    for i in range(n_labels):
            idxs=np.where(label==i)
            idx_embedding=[]
            for idx in idxs[0][:2000]:
                input=tokenizer(text[idx],max_length=128,padding='max_length', truncation=True, return_tensors="pt").to(device)
                embedding = model(**input, output_hidden_states=True, 
                        return_dict=True).pooler_output
                idx_embedding.append(embedding.detach().cpu().numpy())
            idx_embedding=torch.squeeze(torch.tensor(idx_embedding),1)
            
            index=pynndescent.NNDescent(idx_embedding,metric='cosine')
            index.prepare()
            
            
            for k in range(n_labels):
                if i>=k:
                    continue
                c_idxs=np.where(label==k)
                c_embeddings=[]
                for c_idx in c_idxs[0][:1200]:
                    c_input=tokenizer(text[c_idx],max_length=128,padding='max_length', truncation=True, return_tensors="pt").to(device)
                    c_embedding = model(**c_input, output_hidden_states=True, 
                                      return_dict=True).pooler_output
                    c_embeddings.append(c_embedding.detach().cpu().numpy())
                c_embeddings=torch.squeeze(torch.tensor(c_embeddings),1)
                print(c_embeddings.shape)
                c_neighbors=index.query(c_embeddings,k=50)
                for j, neighbor in enumerate(c_neighbors[0]):
                    row={'text1':[],'text2':[],'label':[],'class1':[],'class2':[]}
                    row['text1']=text[c_idxs[0][j]]
                    row['text2']=text[idxs[0][neighbor[0]]]
                    row['label']=-1
                    row['class1']=k
                    row['class2']=i
                    #print(label[c_idxs[0][j]],label[idxs[0][neighbor[0]]])
                    neg_pair.append(row)
    return neg_pair


def make_negpair_archi(dataset,n_labels):
    neg_pair=[]
    text=dataset['text'].tolist()
    label=dataset['label'].values
    for i in range(n_labels-2):
            idxs=np.where(label==i)
            for j in range(i+1,n_labels-1):
                c_idxs=np.where(label==j)
                for idx,c_idx in zip(idxs[0][:240],c_idxs[0][:240]): #곱하기 10
                    row={'text1':[],'text2':[],'label':[],'class1':[],'class2':[]}
                    row['text1']=text[c_idx]
                    row['text2']=text[idx]
                    row['label']=-1
                    row['class1']=j
                    row['class2']=i
                    #print(label[c_idx],label[idx])
                    neg_pair.append(row)
    print(len(neg_pair))             
    idxs=np.where(label==5)
    idx_embedding=[]
    for idx in idxs[0][:240]:
        input=tokenizer(text[idx],max_length=128,padding='max_length', truncation=True, return_tensors="pt").to(device)
        embedding = model(**input, output_hidden_states=True, 
                return_dict=True).pooler_output
        idx_embedding.append(embedding.detach().cpu().numpy())
    idx_embedding=torch.squeeze(torch.tensor(idx_embedding),1)
            
    index=pynndescent.NNDescent(idx_embedding,metric='cosine')
    index.prepare()
    for k in range(5):
        c_idxs=np.where(label==k)
        c_embeddings=[]
        for c_idx in c_idxs[0][:120]: #곱하기 5
            c_input=tokenizer(text[c_idx],max_length=128,padding='max_length', truncation=True, return_tensors="pt").to(device)
            c_embedding = model(**c_input, output_hidden_states=True, 
                                    return_dict=True).pooler_output
            c_embeddings.append(c_embedding.detach().cpu().numpy())
        c_embeddings=torch.squeeze(torch.tensor(c_embeddings),1)
        c_neighbors=index.query(c_embeddings,k=100)
        for j, neighbor in enumerate(c_neighbors[0]):
            row={'text1':[],'text2':[],'label':[],'class1':[],'class2':[]}
            row['text1']=text[c_idxs[0][j]]
            row['text2']=text[idxs[0][neighbor[0]]]
            row['label']=-1
            row['class1']=k
            row['class2']=5
            #print(k,label[c_idxs[0][j]],label[idxs[0][neighbor[0]]])
            neg_pair.append(row)
    return neg_pair

#def make_triplet_dataest(data,n_labels):
    

"""
pos_pair=make_pospair(train_data,n_labels)
print(len(pos_pair))
with open("train_pospair_sim2.jsonl","w") as f:
    json.dump(pos_pair,f)
  

neg_pair=make_negpair(train_data,n_labels)
print(len(neg_pair))
with open("train_negpair_sim2.jsonl","w") as f:
    json.dump(neg_pair,f)



pos_pair=make_pospair(valid_data,n_labels)
print(len(pos_pair))
with open("valid_pospair_sim_arc2.jsonl","w") as f:
    json.dump(pos_pair,f)


neg_pair=make_negpair(valid_data,n_labels)
print(len(neg_pair))
with open("valid_negpair_sim2.jsonl","w") as f:
    json.dump(neg_pair,f)


neg_pair=make_negpair_archi(train_data,n_labels)
print(len(neg_pair))
with open("train_negpair_sim_arc3.jsonl","w") as f:
     json.dump(neg_pair,f)
   

neg_pair=make_negpair_archi(valid_data,n_labels)
print(len(neg_pair))
with open("valid_negpair_sim_arc3.jsonl","w") as f:
    json.dump(neg_pair,f)
    """
  
