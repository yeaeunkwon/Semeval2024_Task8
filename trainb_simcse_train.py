import os
import torch
from transformers import AutoModel, AutoTokenizer,AdamW
import torch.nn as nn
from taskB_simSCE_dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
#2. train
def train_fn(train_dataloader, model,optim,criterion,device):

    model.train()
    optim.zero_grad()
    total_loss=0.0
    for i,batch in enumerate(train_dataloader):
        
        ids_1=batch['input_ids1'].to(device)
        ids_2=batch['input_ids2'].to(device)
        xmsk_1=batch['attention_mask1'].to(device)
        xmsk_2=batch['attention_mask2'].to(device)
        label=batch['label'].to(device)
        
        embedding1 = model(ids_1,xmsk_1, output_hidden_states=True, 
                            return_dict=True).pooler_output
        embedding2 = model(ids_2,xmsk_2, output_hidden_states=True, 
                            return_dict=True).pooler_output
    
        loss=criterion(embedding1,embedding2,target=label)
        total_loss+=loss.item()
        loss.backward()
        optim.step()
        
        if i%100==0:
            print(i, total_loss)
    avg_train_loss=total_loss/len(train_dataloader)
    
    return total_loss,avg_train_loss

#3. valid
def valid_fn(valid_dataloader, criterion, model,device,threshold):

    model.eval()
    total_loss=0.0
    acc=0
    with torch.no_grad():
        for batch in valid_dataloader:
            ids_1=batch['input_ids1'].to(device)
            ids_2=batch['input_ids2'].to(device)
            xmsk_1=batch['attention_mask1'].to(device)
            xmsk_2=batch['attention_mask2'].to(device)
            label=batch['label'].to(device)
            
            
            embedding1 = model(ids_1,xmsk_1, output_hidden_states=True, 
                            return_dict=True).pooler_output
            embedding2 = model(ids_2,xmsk_2, output_hidden_states=True, 
                            return_dict=True).pooler_output
            
            loss=criterion(embedding1,embedding2,target=label)
            total_loss+=loss.item()
            
            cos=nn.functional.cosine_similarity(embedding1,embedding2)
            
            
            preds=[]
            for d in cos:
                if d<threshold:
                    preds.append(-1)
                else:
                    preds.append(1)
            
            for i,pred in enumerate(preds):
                if pred==label[i]:  
                    acc+=1
                    
        avg_valid_loss=total_loss/len(valid_dataloader)
        
        return total_loss,avg_valid_loss,acc
    
    
#4. experiment

def experment_fn(train_dataloader,valid_dataloader, model_name, device,batch_size,lr):
    
    model = AutoModel.from_pretrained(model_name).to(device)
    criterion=nn.CosineEmbeddingLoss(reduction='mean').to(device)
    optimizer=AdamW(model.parameters(),lr=lr,eps=1e-8)
    epochs=10
    threshold=0.8

    for ep in range(epochs):
        
        best_acc=0
        
        train_loss,avg_train_loss=train_fn(train_dataloader,model,optimizer,criterion,device)
        
        print(f"train_loss,avg_train_loss : {train_loss:.2f},{avg_train_loss:.2f}\n")
        save_dict={"model_state_dict":model.state_dict(),
                   "optimizer_state_dict":optimizer.state_dict()}
        
        
        
        valid_loss,avg_valid_loss,valid_acc=valid_fn(valid_dataloader,criterion, model,device,threshold)
        
        if best_acc<=valid_acc:
            best_acc=valid_acc
            torch.save(save_dict,str(batch_size)+"_"+str(lr)+"_"+str(best_acc)+".pt")
            print(f"best_acc : {best_acc:/.2f}\n")
            

if __name__=="__main__":
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

    model_name="princeton-nlp/sup-simcse-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    train_neg=pd.read_json("train_negpair_data.jsonl")
    #train_pos=pd.read_json("train_pospair_data.jsonl")

    valid_neg=pd.read_json("valid_negpair_data.jsonl")
    valid_pos=pd.read_json("valid_pospair_data.jsonl")

    #train_pair=pd.concat([train_pos,train_neg])
    valid_pair=pd.concat([valid_pos,valid_neg])

    #pair= pair.sample(frac=1).reset_index(drop=True)
    #train_df,valid_df=train_test_split(pair,test_size=0.2,stratify=pair['label'])


    train_dataset=Dataset(train_neg,tokenizer)  
    valid_dataset=Dataset(valid_pair,tokenizer)
    print(train_dataset[0])

    batch_size=16
    lr=1e-5

    train_dataloader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
    valid_dataloader=DataLoader(valid_dataset,shuffle=True,batch_size=batch_size)

    
    # tokenize the input

    experment_fn(train_dataloader,valid_dataloader, model_name, device,batch_size,lr)