import os
import torch
from transformers import AutoModel, AutoTokenizer,AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from taskB_triplet_dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.metrics import accuracy_score
import pynndescent
from torchmetrics.functional import pairwise_cosine_similarity

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def make_triplet(embedding,label,xmsk):
    pos_pair=[]
    neg_pair=[]
    token_embeddings=embedding[0]
    input_mask_expanded=xmsk.unsqueeze(-1).expand(token_embeddings.size()).float()
    embedding= torch.sum(token_embeddings*input_mask_expanded,1)/torch.clamp(input_mask_expanded.sum(1),min=1e-9)  
    cos=pairwise_cosine_similarity(embedding,embedding)
    for i,emb in enumerate(embedding):       
        pos_idxs=np.where(label==label[i])
        neg_idxs=np.where(label!=label[i])

        pos_cos=cos[i][pos_idxs]
        neg_cos=cos[i][neg_idxs]
        hard_pos=torch.argmin(pos_cos)
        hard_neg=torch.argmax(neg_cos)
        pos_pair.append(embedding[pos_idxs[0][hard_pos]].detach().cpu().numpy())
        neg_pair.append(embedding[neg_idxs[0][hard_neg]].detach().cpu().numpy())
    return embedding,pos_pair,neg_pair
        
#2. train
def train_fn(train_dataloader, model,optim,criterion,device):

    model.train()
    total_loss=0.0
   
    for i,batch in enumerate(train_dataloader):
        
        ids=batch['input_ids'].to(device)
        xmsk=batch['attention_mask'].to(device)
        label=batch['label'].to(device)
        
        embedding = model(ids,xmsk)
        
        anchor,positive,negative=make_triplet(embedding,label.detach().cpu().numpy(),xmsk)
        
        loss=criterion(torch.Tensor(anchor).to(device),torch.Tensor(positive).to(device),torch.Tensor(negative).to(device))
        total_loss+=loss.item()
        loss.backward()
        optim.step()
        optim.zero_grad()
               
        pos_cos=nn.functional.cosine_similarity(anchor,torch.Tensor(positive).to(device))
        neg_cos=nn.functional.cosine_similarity(anchor,torch.Tensor(negative).to(device))
        #print(f"pos_cos: : {torch.mean(pos_cos)}, neg_cos {torch.mean(neg_cos)}")
        if i%100==0:
            print(i, total_loss)
            
    avg_train_loss=total_loss/len(train_dataloader)
    return avg_train_loss

#3. valid
def valid_fn(valid_dataloader, criterion, model,device):

    model.eval()
    total_loss=0.0
    
    with torch.no_grad():
        for batch in valid_dataloader:
            
            ids=batch['input_ids'].to(device)
            xmsk=batch['attention_mask'].to(device)
            label=batch['label'].to(device)
            
            
            embedding = model(ids,xmsk)
        
            anchor,positive,negative=make_triplet(embedding,label.detach().cpu().numpy(),xmsk)
            loss=criterion(torch.Tensor(anchor).to(device),torch.Tensor(positive).to(device),torch.Tensor(negative).to(device))
            total_loss+=loss.item()
            
            pos_cos=nn.functional.cosine_similarity(anchor,torch.Tensor(positive).to(device))
            neg_cos=nn.functional.cosine_similarity(anchor,torch.Tensor(negative).to(device))
            print(f"pos_cos: : {torch.mean(pos_cos)}, neg_cos {torch.mean(neg_cos)}")
            #for c,l in zip(cos, label):
                #print(f"({c},{l})")

            """
            for i,pred in enumerate(preds):
                if pred==label[i]:  
                    acc+=1
            """
           
                   
        avg_valid_loss=total_loss/len(valid_dataloader)
        return total_loss,avg_valid_loss
    
    
#4. experiment

def experment_fn(train_dataloader,valid_dataloader, model_name, device,batch_size,lr):
    
    epochs=10
    model = AutoModel.from_pretrained(model_name).to(device)
    criterion=nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    #criterion=nn.TripletMarginWithDistanceLoss()
    optimizer=AdamW(model.parameters(),lr=lr,eps=1e-8)

    for ep in range(epochs):
        
        best_loss=float('inf')
        
        avg_train_loss=train_fn(train_dataloader,model,optimizer,criterion,device)
        
    
        
        print(f"avg_train_loss : {avg_train_loss:.2f}\n")
        save_dict={"model_state_dict":model.state_dict(),
                   "optimizer_state_dict":optimizer.state_dict()}
        
        
        
        valid_loss,avg_valid_loss=valid_fn(valid_dataloader,criterion, model,device)
        
        if best_loss>avg_valid_loss:
            best_loss=avg_valid_loss
            torch.save(save_dict,"/home/labuser/Semeval/Triplet/model/"+str(batch_size)+"_"+str(lr)+"_"+str(best_loss)+"tmwd.pt")
            print(f"Ep: {ep}, best_acc : {best_loss}\n")
            

if __name__=="__main__":
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

    MODEL_NAME='sentence-transformers/paraphrase-distilroberta-base-v1' 
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    
    
    
    train_df=pd.read_json("/home/labuser/Semeval/Data/SubtaskB/subtaskB_train.jsonl",lines=True)
    valid_df=pd.read_json("/home/labuser/Semeval/Data/SubtaskB/subtaskB_dev.jsonl",lines=True)

    train_data=train_df['text'].tolist()
    valid_data=valid_df['text'].tolist()

    train_label=train_df['label'].values
    valid_label=valid_df['label'].values


    train_dataset=Dataset(train_data,train_label,tokenizer)  
    valid_dataset=Dataset(valid_data,valid_label,tokenizer)
    

    batch_size=16
    lr=1e-5
    seed=42
    set_seed(seed)
    #train_sampler = StratifiedSampler(class_vector=train_label, batch_size=batch_size)
    #valid_sampler= StratifiedSampler(class_vector=valid_label, batch_size=batch_size)
    train_dataloader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
    valid_dataloader=DataLoader(valid_dataset,shuffle=True,batch_size=batch_size)



    experment_fn(train_dataloader,valid_dataloader, MODEL_NAME, device,batch_size,lr)
    