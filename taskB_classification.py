import os
import torch
from transformers import AutoModel, AutoTokenizer,AdamW,get_linear_schedule_with_warmup
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import random
import numpy as np
from lossfn import lossfn_triplet
from sklearn.metrics import accuracy_score        
from classification_dataset import Dataset       
import torch.nn.functional as F       

class Classifier(nn.Module):
   
    def __init__(self,embed,embed_size,n_labels):
        
        super().__init__()
        
        self.emb=embed
        self.fc=nn.Linear(embed_size,n_labels)
        self.dropout=nn.Dropout(0.5)
        #self.relu=F.relu()
    

    def forward(self,ids,xmsk):
    
        model_embed=self.emb(ids,xmsk, output_hidden_states=True, 
                            return_dict=True).pooler_output
        out=self.fc(model_embed)
        
        
        return out
 
    

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
    total_loss=0.0
    preds=[]
    true_label=[]
    for i,batch in enumerate(train_dataloader):
        
        ids=batch['input_ids'].to(device)
        xmsk=batch['attention_mask'].to(device)
        label=batch['label'].to(device)
        
        output=model(ids,xmsk)
    
        loss=criterion(output,label)
        total_loss+=loss.item()
        loss.backward()
        optim.step()
        optim.zero_grad()
              
        pred=torch.argmax(F.softmax(output),axis=1)
        preds.extend(pred.detach().cpu().numpy())
                    
        true_label.extend(label.detach().cpu().numpy())
        if i%100==0:
            print(i, total_loss)
            
    avg_train_loss=total_loss/len(train_dataloader)
    acc=accuracy_score(true_label,preds)
    return total_loss,avg_train_loss,acc

#3. valid
def valid_fn(valid_dataloader, criterion, model,device):

    model.eval()
    total_loss=0.0
    acc=0
    true_label=[]
    preds=[]
    with torch.no_grad():
        for batch in valid_dataloader:
            ids=batch['input_ids'].to(device)
            xmsk=batch['attention_mask'].to(device)
            label=batch['label'].to(device)
        
            output=model(ids,xmsk)
    
            loss=criterion(output,label)
            total_loss+=loss.item()
            
    
            pred=torch.argmax(F.softmax(output),axis=1)
            #print(pred, label)
            preds.extend(pred.detach().cpu().numpy())
           
            true_label.extend(label.detach().cpu().numpy())
                   
        avg_valid_loss=total_loss/len(valid_dataloader)
        acc=accuracy_score(true_label,preds)
        return total_loss,avg_valid_loss,acc
    

    
#4. experiment

def experment_fn(train_dataloader,valid_dataloader, embed, device,batch_size,lr,n_labels):
    
    embed_size=768
    epochs=10
    model = Classifier(embed,embed_size,n_labels).to(device)
    optimizer=AdamW(model.parameters(),lr=lr,eps=1e-8)
    criterion=nn.CrossEntropyLoss().to(device)

    for ep in range(epochs):
        
        best_acc=0
        
        train_loss,avg_train_loss,train_acc=train_fn(train_dataloader,model,optimizer,criterion,device)
        
        
        print(f"Ep: {ep}, train_acc : {train_acc}\n")
        
        print(f"train_loss,avg_train_loss : {train_loss:.2f},{avg_train_loss:.2f}\n")
        save_dict={"model_state_dict":model.state_dict(),
                   "optimizer_state_dict":optimizer.state_dict()}
        
        
        
        valid_loss,avg_valid_loss,valid_acc=valid_fn(valid_dataloader,criterion, model,device)
        
        if best_acc<valid_acc:
            best_acc=valid_acc
            torch.save(save_dict,"/home/labuser/Semeval/Simcse/classfier_model/"+str(batch_size)+"_"+str(lr)+"_"+str(best_acc)+".pt")
            print(f"Ep: {ep}, best_acc : {best_acc}\n")
            
            

if __name__=="__main__":
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

    model_name="princeton-nlp/sup-simcse-bert-base-uncased"
    embed=AutoModel.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    
    
    MODEL_PATH="/home/labuser/Semeval/model/16_1e-05_0.7153333333333334_arc3.pt"
    embed.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
    
    
    TRAIN_PATH="/home/labuser/Semeval/Data/SubtaskB/subtaskB_train.jsonl"
    VALID_PATH="/home/labuser/Semeval/Data/SubtaskB/subtaskB_dev.jsonl"
    TEST_PATH="/home/labuser/Semeval/Data/SubtaskB/subtaskB_test.jsonl"
  
    train_dataset=Dataset(TRAIN_PATH,tokenizer)  
    valid_dataset=Dataset(VALID_PATH,tokenizer)
    test_dataset=Dataset(TEST_PATH,tokenizer)

    batch_size=32
    n_labels=6
    lr=1e-5
    seed=42
    embed_size=768
    set_seed(seed)
    
    train_dataloader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
    valid_dataloader=DataLoader(valid_dataset,shuffle=True,batch_size=batch_size)
    test_dataloader=DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

    
    # tokenize the input

    experment_fn(train_dataloader,valid_dataloader,embed, device,batch_size,lr,n_labels)
  
