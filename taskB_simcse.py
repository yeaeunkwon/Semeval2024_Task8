import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer,AdamW
import pandas as pd
from sklearn.utils import shuffle
from taskB_dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import accuracy_score,f1_score


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name="princeton-nlp/sup-simcse-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


neg=pd.read_json("negpair_data.jsonl")
pos=pd.read_json("pospair_data.jsonl")

pair=pd.concat([pos,neg])


train_dataset=Dataset(pair,tokenizer=tokenizer)
valid_dataset=Dataset(pair,tokenizer=tokenizer)
batch_size=32
train_dataloader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
valid_dataloader=DataLoader(valid_dataset,shuffle=True,batch_size=batch_size)

optimizer=AdamW(model.parameters(),lr=2e-5,eps=1e-8)
epochs=3

seed_value=42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

critierion=nn.CosineEmbeddingloss().to(device)

for ep in range(epochs):
    model.train()
    total_loss=0
    for i,batch in enumerate(train_dataloader):
        ids=batch['input_ids'].to(device)
        xmsk=batch['attention_mask'].to(device)
        labels=batch['target'].to(device)
        
        outputs=model(ids,attention_mask=xmsk,labels=labels)
        
        loss=outputs[0]
        
        total_loss+=loss.item()
        
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        if i%1000==0:
            print(i, total_loss)
    
    avg_train_loss=total_loss/len(train_dataloader)
    print(" Avergae training loss: {0:.2f}",format(avg_train_loss))
    
    model.eval()
    preds=[]
    true_labels=[]
    for batch in valid_dataloader:
        ids=batch['input_ids'].to(device)
        xmsk=batch['attention_mask'].to(device)
        labels=batch['target']
        
        with torch.no_grad():
            outputs=model(ids,attention_mask=xmsk)
            print(outputs)
        
        logits=outputs[0]
        
        preds.extend(np.argmax(logits.detach().cpu().numpy(),axis=1).flatten())
        true_labels.extend(labels.to('cpu').numpy().flatten())
    
    print(preds, true_labels)    
    print("macro ",f1_score(true_labels, preds,average="macro",zero_division=0))
    print("micro ",f1_score(true_labels, preds,average="micro",zero_division=0))
    print("accuracy ", accuracy_score(true_labels, preds))

        