import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer,AdamW
import torch.nn as nn
from taskB_simSCE_dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

model_name="princeton-nlp/sup-simcse-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

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


train_dataloader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
valid_dataloader=DataLoader(valid_dataset,shuffle=True,batch_size=batch_size)

 
# tokenize the input

criterion=nn.CosineEmbeddingLoss(margin=1).to(device)
optimizer=AdamW(model.parameters(),lr=1e-5,eps=1e-8)
epochs=5
# generate the embeddings


for ep in range(epochs):
    model.train()
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
        optimizer.step()
        model.zero_grad()
        
        if i%100==0:
            print(i, total_loss)
    avg_train_loss=total_loss/len(train_dataloader)
    print(" Avergae training loss: {0:.2f}",format(avg_train_loss))
        
    model.eval()
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
            
            cos1=nn.functional.cosine_similarity(embedding1,embedding2)
            print(cos1,label)
