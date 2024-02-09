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
from taskB_classification import Classifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def test_fn(test_dataloader,embed,embed_size,n_labels,model, device):

    model.eval()

    acc=0
    true_label=[]
    preds=[]
    with torch.no_grad():
        for batch in test_dataloader:
            ids=batch['input_ids'].to(device)
            xmsk=batch['attention_mask'].to(device)
            label=batch['label'].to(device)
        
            output=model(ids,xmsk)
            
    
            pred=torch.argmax(F.softmax(output),axis=1)
            #print(pred, label)
            preds.extend(pred.detach().cpu().numpy())
           
            true_label.extend(label.detach().cpu().numpy())
        acc=accuracy_score(true_label,preds)
        return acc
            
        
if __name__=="__main__":
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

    model_name="princeton-nlp/sup-simcse-bert-base-uncased"
    embed=AutoModel.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    
    
    EMB_MODEL_PATH="/home/labuser/Semeval/model/16_1e-05_0.7091666666666666.pt"
    embed.load_state_dict(torch.load(EMB_MODEL_PATH, map_location=device)['model_state_dict'])
    
    
    TEST_PATH="/home/labuser/Semeval/Data/SubtaskB/subtaskB_test.jsonl"
  
    test_dataset=Dataset(TEST_PATH,tokenizer)

    batch_size=32
    n_labels=6
    lr=1e-5
    seed=42
    embed_size=768
    set_seed(seed)
    
    test_dataloader=DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

    
    # tokenize the input

    MODEL_PATH=""
    model = Classifier(embed,embed_size,n_labels).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
    test_accuracy=test_fn(test_dataloader,embed,embed_size,n_labels,model, device)
    print(f"test_accuracy : {test_accuracy}")