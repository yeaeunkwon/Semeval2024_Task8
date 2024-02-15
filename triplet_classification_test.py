import os
import torch
from transformers import AutoModel, AutoTokenizer,AdamW,get_linear_schedule_with_warmup
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score       
from taskB_triplet_dataset import Dataset       
import torch.nn.functional as F 
from triplet_classification import Classifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def test_fn(test_dataloader,model, device):

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
        precision=precision_score(true_label,preds, average='weighted')
        recall=recall_score(true_label,preds,average='weighted')
        f1=f1_score(true_label,preds,average='weighted')   
        return acc,precision,recall,f1
            
        
if __name__=="__main__":
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    
    model_name='sentence-transformers/paraphrase-distilroberta-base-v1' 
    embed=AutoModel.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    
    EMB_MODEL_PATH='/home/labuser/Semeval/Triplet/model/16_1e-05_1.9286644317685289.pt'
    embed.load_state_dict(torch.load(EMB_MODEL_PATH, map_location=device)['model_state_dict'])

    
    TEST_PATH="/home/labuser/Semeval/Data/SubtaskB/subtaskB_test.jsonl"
    testdf=pd.read_json(TEST_PATH,lines=True)
    test_data=testdf['text'].tolist()
    test_label=testdf['label'].values
    test_dataset=Dataset(test_data,test_label,tokenizer)
   

    batch_size=32
    n_labels=6
    lr=1e-5
    seed=42
    embed_size=768
    set_seed(seed)
    
    test_dataloader=DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

    
    # tokenize the input
    saved_model=['Triplet/classification/32_3e-05_0.6563333333333333.pt','Triplet/classification/32_3e-05_0.669.pt','Triplet/classification/32_3e-05_0.6726666666666666.pt',
                 'Triplet/classification/32_3e-05_0.6536666666666666.pt','Triplet/classification/32_3e-05_0.6886666666666666.pt','Triplet/classification/32_3e-05_0.6516666666666666.pt',
                 'Triplet/classification/32_3e-05_0.712.pt','Triplet/classification/32_3e-05_0.668.pt','Triplet/classification/32_3e-05_0.6713333333333333.pt',
                 'Triplet/classification/32_3e-05_0.6953333333333334.pt']
    for i in range(len(saved_model)):
        print(saved_model[i])
        MODEL_PATH="/home/labuser/Semeval/"+saved_model[i]
        model = Classifier(embed,embed_size,n_labels).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
        test_accuracy,precision,recall,f1=test_fn(test_dataloader,model, device)
        print(f"test_accuracy : {test_accuracy},precision : {precision},recall: {recall},f1 : {f1}")