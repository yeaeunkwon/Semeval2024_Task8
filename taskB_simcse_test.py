import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer,AdamW
import torch.nn as nn


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Import the pretrained models and tokenizer, this will also download and import th
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
 
# input sentence
text1 ="I am writing an article"
text2="Writing an article on Machine learning"
text3="I am not writing."
text4="the article on machine learning is already written"

 
# tokenize the input
input1 = tokenizer(text1, padding='max_length', truncation=True, return_tensors="pt").to(device)
input2 = tokenizer(text2, padding='max_length', truncation=True, return_tensors="pt").to(device)
input3 = tokenizer(text3, padding='max_length', truncation=True, return_tensors="pt").to(device)
input4 = tokenizer(text4, padding='max_length', truncation=True, return_tensors="pt").to(device)

critierion=nn.CosineEmbeddingLoss().to(device)
optimizer=AdamW(model.parameters(),lr=2e-5,eps=1e-8)
epochs=10
# generate the embeddings
model.train()

for ep in range(epochs):
    embedding1 = model(**input1, output_hidden_states=True, 
                        return_dict=True).pooler_output
    embedding2 = model(**input2, output_hidden_states=True, 
                        return_dict=True).pooler_output
    embedding3 = model(**input3, output_hidden_states=True, 
                        return_dict=True).pooler_output
    
    
    loss=critierion(embedding1,embedding2,target=torch.Tensor([1]).to(device))
    print("loss ", loss)
    loss.backward()
    optimizer.step()
    model.zero_grad()
    
    cos1=nn.functional.cosine_similarity(embedding1,embedding2)
    cos2=nn.functional.cosine_similarity(embedding1,embedding3)
    cos3=nn.functional.cosine_similarity(embedding3,embedding2)
    print(cos1,cos2,cos3)
    