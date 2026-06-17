# Installing of Libraries
import numpy as np
import pandas as pd
import random
import json
import torch
import torch.nn as nn
import nltk
from model import NeuralNet
from nltk.stem.porter import PorterStemmer
nltk.download('punkt_tab')
nltk.download('punkt')
from torch.utils.data import Dataset,DataLoader
from nltk_utils import tokenize,bag_of_words,stemmer
import os, json

import json

with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON loaded successfully!")

    
    
all_words=[]
tags=[]
xy=[]
import json

# Open and load the intents.json file
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# Now you can use intents['intents']


for intent in intents['intents']:
    tag=intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag)) # pair of corresponding tokenized patterns and tag
        
        
# ignore punctuation        
ignore_words = ['?', '.', '!']
all_words_without_punctuation=[]
for w in all_words:
    if w not in ignore_words:
        all_words_without_punctuation.append(w)
        
# remove duplicate patterns and tag and sort them
all_words_without_punctuation=sorted(set(all_words_without_punctuation))
tags=sorted(set(tags))

#Print
print(len(xy), "patterns\n")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words_without_punctuation)

# Training Data
X_train=[] # bag of words of patterned sentence
Y_train=[] # contain labels of tags

for (patterned_sentence,tag) in xy:
    bag=bag_of_words(patterned_sentence,all_words_without_punctuation)
    X_train.append(bag)
    label=tags.index(tag)
    Y_train.append(label)
    
# Convert into numpy array
X_train=np.array(X_train)
Y_train=np.array(Y_train)

# Hyper Parameters
num_epochs=1000
batch_size=8 # Number of samples the DataLoader will return per batch.
learning_rate=0.001   
input_size=len(X_train[0])
hidden_size=8 # hidden layer
output_size=len(tags) # output size=no of tags
print(input_size,output_size)


class chatdataset(Dataset):
    def __init__(self): # initialize the dataset object
        super().__init__()
        self.n_samples=len(X_train)
        self.X_data=X_train
        self.Y_data=Y_train
        
    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]
    def __len__(self):
        return self.n_samples
    
dataset=chatdataset()    
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0) # shuffle : Prevents the model from learning the order of training data , Number of subprocesses used for data loading.num_workers = 0 means data will be loaded in the main thread (safe for Windows and small datasets).
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=NeuralNet(input_size,hidden_size,output_size) # define model

# define loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for [input_batch,output_batch] in train_loader:
        input_batch=input_batch.to(device)
        output_batch=output_batch.to(dtype=torch.long).to(device)
        # Forward Propagation
        actual_output_batch=model(input_batch)
        loss=criterion(actual_output_batch,output_batch)
        # Backward Propagation
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()        # compute gradients via backprop
        optimizer.step()       # update model weights
    
    if((epoch+1)%100==0):
        print(f"{epoch+1}/{num_epochs} : Loss = {loss.item():.4f}")    
        
print(f"Final Loss = {loss.item():.4f}")            
data={
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words_without_punctuation,
    "tags": tags
}
FILE="data.pth"
torch.save(data,FILE)
print(f'training complete. file saved to {FILE}')
