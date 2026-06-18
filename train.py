# Installing of Libraries
import numpy as np
import json
import torch
import torch.nn as nn
import nltk
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
nltk.download('punkt_tab')
nltk.download('punkt')
# Load intents
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []


# Create training data
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        words = tokenize(pattern)
        all_words.extend(words)
        xy.append((words, tag))


# Remove punctuation
ignore_words = ['?', '.', '!']
all_words = [
    w for w in all_words 
    if w not in ignore_words
]


# Remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))


print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique words")


# Create training data
X_train = []
Y_train = []


for sentence, tag in xy:

    bag = bag_of_words(sentence,all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)



# Convert to tensors
X_train = torch.tensor(X_train,dtype=torch.float32)

Y_train = torch.tensor(Y_train,dtype=torch.long)



# Hyper parameters

num_epochs = 1000
batch_size = 8
learning_rate = 0.001


input_size = len(X_train[0])

# GRU needs bigger hidden state
hidden_size = 128

output_size = len(tags)


print(
    "Input:",
    input_size,
    "Output:",
    output_size
)



# Dataset class

class ChatDataset(Dataset):

    def __init__(self):

        self.n_samples = len(X_train)

        self.X_data = X_train
        self.Y_data = Y_train


    def __getitem__(self,index):

        return (
            self.X_data[index],
            self.Y_data[index]
        )


    def __len__(self):

        return self.n_samples



dataset = ChatDataset()


train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available()else "cpu")


print("Using:",device)

model = NeuralNet(input_size,hidden_size,output_size).to(device)



# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)



# Training

for epoch in range(num_epochs):

    for words, labels in train_loader:

        words = words.to(device)
        labels = labels.to(device)


        # forward
        output = model(words)


        loss = criterion(
            output,
            labels
        )


        # backward
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.4f}"
        )



print(
    f"Final Loss = {loss.item():.4f}"
)



# Save model

data = {

    "model_state": model.state_dict(),

    "input_size": input_size,

    "hidden_size": hidden_size,

    "output_size": output_size,

    "all_words": all_words,

    "tags": tags

}


FILE = "data.pth"


torch.save(
    data,
    FILE
)


print(
    f"Training complete. Saved {FILE}"
)