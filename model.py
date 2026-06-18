import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()

        self.hidden_size = hidden_size

        # convert bag of words features into hidden representation
        self.embedding = nn.Linear(input_size, hidden_size)

        # GRU layer
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )

        # classifier
        self.fc = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):

        # x shape: (batch, features)
        x = self.embedding(x)

        x = self.relu(x)

        # GRU expects:
        # (batch, sequence, features)
        x = x.unsqueeze(1)

        out, hidden = self.gru(x)

        # take last hidden state
        out = hidden[-1]

        out = self.dropout(out)

        out = self.fc(out)

        # logits (no softmax)
        return out