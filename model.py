# installing libraries
import torch
import torch.nn as nn

# feedforward neural network (MLP) [ for that portion revise pytorch ]
class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.l2=nn.Linear(hidden_size,hidden_size)
        self.l3=nn.Linear(hidden_size,output_size)
        self.relu=nn.ReLU() # ReLU activation introduces non-linearity.
        
    # Defines how data flows through the network    
    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out) # Pass input x through the first linear layer, then apply ReLU.
        out=self.l2(out)
        out=self.relu(out) # Pass input out through the second linear layer, then apply ReLU.
        out=self.l3(out)
        # Final layer outputs raw scores (logits) for each class.No softmax here — PyTorch's CrossEntropyLoss automatically applies it during training. 
        return out