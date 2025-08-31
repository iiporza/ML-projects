# %% creating the model
import torch
import torch.nn as nn

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
#the neural network is defined by subclassing nn.Module,
#  it is initialized in __init__. Every subclass implements the operations in the forward method

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x) #converts input into contiguous array. It maintains the minibatch dimension dim = 0
        logits = self.linear_stack(x)
        return logits

#it creates an instance of NeuralNetwork, and moves it to device
model = NeuralNetwork()
print(model) 

#this approach uses nn.sequential within the definition of a class. The data is passed through all the modules in 
#the same order as it is defined. 
#to use the model, we pass it the input data, which executes the model's "forward".
#do not call directly the model

#by using the model's parameters() or named_parameters() methods, 
# %%
