# %% creating the model
import torch
from torch.nn import Tanh, MaxPool2d, LogSoftmax, Flatten
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
        self.convolution_stack = nn.Sequential( #input = 1x28x28
            nn.Conv2d(1, 16, kernel_size=5), #output = 16x24x24
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2), #output = 16x12x12
            
            nn.Conv2d(16, 32, kernel_size=5), #output = 32x8x8
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2), #output = 32x4x4

            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.Tanh(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1) 
        )

    def forward(self, x):
        x = self.convolution_stack(x)
        return x

#it creates an instance of NeuralNetwork, and moves it to device
model = NeuralNetwork()
print(model) 

#this approach uses nn.sequential within the definition of a class. The data is passed through all the modules in 
#the same order as it is defined. 
#to use the model, we pass it the input data, which executes the model's "forward".
#do not call directly the model

#by using the model's parameters() or named_parameters() methods, 
# %%
