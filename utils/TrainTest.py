# %% Optimizing the model parameters
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
import numpy as np
#to train a model, a loss function and an optimizer are needed. 

#loss_fn = nn.NLLLoss() #the other example uses nn.NLLLoss
#optimizer = torch.optim.SGD(model.parameters(), lr=0.05) #the optimizer is different for HardwareKit

#then we define a training loop to make predictions on the input data, and then adjust

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # Set the model to train mode (important for layers like dropout and batch norm)
    for batch, (X, y) in enumerate(dataloader):
       # X, y = X.to(device), y.to(device)

        #In this case, flatten operation is perfor
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y) # Compute the loss by comparing the predictions with the true labels

        # Backpropagation
        loss.backward() # Backpropagation: Compute the gradients of the loss with respect to the model's parameters
        optimizer.step()  # Update the model's weights based on the computed gradients
        optimizer.zero_grad()  # Zero the gradients for the next step (otherwise gradients will accumulate)

        # Print the loss every 100 batches
        if batch % 100 == 0: 
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, accuracy_values):
    size = len(dataloader.dataset)
    num_batches = len(dataloader) #number of batches in the dataloader
    model.eval() # Set the model to evaluation mode (important for layers like dropout and batch norm)
    test_loss, correct = 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy_values.append(100*correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
