#This code runs a fully connected deep neural network (DNN) training over the MNIST dataset
# %% import libraries
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm  # Import tqdm for progress bar
from torch.optim.lr_scheduler import LambdaLR


# %% download training data 
#every dataset contains two arguments, transform and target transform to modify samples and labels

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

#pass dataset as an argument to dataloader. This wraps an iterable over our dataset.
batch_size = 8

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#this is helpful for debugging and checking the structure of the data before training
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}") #the shape will be batch size, channels, height, width
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#for the X
#batch size is given by the dataloader function 
#channels are 1 since the image is in greyscale
#the images are 28x28 so height, width

#for the y
#the y are the labels. batch size is 64, while the label is a number integer

#the break stops the loop after the first batch of the test dataset, and is useful to inspect the shapes before running the full training process
# %% interacting and visualizing the dataset

#the data can be indexed manually with training_data[index]

import matplotlib.pyplot as plt

labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() #generate random number
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# %% create the model

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

# %% define train and test codes

#then we define a training loop to make predictions on the input data, and then adjust

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # Set the model to train mode (important for layers like dropout and batch norm)
    for batch, (X, y) in enumerate(dataloader):
       # X, y = X.to(device), y.to(device)

        #In this case, flatten operation is performed
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

# %% train the model

loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
accuracy_values = [] #array to store accuracy values

# Define the learning rate schedule
def lr_schedule(epochs):
    if epochs < 10:
        return 1.0  # Keep the base learning rate (0.02)
    elif epochs < 15:
        return 0.5  # Reduce learning rate by 10%
    else:
        return 0.1  # Reduce learning rate by 10%

# Scheduler
scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

epochs = 30
for t in tqdm(range(epochs),  desc="Training Progress"):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn, accuracy_values)
print("Done!")

# %%
import matplotlib.pyplot as plt


plt.plot(range(epochs), accuracy_values, label="Accuracy", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.show()

saveAccuracyFlag = 0
if saveAccuracyFlag == True:
    import csv

    savePathAccuracy = 'modelsAccuracy'
    savePathAccuracy = savePathAccuracy + '/DNN_MNIST_FP.csv'
    # Writing to CSV
    with open(savePathAccuracy, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Accuracy'])  # Write header (optional)
        for epoch, accuracy in enumerate(accuracy_values, 1):
            writer.writerow([epoch, accuracy])  # Write each accuracy value with the epoch number

    print(f"Accuracy values saved to {savePathAccuracy}")

# %%

savePath = 'modelsAccuracy'
saveConfusionMatrix = savePath + '/confusionMatrixDNN.svg'

import importlib
import utils.confusionMatrixPlot as confusionMatrixPlot
importlib.reload(confusionMatrixPlot)
from utils.confusionMatrixPlot import confusionMatrixPlot1

confusionMatrixPlot1(model, test_dataloader, saveConfusionMatrix, 0)

# %%
