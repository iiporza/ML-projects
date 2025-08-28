# %% import libraries
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
# %%
