# %% import the dataset
from utils.MNIST_Loader import train_dataloader, test_dataloader
#import the model made in pyTorch
from utils.DNN_architecture import model
#import the functions used to train and test the model
from utils.TrainTest import train, test

# %% TRAIN THE MODEL
from tqdm import tqdm  # Import tqdm for progress bar
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

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

# %% plot the accuracy
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
    #save the accuracy vector in a CSV file
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
saveConfusionMatrix = savePath + '/confusionMatrix8.svg'

import importlib
import utils.confusionMatrixPlot as confusionMatrixPlot
importlib.reload(confusionMatrixPlot)
from utils.confusionMatrixPlot import confusionMatrixPlot1

confusionMatrixPlot1(model, test_dataloader, saveConfusionMatrix, 0)
# %%
