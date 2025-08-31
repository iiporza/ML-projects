# %% import the dataset
from utils.MNIST_Loader import train_dataloader, test_dataloader
#import the model made in pyTorch
from utils.LeNet5_architecture import model
#import the functions used to train and test the model
from utils.TrainTest import train, test

# %% TRAIN THE MODEL
from tqdm import tqdm  # Import tqdm for progress bar
import torch
import torch.nn as nn

#loss_fn = nn.NLLLoss()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
accuracy_values = [] #array to store accuracy values

epochs = 30
for t in tqdm(range(epochs),  desc="Training Progress"):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn, accuracy_values)
print("Done!")

# %% plot the accuracy
import matplotlib.pyplot as plt

#Plot the accuracy as a function of the epoch
savePath = 'modelsAccuracy'
saveAccuracy = savePath + '/accuracy.svg'

plt.plot(range(epochs), accuracy_values, label="Accuracy", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)

#plt.savefig(saveAccuracy, format='svg')

plt.show()

saveAccuracyFlag = 1

if saveAccuracyFlag == True:
    #save the accuracy vector in a CSV file
    import csv

    savePathAccuracy = 'modelsAccuracy'
    savePathAccuracy = savePathAccuracy + '/MNIST_FP_CNN_ref.csv'
    # Writing to CSV
    with open(savePathAccuracy, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Accuracy'])  # Write header (optional)
        for epoch, accuracy in enumerate(accuracy_values, 1):
            writer.writerow([epoch, accuracy])  # Write each accuracy value with the epoch number

    print(f"Accuracy values saved to {savePathAccuracy}")
# %%

savePath = 'modelsAccuracy'
saveConfusionMatrix = savePath + '/confusionMatrixCNNFP.svg'

import importlib
import utils.confusionMatrixPlot as confusionMatrixPlot
importlib.reload(confusionMatrixPlot)
from utils.confusionMatrixPlot import confusionMatrixPlot1

confusionMatrixPlot1(model, test_dataloader, saveConfusionMatrix, 1)
# %%
savePath = 'modelsAccuracy'
saveConfusionMatrix = savePath + '/confusionMatrixCNNFPcompact.svg'

import importlib
import utils.confusionMatrixPlotCompact as confusionMatrixPlotCompact
importlib.reload(confusionMatrixPlotCompact)
from utils.confusionMatrixPlotCompact import confusionMatrixPlot2

confusionMatrixPlot2(model, test_dataloader, saveConfusionMatrix, 1)
# %%
