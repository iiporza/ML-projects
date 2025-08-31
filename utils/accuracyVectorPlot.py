# %%
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%
# Define a function to load accuracy values from a CSV file
def load_accuracy_from_csv(file_path):
    accuracy_values = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        accuracy_values = [float(row[1]) for row in reader]  # Read accuracy values
    return accuracy_values

savePathAccuracy = 'modelsAccuracy'
# List of files to load the accuracy values from
file_paths = [  # Example CSV file
        (savePathAccuracy + '/accuracyVector_Ideal.csv', 'Ideal Device'),
        (savePathAccuracy + '/accuracyVector_FP.csv', 'FP, batch size 64'),
        (savePathAccuracy + '/accuracyVector_FP2.csv', 'FP, batch size 8'),
        (savePathAccuracy + '/accuracyVector_FP4.csv', 'FP, batch size 2'),   
 ]


# Initialize a list to store the accuracy values for each run
all_accuracy_values = []
labels = []

for file_path, label in file_paths:
        if file_path.endswith('.csv'):
            all_accuracy_values.append(load_accuracy_from_csv(file_path))
            labels.append(label)

# %% plot everything
# Plotting all accuracy values on the same plot
plt.figure(figsize=(10, 6))  # Set figure size

# Loop over all loaded accuracy values and plot them
for accuracy_values, label in zip(all_accuracy_values, labels):
    plt.plot(accuracy_values, label=label)  # Plot each run with a custom label

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Epochs for Different Runs')
plt.legend()  # Show the legend to distinguish the lines
plt.grid(True)

# Show the plot
plt.show()
# %% plot the test error

plt.figure(figsize=(10, 6))  # Set figure size

for accuracy_values, label in zip(all_accuracy_values, labels):
    # Calculate test error and plot it
    test_error_values = [100 - acc for acc in accuracy_values]  # Test error = 100 - Accuracy
    plt.plot(test_error_values, label=f'{label} - Test Error', linestyle='--')  # Plot test error with dashed line

# Adding labels and title
plt.xlabel('Epochs')
plt.yscale("log") 
plt.ylabel('Test Error (%)')
plt.title('Test Error Over Epochs for Different Runs')
plt.legend()  # Show the legend to distinguish the lines
plt.grid(True)

# Show the plot
plt.show()

# %%
