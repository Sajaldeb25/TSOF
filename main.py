import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from TSOF import sof_classifier

# Load MATLAB data file
# You'll need to replace 'GM_001.mat' with your actual data file
try:
    mat_data = scipy.io.loadmat('GM_001.mat')
    
    # Extract variables from the MAT file
    GM_data = mat_data['GM_data']
    GM_label = mat_data['GM_label'].flatten()  # Ensure labels are flattened
    GM_test_data = mat_data['GM_test_data']
    GM_test_label = mat_data['GM_test_label'].flatten()  # Ensure labels are flattened
    no_of_instance = mat_data['no_of_instance'][0][0]
    for_offine_training = mat_data['for_offine_training'][0][0]
    for_online_training = mat_data['for_online_training'][0][0]
    
except FileNotFoundError:
    print("Data file not found! This is a sample implementation.")
    print("Creating synthetic data for demonstration purposes...")
    
    # Create synthetic data for demonstration if the MAT file is not found
    np.random.seed(42)
    
    # Set parameters
    no_of_instance = 300
    for_offine_training = 100
    for_online_training = 200
    n_features = 10
    n_classes = 3
    
    # Generate synthetic data
    GM_data = np.random.randn(no_of_instance, n_features)
    GM_label = np.random.randint(0, n_classes, size=no_of_instance)  # Flattened labels
    GM_test_data = np.random.randn(100, n_features)
    GM_test_label = np.random.randint(0, n_classes, size=100)  # Flattened labels

# Shuffle the data
np.random.seed(42)  # For reproducibility
k = np.random.permutation(no_of_instance)

# Split data for offline training
data_train = GM_data[k[:for_offine_training]]
label_train = GM_label[k[:for_offine_training]]

print("Starting offline training...")

# Phase 1: Offline Training
input_data = {
    'TrainingData': data_train,
    'TrainingLabel': label_train
}
gran_level = 12
distance_type = 'Cosine'
mode = 'OfflineTraining'

output0 = sof_classifier(input_data, gran_level, mode, distance_type)

print("Offline training completed.")
print("Starting online training...")

# Phase 2: Online/Evolving Training
data_train3 = GM_data[k[for_offine_training:for_online_training]]
label_train3 = GM_label[k[for_offine_training:for_online_training]]

input_data = output0
input_data['TrainingData'] = data_train3
input_data['TrainingLabel'] = label_train3
mode = 'EvolvingTraining'

output1 = sof_classifier(input_data, gran_level, mode, distance_type)

print("Online training completed.")
print("Starting validation...")

# Phase 3: Validation
input_data = output1
input_data['TestingData'] = GM_test_data
input_data['TestingLabel'] = GM_test_label
mode = 'Validation'

output2 = sof_classifier(input_data, gran_level, mode, distance_type)

# Display results
confusion_matrix = output2['ConfusionMatrix']
print("\nConfusion Matrix:")
print(confusion_matrix)

# Calculate classification accuracy
accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
print(f"\nClassification Accuracy: {accuracy:.4f}")

# Visualize the confusion matrix if matplotlib is available
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Add labels and tick marks
n_classes = confusion_matrix.shape[0]
plt.xticks(np.arange(n_classes))
plt.yticks(np.arange(n_classes))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = confusion_matrix.max() / 2
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print("\nValidation completed. Results saved.")
