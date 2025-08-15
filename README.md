# Tree-based Self-Organizing Fuzzy (T-SOF) Classifier
This repository contains MATLAB code for a Tree-based Self-Organizing Fuzzy (T-SOF) classifier. The classifier is designed to learn from both static (offline) and streaming (online) data.

# ⚙️ How It Works
*The T-SOF classifier operates in three main phases:*

Offline Learning: The classifier is initially trained on a batch of static data. This phase establishes a foundational knowledge base using the OfflineTraining mode.

Online Learning: The pre-trained classifier is then updated incrementally with new data points as they arrive in a stream. This is handled by the EvolvingTraining mode, which allows the model to adapt over time.

Validation: After training, the classifier's performance is evaluated on a separate test dataset using the Validation mode. The script calculates and displays the classification accuracy and the confusion matrix.

# 🚀 Getting Started
* Prerequisites
* MATLAB
* Fuzzy Logic Toolbox (required for T-SOF functions)
* The TSOF.m function file
* The GM_001.mat data file

Running the Code
Ensure all required files (TSOF.m, .mat) are in your MATLAB path.

* Open the main script in MATLAB.

# Adjust the following parameters at the top of the script as needed:

no_of_instance: Total number of data instances in GM_data.

for_offline_training: Number of instances to use for offline learning.

for_online_training: Number of instances to use for online learning.

Run the script. The output will display the final confusion matrix and the classification accuracy.
