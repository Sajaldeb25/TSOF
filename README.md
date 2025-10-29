# Tree-based Self-Organizing Fuzzy (T-SOF) Classifier

This repository contains both MATLAB and Python implementations of a Tree-based Self-Organizing Fuzzy (T-SOF) classifier. The classifier is designed to learn from both static (offline) and streaming (online) data.

# ⚙️ How It Works
*The T-SOF classifier operates in three main phases:*

Offline Learning: The classifier is initially trained on a batch of static data. This phase establishes a foundational knowledge base using the OfflineTraining mode.

Online Learning: The pre-trained classifier is then updated incrementally with new data points as they arrive in a stream. This is handled by the EvolvingTraining mode, which allows the model to adapt over time.

Validation: After training, the classifier's performance is evaluated on a separate test dataset using the Validation mode. The script calculates and displays the classification accuracy and the confusion matrix.

# 🚀 Getting Started

## MATLAB Implementation

### Prerequisites
* MATLAB
* Fuzzy Logic Toolbox (required for T-SOF functions)
* The TSOF.m function file
* The GM_001.mat data file

### Running the MATLAB Code
1. Ensure all required files (TSOF.m, GM_001.mat) are in your MATLAB path.
2. Open the Main.m script in MATLAB.
3. Adjust the following parameters at the top of the script as needed:
   * `no_of_instance`: Total number of data instances in GM_data.
   * `for_offline_training`: Number of instances to use for offline learning.
   * `for_online_training`: Number of instances to use for online learning.
4. Run the script. The output will display the final confusion matrix and the classification accuracy.

## Python Implementation

### Prerequisites
* Python 3.7 or higher
* Required Python packages:
  * NumPy
  * SciPy
  * Matplotlib
  * NetworkX
  * scikit-learn

### Installation

#### Option 1: Using requirements.txt (Recommended)
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/Mac
source venv/bin/activate
# On Windows
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Manual installation
```bash
pip install numpy scipy matplotlib networkx scikit-learn
```

### Running the Python Code
1. Ensure all required files (TSOF.py, main.py, GM_001.mat) are in the same directory.
2. Run the main script using Python:
```bash
python main.py
```
3. If the original GM_001.mat data file is not found, the script will generate synthetic data for demonstration purposes.
4. The script will execute the three phases of the classifier (offline training, online training, validation) and display the confusion matrix and classification accuracy.
5. A visualization of the confusion matrix will be saved as 'confusion_matrix.png'.

### Adjusting Parameters
You can modify the parameters in the main.py script:
* `gran_level`: Controls the granularity level of the classifier (default: 12)
* `distance_type`: Distance metric used ('Euclidean', 'Mahalanobis', or 'Cosine')

The Python implementation includes the same functionality as the MATLAB version and follows the same methodology.
