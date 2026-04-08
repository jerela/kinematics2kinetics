# Using the code

The code contains scripts that are meant to be run as entry points (file names starting with "main") and additional scripts that shouldn't be executed directly.

The entry points are as follows:
- main_train_full.py
  * For training and saving prediction models
- main_train_kfold.py
  * For prototyping different options and selecting hyperparameters
- main_test.py
  * For testing that trained models work as intended
- main_network_info.py
  * For showing information about trained models, such as their layer structure or number of parameters

The additional scripts are as follows:
- datasets.py
  * Contains one class, CustomTimeSeriesDataset, which defines how data (e.g., training data) is read, preprocessed and accessed in batches for network training/evaluation
- helpers_train_test.py
  * Contains various "helper" functions for training and testing models, including the actual training loop.
- networks.py
  * Contains the practical implementations of several artificial neural network architectures.
  * The ones used in the publication are KineticsCNN and DemographicScaler
- options.py
  * Contains options used by other scripts, such as input/output scaling boundaries, file paths, and training parameters.
- visualization.py
  * Contains helper functions and classes specifically for visualizing data
