# Python code for training and evaluating knee contact force prediction models

## Using the code

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

## License

The Python scripts in this folder are shared under the license, described below.

```
MIT License

Copyright (c) 2026 University of Eastern Finland

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
