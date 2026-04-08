# kinematics2kinetics

- [Welcome](#welcome)
  * [Prerequisites](#prerequisites)
  * [Installing](#installing)
  * [Running the program](#running-the-program)
- [Getting started](#getting-started)
  * [Using the code](#using-the-code)
  * [Using the data](#using-the-data)
  * [Using the models](#using-the-models)
- [Authors](#authors)
- [License and copyright](#license-and-copyright)
- [Publication and citation](#publication-and-citation)
- [Acknowledgements](#acknowledgements)
<!-- toc -->

## Welcome

This repository contains three things:

1. Python **code** for training artificial neural networks for predicting knee contact force time series from joint angle time series and demographic variables (folder [scripts](scripts/))
2. **Data** for training the artificial neural networks (folder [data](data/))
3. Existing trained **models** for predicting knee contact forces (folder [models](models/))

The repository accompanies and is better understood by reading our [publication](#publication-and-citation).

## Getting started

Below are instructions for using the **code**, **data**, or **models** in this repository. Make sure to read also check their [licenses](#license-and-copyright) and [citation instructions](#publication-and-citation) further below.

### Using the code

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

### Using the data


### Using the models

The models are exported as Numpy ...
The exported model files do not include the preprocessing or postprocessing steps, so make sure to check them out from the **code** if you wish to replicate them.



## Authors

Jere Lavikainen, jere.lavikainen (at) uef.fi

## License and copyright

Different licenses apply to different parts of the repository. 
- The **code** is shared under the MIT license. For the license, see the **code** folder.
- The **data** and **models** are shared under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. This is because the **data** is reprocessed data from four original sources, each under the CC BY 4.0 license, and the **models** are trained on that **data**. For the license, see the **models** or **data** folders.

## Publication and citation

Publication pending.

If you use or modify the **code**, the **data**, or any of the **models** in your work, please cite [WIP]. In the case of the **data** or **models**, please also cite the source datasets individually:
- Camargo
- Horst
- Schreiber
- Fukuchi

## Acknowledgements

We thank Peiffer et al. for their work on [MonocularBiomechanics](https://intelligentsensingandrehabilitation.github.io/MonocularBiomechanics/), which was utilized in our publication. The fork of MonocularBiomechanics used in our publication can be found [here](https://github.com/jerela/MonocularBiomechanics/tree/batch-processing).


