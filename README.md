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

This repository contains two things:

1. Python **code** for training artificial neural networks for predicting knee contact force time series from joint angle time series and demographic variables (in the [scripts](scripts/) directory)
2. Existing trained **models** for predicting knee contact forces (in the [models](models/) directory)

The data for training the artificial neural networks will be put in a separate data repository and will be linked later.

The repository accompanies and is better understood by reading our [publication](#publication-and-citation).

## Getting started

Below are instructions for using the **code** or **models** in this repository. Make sure to read also check their [licenses](#license-and-copyright) and [citation instructions](#publication-and-citation) further below.

### Using the code
See under [scripts](scripts/README.md).

### Using the models

See under [models](models/README.md).

## Authors

Jere Lavikainen, jere.lavikainen (at) uef.fi

## License and copyright

Different licenses apply to different parts of the repository. 
- The **code** is shared under the MIT license. For the license, see [here](scripts/#license).
- The **models** are shared under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. This is because the data is reprocessed data from five original sources, each under the CC BY 4.0 license, and the **models** are trained on that data. For the license, see the [here](models/#license).

## Publication and citation

Publication pending.

## Acknowledgements

We thank Peiffer et al. for their work on [MonocularBiomechanics](https://intelligentsensingandrehabilitation.github.io/MonocularBiomechanics/), which was utilized in our publication. The fork of MonocularBiomechanics used in our publication can be found [here](https://github.com/jerela/MonocularBiomechanics/tree/batch-processing).
