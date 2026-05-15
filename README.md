# KALS - Dynamical System Neural Network code

## Author

Derek Karssenberg, Utrecht University


## Introduction

This repository provides all the code required to create data,
tables, and figures for the DSNN model for catchment hydrology. For
explanation the reader is referred to the associated publication.
This is not intended as a 'ready-to-use' model. To use the code
some explanation is needed and possible users are requested to reach
out to the author.

The main Python scripts are:
- kals_model.py, the DSNN model itself
- postprocessing.py, script to postprocess the DSSN model outputs

All other files are for running in batch or pdf post processing.

## Structure of kals_model.py

The script contains the following components (from top to bottom).

1. Main configurations. Main configurations and explanation
of how the model can be run in batch.
1. Data preparation. Functions to convert the input data (timeseries
of meteo data) into a format that can be used in the model.
2. Dynamical System Neural Network model. This is the model itself.
The first part of the code defines the neural network components.
The second part defines the three model components, `eva_f_calculate`,
`sno_f_calculate`, `sub_f_calculate`, each time with three options
of code for synthetic data generation, process-based expert model,
or neural network. The third part defines the iteration over time,
calling the model components.
3. Create artificial observations. Code to create the synthetic data
set. It calls the model with the configuration such that synthetic
data model components are used.
4. Training. Code for training the model (function: `training`).
It loops over epochs. Each epoch the model is run twice. The first
run is for training, on the training fold. This same run is used
for validation (testing), on the remaining data. The second run is
for testing (independent validation), on the independent validation
(testing) data. The second part of this code creates intermediate
figures and write data to disk.
5. Data set creation. Calls the function from Data preparation to
create the input data for the model.
6. Fitting scenarios. Defines fitting scenarios.



