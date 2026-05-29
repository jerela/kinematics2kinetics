import os
import copy

import torch

from datasets import CustomTimeSeriesDataset
from networks import KineticsCNN, DemographicScaler
from options import file_dataset, rng_seed, batch_size, plot_losses, plot_sample, workers, path_output, lr_initial
from helpers_train_test import train, kinetics_variable, kinetics_bounds, denormalize

torch.manual_seed(rng_seed)



def run_training():
    
    dataset = CustomTimeSeriesDataset(file_dataset)
    
    # split the original training set to 80% and 20% fractions; the 80% will be used as training data, while validation loss will be evaluated on the held-out 20% to determine when to stop training
    idx_training, idx_validation = dataset.get_split_indices(fractions=(80,20))
    training_set = dataset.subset(idx_training)
    validation_set = dataset.subset(idx_validation)
    
    n_inputs, n_targets = dataset.get_num_features()
    sequence_length = dataset.get_sequence_length()
    print(f'Sequence length: {sequence_length}')
    
    # define model hyperparameters, ideally based on previous hyperparameter optimization run
    hyperparameter_kernel_size = 5
    # epochs should be exaggeratedbly high when we're using early stopping so that we don't accidentally reach the maximum epochs before validation loss plateaus
    hyperparameter_epochs = 10000
    # construct the model
    ts_model = KineticsCNN(n_inputs,n_targets,kernel_size=hyperparameter_kernel_size,name=f'CNN_kernelsize{hyperparameter_kernel_size}')
    model = DemographicScaler(time_series_model=ts_model, num_input_vectors=n_inputs, num_output_vectors=n_targets, sequence_length=sequence_length, name=f'Demographic_CNN_full_kernelsize_{hyperparameter_kernel_size}')
    # train the model
    training_output = train(model=model, training_set=training_set, validation_set=validation_set, n_epochs=hyperparameter_epochs, early_stopping=200, lr=lr_initial, plot_losses=plot_losses, plot_sample=plot_sample)
    training_loss = training_output['training_loss'][-1]
    validation_loss = training_output['validation_loss'][-1]
    
    print(f'Final training loss after training the full model: {training_loss}, validation loss: {validation_loss}')
    print(f'Best validation loss of {denormalize(training_output['minimum_loss'], kinetics_bounds[f'kcf_{kinetics_variable}'])} at epoch {training_output['epoch_at_minimum_loss']+1}')
#denormalize(training_loss/float(len(training_set)), kinetics_bounds[f'kcf_{kinetics_variable}'])

def main():
    run_training()
    
    

if __name__ == "__main__":
    main()




