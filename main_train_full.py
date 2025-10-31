import os
import copy

import torch

from datasets import CustomTimeSeriesDataset
from networks import KineticsLSTM, KineticsFFN, KineticsCNN, KineticsCNN2D, KineticsGRU, WeightedMSELoss, DemographicScaler
from options import file_dataset, rng_seed, batch_size, plot_losses, plot_sample, workers, path_output, lr_initial
from helpers_train_test import train

torch.manual_seed(rng_seed)



def run_training():
    
    dataset = CustomTimeSeriesDataset(file_dataset)
    
    n_inputs, n_targets = dataset.get_num_features()
    sequence_length = dataset.get_sequence_length()
    print(f'Sequence length: {sequence_length}')
    
    # define model hyperparameters, ideally based on previous hyperparameter optimization runs
    hyperparameter_kernel_size = 9
    hyperparameter_epochs = 700
    # construct the model
    ts_model = KineticsCNN(n_inputs,n_targets,kernel_size=hyperparameter_kernel_size,name=f'CNN_kernelsize{hyperparameter_kernel_size}')
    model = DemographicScaler(time_series_model=ts_model, num_input_vectors=n_inputs, num_output_vectors=n_targets, sequence_length=sequence_length, name=f'Demographic_CNN_full_kernelsize_{hyperparameter_kernel_size}')
    # train the model
    training_output = train(model=model, training_set=dataset, validation_set=None, n_epochs=hyperparameter_epochs, early_stopping=100, lr=lr_initial, plot_losses=plot_losses, plot_sample=plot_sample)
    training_loss = training_output['training_loss'][-1]
    
    print(f'Final loss after training the full model: {training_loss}')


def main():
    run_training()
    
    

if __name__ == "__main__":
    main()




