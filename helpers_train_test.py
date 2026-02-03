import statistics
import os
import copy

import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import CustomTimeSeriesDataset
from networks import KineticsLSTM, KineticsFFN, KineticsCNN, KineticsCNN2D, KineticsGRU, KineticsCNNLSTM, KineticsMLSTMFCN, WeightedMSELoss, DemographicScaler
from visualization import Plotter, save_loss_figure, save_sample_figure
from options import batch_size, early_stopping_threshold, max_epochs, file_dataset, lr_initial, plot_losses, plot_sample, workers, path_output






def get_time_series(model, dataset, loss_fn, n_samples=1, start_index=0):
    output = []
    model.eval()
    with torch.no_grad():
        for i_sample in range(n_samples):
            sample_input_scalars, sample_input_time_series, sample_target = dataset[i_sample+start_index]
            sample_input_scalars = sample_input_scalars.unsqueeze(0)
            sample_input_time_series = sample_input_time_series.unsqueeze(0)
            sample_target = sample_target.unsqueeze(0).permute(0,2,1)
            output_trained = model((sample_input_scalars, sample_input_time_series)).detach()
            loss_trained = loss_fn(sample_target,output_trained)
            plottable_data = (sample_target.detach().squeeze(0), output_trained.squeeze(0))
            output.append({
                'time_series': plottable_data,
                'loss': loss_trained
            })
    return output
            
#def loss_fn(predicted, target):
#    return ((predicted - target)**2).mean()

def save_checkpoint(checkpoint, name):
    full_path = os.path.join(path_output, 'Checkpoints', f'checkpoint_{name}.pt')
    torch.save(checkpoint, full_path)


def train(model, training_set, validation_set=None, n_epochs = 100, lr = 0.01, early_stopping = 25, plot_losses=False, plot_sample=False):
    
    loss_fn = WeightedMSELoss()
    
    # construct DataLoader
    data_loader_training = DataLoader(training_set, shuffle=True, batch_size=batch_size, num_workers=workers)
    if validation_set:
        data_loader_validation = DataLoader(validation_set, shuffle=True, batch_size=batch_size, num_workers=workers)
        sample_set = validation_set
    else:
        sample_set = training_set
        print('Validation set is not set, treating training loss also as validation loss.')
    
    # select optimizer and learning rate
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, threshold=1e-4, min_lr=1e-9)
    
    # keep track of previous total loss so we know to break out of the epoch loop if total loss doesn't change between epochs
    previous_validation_loss = 0.0
    
    # lists to track training and validation losses per epoch
    losses_training = []
    losses_validation = []
    
    best_loss = float('inf')
    best_epoch = 0
    
    if plot_sample:
        sample_input_scalars, sample_input_time_series, sample_target = sample_set[0]
        sample_input_scalars = sample_input_scalars.unsqueeze(0)
        sample_input_time_series = sample_input_time_series.unsqueeze(0)
        sample_target = sample_target.unsqueeze(0).permute(0,2,1)
        output_untrained = model((sample_input_scalars, sample_input_time_series)).detach()
        loss_untrained = loss_fn(sample_target,output_untrained)
        plotter = Plotter(training_set.get_sequence_length())

    if plot_losses:
        loss_plotter = Plotter()
        
    # iterate through epochs, n_epochs is the maximum number of epochs allowed
    for epoch in range(n_epochs):
        
        # set the model to training mode
        model.train(True)
        
        training_loss = 0.0
        
        for i_input_scalars, i_input_time_series, i_target in data_loader_training:
            i_output = model((i_input_scalars, i_input_time_series)).permute(0,2,1)
            
            loss = loss_fn(i_output,i_target)
            
            loss.backward()
            
            training_loss += loss
            
        
        # next, we do validation and set the model to evaluation mode for that
        if validation_set:
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for i_input_scalars, i_input_time_series, i_target in data_loader_validation:
                    i_output = model((i_input_scalars, i_input_time_series)).permute(0,2,1)
                
                    loss = loss_fn(i_output,i_target)
                    validation_loss += loss
        else:
            validation_loss = training_loss
        
        print(f'Epoch: {str(epoch+1)}, training loss: {training_loss:.5f}, validation loss: {validation_loss:.5f}, learning rate: {scheduler.get_last_lr()}')
        losses_training.append(training_loss.detach())
        losses_validation.append(validation_loss.detach())
        
        if plot_losses:
            plottable_titles = ('training loss', 'validation loss')
            loss_plotter.plot_losses((losses_training, losses_validation), plottable_titles)
        
        if plot_sample:
            model.eval()
            with torch.no_grad():
                output_trained = model((sample_input_scalars, sample_input_time_series)).detach()
                loss_trained = loss_fn(sample_target,output_trained)
                plottable_data = (sample_target.detach().squeeze(0), output_untrained.squeeze(0), output_trained.squeeze(0))
                plottable_titles = ('target', 'untrained prediction', 'trained prediction')
                plotter.plot_samples(plottable_data, plottable_titles)

        # early stopping: best loss means minimum loss
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            # we must take a deep copy of the model state_dict or it'll change whenever model is updated
            model_state_dict_at_best_loss = copy.deepcopy(model.state_dict())
        elif epoch > best_epoch+early_stopping:
            print(f'Breaking because validation loss has not reached a new minimum in {early_stopping} epochs. Steps taken: {epoch+1}')
            break
        
        previous_validation_loss = validation_loss
        
        # set the model back to training mode, update the weights, reset the gradient and check if learning rate needs to be reduced
        model.train(True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step(training_loss)
        
        
        if epoch%100 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': losses_training,
                'validation_loss': losses_validation,
                'minimum_loss': best_loss,
                'epoch_at_minimum_loss': best_epoch,
                'model_state_dict_at_minimum_loss': model_state_dict_at_best_loss
            }
            save_checkpoint(checkpoint, f'{model.model_name}_epoch{epoch+1}')
        
    
    # log model and optimizer states and figures
    save_checkpoint(checkpoint, f'{model.model_name}_epoch{epoch+1}_finished')
    save_loss_figure((losses_training,losses_validation), f'{model.model_name}_epoch{epoch+1}')
    save_sample_figure(get_time_series(model=model, dataset=sample_set, loss_fn=loss_fn, n_samples=torch.min(torch.tensor([len(sample_set), 9]))), name=f'{model.model_name}_epoch{epoch+1}')
    
    training_output = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_loss': losses_training,
        'validation_loss': losses_validation,
        'minimum_loss': best_loss,
        'epoch_at_minimum_loss': best_epoch,
        'model_state_dict_at_minimum_loss': model_state_dict_at_best_loss
    }
    
    return training_output



def run_kfold_gru():
    hyperparams = ['num_layers', (1,2,4,8)]
    run_kfold_validation(model_kinetics=KineticsGRU, model_name='GRU', hyperparameters=hyperparams)

def run_kfold_cnn():
    hyperparams = ['kernel_size', (7,9)]
    run_kfold_validation(model_kinetics=KineticsCNN, model_name='CNN', hyperparameters=hyperparams)

def run_kfold_cnn2d():
    hyperparams = ['kernel_width', (7,9)]
    run_kfold_validation(model_kinetics=KineticsCNN2D, model_name='CNN', hyperparameters=hyperparams)

def run_kfold_lstm():
    #hyperparams = ['hidden_size', (10,20,32,64,128)]
    hyperparams = ['hidden_size', [20]]
    run_kfold_validation(model_kinetics=KineticsLSTM, model_name='LSTM', hyperparameters=hyperparams)

def run_kfold_cnnlstm():
    hyperparams = ['hidden_size', [20]]
    args_kinetics = {'kernel_size': 7}
    run_kfold_validation(model_kinetics=KineticsCNNLSTM, model_name='CNN-LSTM', hyperparameters=hyperparams, kinetics_arguments=args_kinetics)

def run_kfold_mlstmfcn():
    hyperparams = ['hidden_size', [20]]
    run_kfold_validation(model_kinetics=KineticsMLSTMFCN, model_name='MLSTM-CNN', hyperparameters=hyperparams)

# generic function for running k-fold cross-validation with different models
def run_kfold_validation(model_kinetics, hyperparameters, kinetics_arguments={}, num_folds=5, model_name='unnamed'):
    dataset = CustomTimeSeriesDataset(file_dataset)
    
    n_inputs, n_targets = dataset.get_num_features()
    sequence_length = dataset.get_sequence_length()
    
    args_kinetics = {'num_input_vectors': n_inputs, 'num_output_vectors': n_targets}
    
    k = num_folds
    idxs = dataset.kfold(k)
    
    loss_per_hyperparameter = []
    
    hyperparameter_name = hyperparameters[0]
    hyperparameter_values = hyperparameters[1]

    # loop through the number of hyperparameters for loss evaluation and hyperparameter selection
    for j in range(len(hyperparameter_values)):
        
        args_hyperparams = {hyperparameter_name: hyperparameter_values[j]}
        args_kinetics_full = args_hyperparams | args_kinetics | kinetics_arguments
        
        losses = []
        # loop through each fold
        for i in range(k):
            
            kinetics_model_name = f'{model_name}_{hyperparameter_name}_{hyperparameter_values[j]}_fold{i+1}'
            # first, we instantiate a model for predicting the time series from input time series
            ts_model = model_kinetics(**args_kinetics_full, name=kinetics_model_name)
            # then, we instantiate a model that incorporates the time series model and additionally applies scaling based on demographics
            model = DemographicScaler(num_input_vectors=n_inputs, num_output_vectors=n_targets, sequence_length=sequence_length, time_series_model=ts_model, name=f'Demographic_{kinetics_model_name}')
            print(f'- - - FOLD {i+1} - - -')
            print(f'Dataset length: {len(dataset)}, numbers of indices: {len(idxs[i])}')
            # construct training and validation set for each fold by concatenating the indices in all but one fold (training) and counting the indices in the leftover fold (validation)
            idx_training = []
            idx_validation = []
            for m in range(k):
                if i == m:
                    idx_validation = idxs[i]
                else:
                    idx_training = idx_training + idxs[m]
            training_set = dataset.subset(idx_training)
            validation_set = dataset.subset(idx_validation)
            training_output = train(model=model, training_set=training_set, validation_set=validation_set, n_epochs=max_epochs, early_stopping=early_stopping_threshold, lr=lr_initial, plot_losses=plot_losses, plot_sample=plot_sample)
            validation_loss = training_output['validation_loss'][-1]
            losses.append(validation_loss)
            
            
        mean_loss = statistics.fmean(losses)
        print(f'All {k} folds iterated. Losses: {losses}, mean loss: {mean_loss}')
        loss_per_hyperparameter.append(mean_loss)
        
    
    print(f'Variations of hyperparameter {hyperparameter_name}: {hyperparameter_values}, corresponding losses: {loss_per_hyperparameter}')
    # find the hidden size with the smallest mean loss over all folds
    min_loss = float('inf')
    min_idx = -1
    for i,value in enumerate(loss_per_hyperparameter):
        if value < min_loss:
            min_loss = value
            min_idx = i
    print(f'Smallest loss of {min_loss} at {hyperparameter_name} value {hyperparameter_values[min_idx]}.')