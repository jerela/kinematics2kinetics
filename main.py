import statistics

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from datasets import CustomTimeSeriesDataset
from networks import KineticsLSTM, DemographicKineticsLSTM, FFN, CNN, KineticsGRU
from visualization import Plotter

from options import rng_seed, batch_size, early_stopping_threshold, max_epochs, file_dataset, lr_initial, lr_gamma, lr_step_size, plot_losses

torch.manual_seed(rng_seed)

def loss_fn(predicted, target):
    return ((predicted - target)**2).mean()

def train(model, training_set, validation_set=None, n_epochs = 100, lr = 0.01, early_stopping = 25, plot_losses=False, plot_sample=False):
    
    # construct DataLoader
    data_loader_training = DataLoader(training_set, shuffle=True, batch_size=batch_size)
    if validation_set:
        data_loader_validation = DataLoader(validation_set, shuffle=True, batch_size=batch_size)
    else:
        print('Validation set is not set, treating training loss also as validation loss.')
    
    # select optimizer and learning rate
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, min_lr=1e-9)
    
    # keep track of previous total loss so we know to break out of the epoch loop if total loss doesn't change between epochs
    previous_validation_loss = 0.0
    
    # lists to track training and validation losses per epoch
    losses_training = []
    losses_validation = []
    
    best_loss = float('inf')
    best_epoch = 0
    
    if plot_sample:
        sample_input_scalars, sample_input_time_series, sample_target = validation_set[0]
        sample_input_scalars = sample_input_scalars.unsqueeze(0)
        sample_input_time_series = sample_input_time_series.unsqueeze(0)
        sample_target = sample_target.unsqueeze(0).permute(0,2,1)
        output_untrained = model((sample_input_scalars, sample_input_time_series)).detach()
        loss_untrained = loss_fn(sample_target,output_untrained)
        plotter = Plotter()

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
            output_trained = model((sample_input_scalars, sample_input_time_series)).detach()
            loss_trained = loss_fn(sample_target,output_trained)
            plottable_data = (sample_target.detach().squeeze(0), output_untrained.squeeze(0), output_trained.squeeze(0))
            plottable_titles = ('target', 'untrained prediction', 'trained prediction')
            plotter.plot_samples(plottable_data, plottable_titles)

        # early stopping
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
        elif epoch > best_epoch+early_stopping:
            print(f'Breaking because validation loss has not reached a new minimum in {early_stopping} epochs. Steps taken: {epoch+1}')
            break
        
        previous_validation_loss = validation_loss
        
        # set the model back to training mode, update the weights, reset the gradient and check if learning rate needs to be reduced
        model.train(True)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(training_loss)
        
        
    return (training_loss.detach(), validation_loss.detach())


def main():
    
    dataset = CustomTimeSeriesDataset(file_dataset)
    
    
    
    n_inputs, n_targets = dataset.get_num_features()
    #model = KineticsLSTM(n_inputs,n_targets)
    #model = CNN(n_inputs,n_targets)
    #model = KineticsGRU(n_inputs,n_targets)
    
    k = 5
    idxs = dataset.kfold(k)
    
    
    layers = (1,2,3,4,5)
    
    loss_per_layer = []
    
    # loop through the number of layers for loss evaluation and hyperparameter selection
    for j in range(len(layers)):
        num_layers = layers[j]
        model = KineticsGRU(n_inputs,n_targets,num_layers)
    
        losses = []
        # loop through each fold
        for i in range(k):
            print(f'- - - FOLD {i+1} - - -')
            print(f'Dataset length: {len(dataset)}, numbers of indices: {len(idxs[i])}')
            # construct training and validation set for each fold by concatenating the indices in all but one fold (training) and counting the indices in the leftover fold (validation)
            idx_training = []
            idx_validation = []
            for j in range(k):
                if i == j:
                    idx_validation = idxs[i]
                else:
                    idx_training = idx_training + idxs[j]
            training_set = dataset.subset(idx_training)
            validation_set = dataset.subset(idx_validation)
            training_loss, validation_loss = train(model=model, training_set=training_set, validation_set=validation_set, n_epochs=max_epochs, early_stopping=early_stopping_threshold, lr=lr_initial, plot_losses=plot_losses, plot_sample=False)
            losses.append(validation_loss)
            
            
        mean_loss = statistics.fmean(losses)
        print(f'All {k} folds iterated. Losses: {losses}, mean loss: {mean_loss}')
        loss_per_layer.append(mean_loss)
        
    
    print(f'Numbers of layers: {layers}, corresponding losses: {loss_per_layer}')
    # find the number of layers with the smallest mean loss over all folds
    min_loss = float('inf')
    min_idx = -1
    for i,value in enumerate(loss_per_layer):
        if value < min_loss:
            min_loss = value
            min_idx = i
    print(f'Smallest loss of {min_loss} at {layers[min_idx]} layers.')

    
    

if __name__ == "__main__":
    main()




