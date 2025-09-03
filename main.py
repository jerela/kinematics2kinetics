import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from datasets import CustomTimeSeriesDataset
from networks import KineticsLSTM, DemographicKineticsLSTM, FFN, CNN, KineticsGRU
from visualization import Plotter

from options import rng_seed, batch_size, training_threshold, max_epochs, file_dataset, lr_initial, lr_gamma, lr_step_size

torch.manual_seed(rng_seed)

def loss_fn(predicted, target):
    return ((predicted - target)**2).mean()

def train(model, data_set, n_epochs = 100, lr = 0.01, threshold = 1e-5, plot=True):
    
    # construct DataLoader
    data_loader = DataLoader(data_set, shuffle=True, batch_size=batch_size)
    
    # select optimizer and learning rate
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, min_lr=1e-9)
    
    # keep track of previous total loss so we know to break out of the epoch loop if total loss doesn't change between epochs
    previous_total_loss = 0.0
    
    best_loss = float('inf')
    best_epoch = 0
    
    if plot:
        sample_input_scalars, sample_input_time_series, sample_target = data_set[0]
        sample_input_scalars = sample_input_scalars.unsqueeze(0)
        sample_input_time_series = sample_input_time_series.unsqueeze(0)
        sample_target = sample_target.unsqueeze(0).permute(0,2,1)
        output_untrained = model((sample_input_scalars, sample_input_time_series)).detach()
        loss_untrained = loss_fn(sample_target,output_untrained)
        plotter = Plotter()

        
    # iterate through epochs, n_epochs is the maximum number of epochs allowed
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        for i_input_scalars, i_input_time_series, i_target in data_loader:
            i_output = model((i_input_scalars, i_input_time_series)).permute(0,2,1)
            
            loss = loss_fn(i_output,i_target)
            
            loss.backward()
            
            total_loss += loss
            
        
        print(f'Epoch: {str(epoch+1)}, final loss: {total_loss}, learning rate: {scheduler.get_last_lr()}')
        
        if plot:
            
            output_trained = model((sample_input_scalars, sample_input_time_series)).detach()
            loss_trained = loss_fn(sample_target,output_trained)
            plottable_data = (sample_target.detach().squeeze(0), output_untrained.squeeze(0), output_trained.squeeze(0))
            plottable_titles = ('target', 'untrained prediction', 'trained prediction')
            plotter.plot(plottable_data, plottable_titles)
        
        
        if abs(total_loss-previous_total_loss) < threshold:
            print(f'Breaking because threshold loss was subceeded. Steps taken: {str(epoch+1)}')
            break
            
        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch
        elif epoch > best_epoch+100:
            print(f'Breaking because loss has not reached a new minimum in 100 epochs. Steps taken: {str(epoch+1)}')
            break
        
        previous_total_loss = total_loss
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(total_loss)
        
        
    


def main():
    
    dataset = CustomTimeSeriesDataset(file_dataset)
    
    
    
    n_inputs, n_targets = dataset.get_num_features()
    #model = KineticsLSTM(n_inputs,n_targets)
    #model = CNN(n_inputs,n_targets)
    model = KineticsGRU(n_inputs,n_targets)
    
    
    train(model=model, data_set=dataset, n_epochs=max_epochs, threshold=training_threshold, lr=lr_initial)

    
    

if __name__ == "__main__":
    main()




