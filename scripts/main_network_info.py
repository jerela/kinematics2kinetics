import statistics

import torch

from torchinfo import summary

from datasets import CustomTimeSeriesDataset
from networks import KineticsCNN, DemographicScaler

from visualization import save_sample_figure
from options import path_test_data, path_trained_model



def main():
        
    dataset = CustomTimeSeriesDataset(path_test_data)
    
    n_inputs, n_targets = dataset.get_num_features()
    sequence_length = dataset.get_sequence_length()
    print(f'Sequence length: {sequence_length}')
    
    # put the kernel size of the saved model here
    krnsz = 9
    
    # construct the model and load its previously optimized weights
    ts_model = KineticsCNN(n_inputs,n_targets,kernel_size=krnsz)
    model = DemographicScaler(time_series_model=ts_model, num_input_vectors=n_inputs, num_output_vectors=n_targets, sequence_length=sequence_length, name=f'Demographic_CNN_loaded')
    checkpoint = torch.load(path_trained_model)
    
    epoch_at_minimum_loss = checkpoint['epoch_at_minimum_loss']
    training_loss = checkpoint['training_loss']
    
    info_str = (
        f'Loaded checkpoint, summary:\n'
        f'- epoch where the iteration stopped: {checkpoint['epoch']}\n'
        f'- final training loss: {training_loss[-1]}\n'
        f'- final validation loss: {checkpoint['validation_loss'][-1]}\n'
        f'- minimum validation loss: {checkpoint['minimum_loss']}\n'
        f'- epoch where the minimum validation loss was achieved: {epoch_at_minimum_loss}\n'
        f'- training loss at the epoch of minimum validation loss: {training_loss[epoch_at_minimum_loss]}\n'
    )
    print(info_str)
    
    
    #print(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict_at_minimum_loss'])
        
    i_input_scalars, i_input_time_series, i_target = dataset[0]
    
    # add a dimension in the beginning to make the data readable by the networks, which assume a batched shape (batch size being the first dimension)
    i_input_scalars = i_input_scalars.unsqueeze(0)
    i_input_time_series = i_input_time_series.unsqueeze(0)
    i_target = i_target.unsqueeze(0)
    
    # compute the predicted time series
    i_output = model((i_input_scalars, i_input_time_series)).permute(0,2,1)

    
    print(f'checkpoint keys: {checkpoint.keys()}')
    print(f'model: {model}')
    print(f'model summary: {summary(model, input_data=[(i_input_scalars, i_input_time_series)], col_names=('input_size', 'output_size', 'num_params'), mode='eval')}')
    
    # get the weights of the first linear layer in the demographics using module; this layer will take 4 preprocessed demographics scalars and turns them into 16 features
    demographic_fc1_weights = [x for x in model.fc1.parameters()][0]
    # print the weights, showing a 16-by-4 tensor
    print(f'demographic fc1 weights: {demographic_fc1_weights}')
    # take the mean of the absolute values of the weights to get an idea of how much each input scalar is weighted; note that this interpretation assumes that the scalars are similarly normalized during preprocessing
    print(torch.mean(torch.abs(demographic_fc1_weights), dim=0))
    

if __name__ == "__main__":
    main()


