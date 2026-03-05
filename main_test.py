import statistics

import torch

from datasets import CustomTimeSeriesDataset
from networks import KineticsCNN, DemographicScaler

from visualization import save_sample_figure
from options import path_test_data, path_trained_model, scalar_bounds, kinetics_bounds, path_output_predicted_time_series
from helpers_train_test import get_time_series

import pandas as pd

# gravitational acceleration for calculating body weight from mass
g = 9.81

# root mean square error loss function
def rmse(target,prediction):
    return torch.sqrt(torch.mean(torch.square(target-prediction)))

# find the index of the first trailing zero (result of zero-padding to 250 elements); this is the number of "information" (non-trailing zero) elements in the time series
def find_information_length(data):
    time_series = data[0,0,:]
    nonzeros = torch.abs(time_series) > 1e-15
    idx_info = torch.nonzero(nonzeros)
    rightmost = idx_info[-1]
    return rightmost

# a function to return the output to its physically meaningful scale by undoing normalization
def denormalize(x,bounds):
    b_min = bounds[0]
    b_max = bounds[1]
    return b_min + x*(b_max-b_min)

def test(model, test_set):
    
    # define loss function for evaluation loss
    loss_fn = rmse
    
    # ensure the model is in evaluation mode rather than training mode
    model.eval()
    # prepare a list that contains the test loss for each sample in the test data
    losses = []
    losses_N = []
    losses_bw_normalized = []
    # initialize a tensor of empty values that will be replaced by the predicted time series
    predicted_time_series = torch.empty(len(test_set),test_set.get_num_features()[1],test_set.get_sequence_length())
    # loop through all samples in the test set and calculate test loss
    with torch.no_grad():
        for i in range(len(test_set)):
            i_input_scalars, i_input_time_series, i_target = test_set[i]
            
            # add a dimension in the beginning to make the data readable by the networks, which assume a batched shape (batch size being the first dimension)
            i_input_scalars = i_input_scalars.unsqueeze(0)
            i_input_time_series = i_input_time_series.unsqueeze(0)
            i_target = i_target.unsqueeze(0)
            
            # compute the predicted time series
            i_output = model((i_input_scalars, i_input_time_series)).permute(0,2,1)
            
            # now i_output and i_target are both of shape (1, 1, 250)
            
            # get the index of the first trailing zero-padded element in the time series
            info_length = find_information_length(i_target)

            # compute the error for the currently iterated sample and append it to the list of errors; use RMSE that ignores the trailing zeros
            loss = loss_fn(i_output[:,:,0:info_length],i_target[:,:,0:info_length])

            loss_N = denormalize(loss, kinetics_bounds['kcf_medial'])
            
            mass = denormalize(i_input_scalars[0,0], scalar_bounds['body_mass'])
            bw = mass*g
            
            losses.append(loss)            
            losses_N.append(loss_N)            
            losses_bw_normalized.append(loss_N/bw)
            
            predicted_time_series[i,:,:] = denormalize(i_output, kinetics_bounds['kcf_medial'])
    
    # compute the final test error as the mean of the losses all samples
    test_loss = statistics.fmean(losses)
    test_loss_N = statistics.fmean(losses_N)
    test_loss_bw_normalized = statistics.fmean(losses_bw_normalized)
    print(f'BW normalized loss: {test_loss_bw_normalized}')
    print(f'Newton loss: {test_loss_N}')
    print(f'Predicted time series list shape: {predicted_time_series.shape}')
    print(predicted_time_series)
    
    for i in range(int(len(test_set)/9)):
        save_sample_figure(get_time_series(model=model, dataset=test_set, loss_fn=loss_fn, n_samples=torch.min(torch.tensor([len(test_set), 9])), start_index=i*9), name=f'{model.model_name}_test_{i}')
    
    df = pd.DataFrame(data=predicted_time_series.squeeze(1).numpy())
    print(df)
    
    df.to_csv(path_output_predicted_time_series)
    
    # return the final test loss
    return (test_loss, test_loss_N, test_loss_bw_normalized)






def run_test_cnn():
    
    dataset = CustomTimeSeriesDataset(path_test_data)
    
    n_inputs, n_targets = dataset.get_num_features()
    sequence_length = dataset.get_sequence_length()
    print(f'Sequence length: {sequence_length}')
    
    # put the kernel size of the saved model here
    krnsz = 7
    
    # construct the model and load its previously optimized weights
    ts_model = KineticsCNN(n_inputs,n_targets,kernel_size=krnsz)
    model = DemographicScaler(time_series_model=ts_model, num_input_vectors=n_inputs, num_output_vectors=n_targets, sequence_length=sequence_length, name=f'Demographic_CNN_loaded')
    checkpoint = torch.load(path_trained_model)
    #print(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict_at_minimum_loss'])

    print(f'Dataset length: {len(dataset)}')
    
    # evaluate test performance
    test_loss = test(model=model, test_set=dataset)
    print(f'Normalized test loss: {test_loss[0]:.5f}, denormalized to Newtons: {test_loss[1]:.1f}, with respect to body weight: {test_loss[2]:.2f}')
    




def main():
    # evaluate test performance on given directory and model file
    run_test_cnn()
    

if __name__ == "__main__":
    main()


