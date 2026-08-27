"""
Conduct basic sensitivity analysis on the effect of certain generalized coordinates (joint angles) on predicted KCF.
One coordinate at a time, KCF in the test set is predicted by using different conditions: keeping the coordinate as it was, setting it to zeros, or multiplying it by 0.25, 0.5, 2, 3, or 4.
The resulting KCF predictions are written as CSV.
"""

import statistics

import torch

from datasets import CustomTimeSeriesDataset
from networks import KineticsCNN, DemographicScaler

from visualization import save_sample_figure
from options import path_test_data, path_trained_model, scalar_bounds, kinetics_bounds, path_output_predicted_time_series, kinetics_variable, included_generalized_coordinates
from helpers_train_test import get_time_series, denormalize

import pandas as pd


# joint angles to perturb one at a time
joint_angle_names = [
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "knee_angle_primary"
]


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

def test(model, test_set):
    
    # define loss function for evaluation loss
    loss_fn = rmse
    
    # ensure the model is in evaluation mode rather than training mode
    model.eval()
    
    # loop through the joint angles that are to be perturbed one at a time
    for i_coordinate in range(len(joint_angle_names)):
        current_coordinate_name = joint_angle_names[i_coordinate]
        # find the index of the current joint angle in the dataset
        current_coordinate_idx = [i for i, val in enumerate(included_generalized_coordinates) if val == current_coordinate_name]
        print(f'Coordinate {current_coordinate_name} at index {current_coordinate_idx}')
        
        # loop through different perturbation conditions; their behaviors are defined below inside the loop in an if statement
        for condition in ['default', 'zeros', 'quartered', 'halved', 'doubled', 'tripled', 'quadrupled']:
            
            
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
                    
                    # clone, because accessing by reference would modify the data in the dataset, which would cause problems in subsequent iterations
                    current_input_time_series = i_input_time_series.detach().clone()
                    
                    # behavior of different perturbation conditions; 'default' doesn't apply any perturbation to the joint angles
                    if condition == 'zeros':
                        current_input_time_series[current_coordinate_idx,:] = 0
                    elif condition == 'quartered':
                        current_input_time_series[current_coordinate_idx,:] *= 0.25
                    elif condition == 'halved':
                        current_input_time_series[current_coordinate_idx,:] *= 0.5
                    elif condition == 'doubled':
                        current_input_time_series[current_coordinate_idx,:] *= 2
                    elif condition == 'tripled':
                        current_input_time_series[current_coordinate_idx,:] *= 3
                    elif condition == 'quadrupled':
                        current_input_time_series[current_coordinate_idx,:] *= 4
                    elif condition == 'default':
                        pass
                    
                    # add a dimension in the beginning to make the data readable by the networks, which assume a batched shape (batch size being the first dimension)
                    i_input_scalars = i_input_scalars.unsqueeze(0)
                    current_input_time_series = current_input_time_series.unsqueeze(0)
                    i_target = i_target.unsqueeze(0)
                    
                    # compute the predicted time series
                    i_output = model((i_input_scalars, current_input_time_series)).permute(0,2,1)
                    
                    # now i_output and i_target are both of shape (1, 1, 250)
                    
                    # get the index of the first trailing zero-padded element in the time series
                    info_length = find_information_length(i_target)

                    # compute the error for the currently iterated sample and append it to the list of errors; use RMSE that ignores the trailing zeros
                    loss = loss_fn(i_output[:,:,0:info_length],i_target[:,:,0:info_length])

                    loss_N = denormalize(loss, kinetics_bounds[f'kcf_{kinetics_variable}'])
                    
                    mass = denormalize(i_input_scalars[0,0], scalar_bounds['body_mass'])
                    bw = mass*g
                    
                    losses.append(loss)            
                    losses_N.append(loss_N)            
                    losses_bw_normalized.append(loss_N/bw)
                    
                    predicted_time_series[i,:,:] = denormalize(i_output, kinetics_bounds[f'kcf_{kinetics_variable}'])
            
            # compute the final test error as the mean of the losses all samples
            test_loss = statistics.fmean(losses)
            test_loss_N = statistics.fmean(losses_N)
            test_loss_bw_normalized = statistics.fmean(losses_bw_normalized)
            print(f' Condition: {condition}, Newton loss: {test_loss_N}')
            
            # write CSV of predicted KCFs with current perturbation condition (or no perturbation in case of condition 'default')
            df = pd.DataFrame(data=predicted_time_series.squeeze(1).numpy())
            df.to_csv(f'{path_output_predicted_time_series}/Sensitivity/predicted_time_series_{kinetics_variable}_{current_coordinate_name}_{condition}.csv')
            
            
    
    # return the final test loss
    return (test_loss, test_loss_N, test_loss_bw_normalized)






def run_test_cnn():
    
    dataset = CustomTimeSeriesDataset(path_test_data)
    #dataset = CustomTimeSeriesDataset(file_dataset) # for using the training dataset; however, not recommended since training dataset has datasets that don't have lumbar kinematics enabled
    
    n_inputs, n_targets = dataset.get_num_features()
    sequence_length = dataset.get_sequence_length()
    print(f'Sequence length: {sequence_length}')
    
    # put the kernel size of the saved model here
    krnsz = 5
    
    # construct the model and load its previously optimized weights
    ts_model = KineticsCNN(n_inputs,n_targets,kernel_size=krnsz)
    model = DemographicScaler(time_series_model=ts_model, num_input_vectors=n_inputs, num_output_vectors=n_targets, sequence_length=sequence_length, name=f'Demographic_CNN_loaded')
    checkpoint = torch.load(path_trained_model)
    model.load_state_dict(checkpoint['model_state_dict_at_minimum_loss'])

    print(f'Dataset length: {len(dataset)}')
    
    # evaluate test performance
    test_loss = test(model=model, test_set=dataset)
    #print(f'Normalized test loss: {test_loss[0]:.5f}, denormalized to Newtons: {test_loss[1]:.1f}, with respect to body weight: {test_loss[2]:.2f}')
    




def main():
    # evaluate test performance on given directory and model file
    run_test_cnn()
    

if __name__ == "__main__":
    main()


