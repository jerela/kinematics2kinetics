import torch
from torch.utils.data import DataLoader

from datasets import CustomTimeSeriesDataset
from networks import KineticsLSTM, KineticsFFN, KineticsCNN, KineticsCNN2D, KineticsGRU, WeightedMSELoss, DemographicScaler

from visualization import save_sample_figure
from options import path_test_data, path_trained_model, batch_size, workers
from helpers_train_test import get_time_series


def test(model, test_set):
    
    # define loss function for evaluation loss
    loss_fn = WeightedMSELoss()
    
    # construct DataLoader to evaluate the samples in batches
    data_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=workers)
    
    # ensure the model is in evaluation mode rather than training mode
    model.eval()
    # initialize test loss as 0
    test_loss = 0.0
    # update test loss in batches
    with torch.no_grad():
        for i_input_scalars, i_input_time_series, i_target in data_loader:
            i_output = model((i_input_scalars, i_input_time_series)).permute(0,2,1)
        
            loss = loss_fn(i_output,i_target)
            test_loss += loss
    
    for i in range(int(len(test_set)/9)):
        save_sample_figure(get_time_series(model=model, dataset=test_set, loss_fn=loss_fn, n_samples=torch.min(torch.tensor([len(test_set), 9])), start_index=i*9), name=f'{model.model_name}_test_{i}')
    
    
    # return the final test loss
    return test_loss






def run_test_cnn():
    
    dataset = CustomTimeSeriesDataset(path_test_data)
    
    n_inputs, n_targets = dataset.get_num_features()
    sequence_length = dataset.get_sequence_length()
    print(f'Sequence length: {sequence_length}')
    
    krnsz = 9
    
    # construct the model and load its previously optimized weights
    ts_model = KineticsCNN(n_inputs,n_targets,kernel_size=krnsz)
    model = DemographicScaler(time_series_model=ts_model, num_input_vectors=n_inputs, num_output_vectors=n_targets, sequence_length=sequence_length, name=f'Demographic_CNN_loaded')
    checkpoint = torch.load(path_trained_model)
    #print(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict_at_minimum_loss'])

    print(f'Dataset length: {len(dataset)}')
    
    # evaluate test performance
    test_loss = test(model=model, test_set=dataset)
    print(f'Test loss: {test_loss}')
    




def main():
    # evaluate test performance on given directory and model file
    run_test_cnn()
    

if __name__ == "__main__":
    main()




