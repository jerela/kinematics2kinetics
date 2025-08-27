import torch
from torch.utils.data import Dataset
import pandas as pd



class CustomTimeSeriesDataset(Dataset):
    
    def __init__(self,file):
        df = pd.read_csv(file)
        print(df)
        
        unique_cols = self.find_unique_column_labels(df)
                    
        print(unique_cols)
        
        data = self.time_series_data_to_tensor(df,unique_cols)
        print(data)
        
        idx_jointangles = []
        idx_jointmoments = []
        for i_col in range(len(unique_cols)):
            if '_jointangles_' in unique_cols[i_col]:
                idx_jointangles.append(i_col)
            elif '_jointmoments_' in unique_cols[i_col]:
                idx_jointmoments.append(i_col)
            
        print(f'Idx of joint angles: {idx_jointangles}')
        print(f'Idx of joint moments: {idx_jointmoments}')
        
        # split the data tensor to inputs and targets knowing that the first 12 labels are for joint kinematics (inputs) and the remaining 9 for joint moments (targets)
        #self.inputs = data[:,:n_jointangles,:]
        self.inputs = data[:,idx_jointangles,:]
        #self.targets = data[:,n_jointangles:,:]
        self.targets = data[:,idx_jointmoments,:]
        
        # store the number of rows
        self.rows = len(df.index)
        
        
    def __len__(self):
        return self.rows
        
    def __getitem__(self,idx):
        sample_input = self.inputs[idx,:,:]
        sample_target = self.targets[idx,:,:]
        return sample_input, sample_target
        
    # get the number of features (input and target vectors) as a 2-element tuple
    def get_num_features(self):
        n_input_features = len(self.inputs[0,:,0])
        n_target_features = len(self.targets[0,:,0])
        return n_input_features, n_target_features

    # get a list of unique column labels of time series variables
    def find_unique_column_labels(self, dataframe):
        # construct a list of unique column labels (i.e., "HipFlx_12" and "HipFlx_45" become just "HipFlx")
        cols = list(dataframe.columns.values)
        unique_cols = []
        for col in cols:
            # we look for "timeseries_", which is our indicator of time-series formatted data
            if col.find('timeseries_') != -1:
                # we use rfind to find the index of the last '_', which we assume is the underscore preceding the index of the value in the time series
                col_name = col[0:col.rfind('_')]
                if col_name not in unique_cols:
                    unique_cols.append(col_name)
        return unique_cols
    
    # insert the read time series into a single tensor
    def time_series_data_to_tensor(self, dataframe, unique_cols):
        # read all time series data into a tensor
        data = torch.empty(len(dataframe.index), len(unique_cols), 101)
        for s in range(len(dataframe.index)):
            #print(s)
            for i_col in range(len(unique_cols)):
                for i in range(0,101):
                    current_col = unique_cols[i_col] + '_' + str(i+1)
                    data[s, i_col, i] = dataframe[current_col][s]
        return data
                    