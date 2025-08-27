import torch
from torch.utils.data import Dataset
import pandas as pd



class CustomTimeSeriesDataset(Dataset):
    
    def __init__(self,file):
        df = pd.read_csv(file)
        print(df)
        
        # construct a list of unique column labels (i.e., "HipFlx_12" and "HipFlx_45" become just "HipFlx")
        cols = list(df.columns.values)
        unique_cols = []
        for col in cols:
            if col.find('_') != -1:
                col_name = col[0:col.find('_')]
                if col_name not in unique_cols:
                    unique_cols.append(col_name)
                    
        print(unique_cols)
        
        # read all data into a tensor
        data = torch.empty(len(df.index), len(unique_cols), 101)
        for s in range(len(df.index)):
            #print(s)
            for i_col in range(len(unique_cols)):
                for i in range(0,101):
                    current_col = unique_cols[i_col] + '_' + str(i+1)
                    data[s, i_col, i] = df[current_col][s]
                    
        print(data)
        
        # split the data tensor to inputs and targets knowing that the first 12 labels are for joint kinematics (inputs) and the remaining 9 for joint moments (targets)
        self.inputs = data[:,:12,:]
        self.targets = data[:,12:,:]
        
        # store the number of rows
        self.rows = len(df.index)
        
        
    def __len__(self):
        return self.rows
        
    def __getitem__(self,idx):
        sample_input = self.inputs[idx,:,:]
        sample_target = self.targets[idx,:,:]
        return sample_input, sample_target
        
    def get_num_features(self):
        n_input_features = len(self.inputs[0,:,0])
        n_target_features = len(self.targets[0,:,0])
        return n_input_features, n_target_features
