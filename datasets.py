import torch
import copy
from torch.utils.data import Dataset
import pandas as pd

class CustomTimeSeriesDataset(Dataset):
    
    def __init__(self,file):
        df = pd.read_csv(file)
        print(f'Loaded DataFrame: {df}')
        
        self.angles_in_radians = True
        self.__process_scalar_data(df)
        self.__process_metadata(df)
        self.__process_time_series_data(df)
        
    def __len__(self):
        return self.num_rows
        
    def __getitem__(self,idx):
        sample_input_scalars = self.input_scalars[idx,:]
        sample_input_time_series = self.input_time_series[idx,:,:]
        sample_target_time_series = self.target_time_series[idx,:,:]
        return sample_input_scalars, sample_input_time_series, sample_target_time_series
    
    def subset(self,idx_list):
        idx_list.sort()
        idx = torch.tensor(idx_list)
        subset = copy.deepcopy(self)
        subset.num_rows = len(idx_list)
        subset.angles_in_radians = self.angles_in_radians
        subset.dataset_name = [self.dataset_name[x] for x in idx_list]
        # scalar inputs
        subset.input_scalars = self.input_scalars[idx,:]
        # subject identifiers
        subset.subject_ids = [self.subject_ids[x] for x in idx_list]
        subset.unique_subject_ids = list(set(subset.subject_ids))
        subset.unique_subject_ids.sort()
        # time series
        subset.input_time_series = self.input_time_series[idx]
        subset.target_time_series = self.target_time_series[idx]
        return subset
        
    
    def kfold(self,k):
        fractions = [1 for x in range(k)]
        idxs = self.get_split_indices(fractions)
        #subsets = []
        #for i in range(k):
        #    subsets.append(self[idxs[i]])
        return idxs
    
    
    # get indices that are split according to a given list of fractions, while making sure no data rows of the same subject are placed in different subsets
    # note that all subjects in this context are also from different datasets (ensured through a "subject identifier" that contains the name of the dataset), i.e., if dataset A and dataset B both have a subject called "Subject1", then the code will not be tricked because the subject identifiers will be "A_Subject1" and "B_Subject1"
    def get_split_indices(self, fractions=(70,20,10)):
        
        # inner function for inserting indices into a subset; because this is used in two different points in the outer function, we'll just define it once here
        def populate(target_index):
            # define subset_sizes and subset_indices as nonlocal so that we can modify them inside this inner function instead of just reading them
            nonlocal subset_sizes
            nonlocal subset_indices
            subset_sizes[target_index] += n_selected_rows
            for row in range(len(idx_rows)):
                if idx_rows[row]:
                    subset_indices[target_index].append(row)
        
        # calculate normalized fractions in case the user gives the fractions as percentage points or another format that doesn't add up to 1.0
        fractions_normalized = [float(x)/sum(fractions) for x in fractions]
        
        # we track the number of data rows in each of the subsets with subset_sizes, while subset_indices stores the corresponding row indices
        subset_sizes = [0 for x in fractions]
        subset_indices = [[] for x in fractions]
        # for each subject in a random order, place the trials of that subject in a subset, filling each subset one at a time
        n_subjects = len(self.unique_subject_ids)
        # shuffle the indices randomly so we don't go through the subjects in order
        idx_shuffled = torch.randperm(n_subjects)
        for i in idx_shuffled:
            # get the unique identifier of the current subject
            current_subject = self.unique_subject_ids[i]
            # calculate how many rows of data exist for that subject
            idx_rows = [x==current_subject for x in self.subject_ids]
            # count the number of the data rows for the subject
            n_selected_rows = idx_rows.count(True)
            
            can_fit_in_subset = False
            # check if there is available space in any of the subsets
            for j in range(len(fractions)):
                if subset_sizes[j]+n_selected_rows <= self.num_rows*fractions_normalized[j]:
                    can_fit_in_subset = True
                    populate(j)
                    break
            
            # if there was not enough space in any of the subjects, then we find the subset that is the least full, and append to it
            if not can_fit_in_subset:
                # identify the subset with the most room remaining
                occupation = [float(x)/self.num_rows*fractions_normalized[j] for x in subset_sizes]
                idx_min = torch.argmin(torch.tensor(occupation))
                populate(idx_min)
        
        [x.sort() for x in subset_indices]
        
        print(f'Final subset sizes: {subset_sizes}')
        print(f'Fractions of the final subset sizes: {[round(float(x)/sum(subset_sizes),2) for x in subset_sizes]}')
        return subset_indices
    
    # get the number of features (input and target vectors) as a 2-element tuple
    def get_num_features(self):
        n_input_features = len(self.input_time_series[0,:,0])
        n_target_features = len(self.target_time_series[0,:,0])
        return n_input_features, n_target_features
    
    # process information like dataset name
    def __process_metadata(self, data_frame):
        self.dataset_name = data_frame['dataset']
        unique_datasets = set([x for x in self.dataset_name])
        print(f'Unique datasets in the data: {unique_datasets}')
        
        self.subject_ids = [x+'_'+y for x,y in zip(data_frame['dataset'], data_frame['subject_name'])]
        self.unique_subject_ids = list(set(self.subject_ids))
        # NOTE: we must sort the list if we want reproducible results while using it because set() has non-deterministic behaviour
        self.unique_subject_ids.sort()
        print(f'Unique subject IDs: {self.unique_subject_ids}')
    
    # process all scalar-format data like subject demographics
    def __process_scalar_data(self,data_frame):
        # store the number of rows
        self.num_rows = len(data_frame.index)
        
        # process subject demographics
        body_mass = self.__series_to_tensor(data_frame['body_mass'])
        body_height = torch.tensor(self.__convert_height_to_meters(data_frame['body_height']))
        age = self.__series_to_tensor(data_frame['age'])
        sex = self.__map_sex_to_binary(data_frame['sex'])
        
        # process gait info
        gait_speed = self.__series_to_tensor(data_frame['gait_speed'])
        gait_cycle_duration = self.__series_to_tensor(data_frame['gait_cycle_duration'])
        
        self.input_scalars = torch.cat((body_mass.unsqueeze(1), body_height.unsqueeze(1), age.unsqueeze(1), sex.unsqueeze(1)), 1)
        print(f'Scalar inputs: {self.input_scalars}')
    
    # process all time series format data like input kinematics and output kinetics
    def __process_time_series_data(self,data_frame):
        
        
        unique_cols = self.__find_unique_column_labels(data_frame)
        
        data = self.__time_series_data_to_tensor(data_frame,unique_cols)
        
        idx_jointangles = []
        idx_jointmoments = []
        for i_col in range(len(unique_cols)):
            if '_jointangles_' in unique_cols[i_col]:
                idx_jointangles.append(i_col)
            elif '_jointmoments_' in unique_cols[i_col]:
                idx_jointmoments.append(i_col)
            
        print(f'Index of joint angles: {idx_jointangles}')
        print(f'Index of joint moments: {idx_jointmoments}')
        
        # split the data tensor to inputs and targets knowing that the first 12 labels are for joint kinematics (inputs) and the remaining 9 for joint moments (targets)
        # convert kinematics time series from degrees to radians, which sort of normalizes the data
        if self.angles_in_radians:
            self.input_time_series = torch.deg2rad(data[:,idx_jointangles,:])
        else:
            self.input_time_series = data[:,idx_jointangles,:]
        self.target_time_series = data[:,idx_jointmoments,:]

    # get a list of unique column labels of time series variables
    def __find_unique_column_labels(self, dataframe):
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
    def __time_series_data_to_tensor(self, dataframe, unique_cols):
        # read all time series data into a tensor
        data = torch.empty(len(dataframe.index), len(unique_cols), 101)
        for s in range(len(dataframe.index)):
            for i_col in range(len(unique_cols)):
                for i in range(0,101):
                    current_col = unique_cols[i_col] + '_' + str(i+1)
                    data[s, i_col, i] = dataframe[current_col][s]
        return data
    
    # if height is given in millimeters or centimeters, convert it to meters
    def __convert_height_to_meters(self,heights):
        heights_float = [float(height) for height in heights]
        for i in range(len(heights_float)):
            value = heights_float[i]
            if value > 1000:
                value /= 1000.0
            elif value > 100:
                value /= 100.0
            heights_float[i] = value
        return heights_float
    
    # convert from a pandas Series to a torch Tensor
    def __series_to_tensor(self,series):
        return torch.tensor(series.values)
    
    # map sex/gender, which may be given as string 'F' or 'M', to binary integers 0 and 1 so that it can be processed numerically later by the neural network
    def __map_sex_to_binary(self,sexes):
        if sexes.dtype == 'object':
            return torch.tensor([int(sex=='F') for sex in sexes])
        # if sex is not a string (represented by the 'object' dtype), we assume it is already an integer and just convert to tensor
        else:
            return self.__series_to_tensor(sexes)
            