import torch
import copy
from torch.utils.data import Dataset
import pandas as pd

class CustomTimeSeriesDataset(Dataset):
    
    def __init__(self,file):

        # max number of samples to read; if this is exceeded, nothing is read any longer
        self.max_rows = 50000
        
        # this will store the length of a single sequence of a single feature in a sample; None indicates it hasn't been set yet
        self.sequence_length = None
        
        self.normalize_data = True
        if self.normalize_data:
            self.__initialize_bounds()
        
        df = pd.read_csv(file,nrows=self.max_rows)
        print(f'Loaded DataFrame: {df}')
        
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
    
    # initialize the lower and upper bounds for inputs and outputs, used in normalizing the values
    def __initialize_bounds(self):
        
        # kinematics bounds are based on the joint ranges of motion from the musculoskeletal model, but restricted further to map common values to [-1, 1]
        self.kinematics_bounds = { # instead of using joint ranges of motion from the musculoskeletal model, let's assume plausible ranges during the stance phase
            'pelvis_tilt': (-45.0, 45.0),#(-90.0, 90.0),
            'pelvis_list': (-45.0, 45.0),#(-90.0, 90.0),
            'pelvis_rotation': (-180.0, 180.0), #unclamped, but let's go with -180 to 180 degrees
            'lumbar_extension': (-45.0, 45.0),#(-90.0, 90.0),
            'lumbar_bending': (-45.0, 45.0),#(-90.0, 90.0),
            'lumbar_rotation': (-45.0,45.0),#(-90.0, 90.0),
            'hip_flexion': (-30.0,60.0),#(-30.0, 120.0),
            'hip_adduction': (-20.0,20.0),#(-50.0, 30.0),
            'hip_rotation': (-25.0,25.0),#(-40.0, 40.0),
            'knee_angle': (0.0,60.0),#(0.0, 120.0),
            'ankle_angle': (-25.0,30.0),#(-40.0, 30.0)
        }
        
        self.scalar_bounds = {
            'body_mass': (40.0, 150.0),
            'body_height': (1.4, 2.2),
            'age': (18.0, 80.0)
        }
        
        self.kinetics_bounds = {
            'kcf_summed': (0.0, 6000.0),
            'kcf_medial': (0.0, 4000.0),
            'kcf_lateral': (0.0,3000.0),
            'kcf_patellofemoral': (0.0,4000.0)
        }
    
    # return the length of a sequence (e.g., if you have 1500 samples (rows) of 3 different loading feature time series that are each 101 data points long, this returns 101)
    # in other words, the order is (row, feature, time point)
    # note that in networks, the order is different: (batch, time point, feature)
    def get_sequence_length(self):
        #return len(self.target_time_series[0,0,:])
        return self.sequence_length
    
    # get a subset (another dataset object) of this dataset that only contains data samples in the given indices
    def subset(self,idx_list):
        idx_list.sort()
        idx = torch.tensor(idx_list)
        subset = copy.deepcopy(self)
        subset.num_rows = len(idx_list)
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
        
    # get indices for k approximately equally sized folds
    def kfold(self,k):
        fractions = [1 for x in range(k)]
        idxs = self.get_split_indices(fractions)
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
        
        self.subject_ids = [str(x)+'_'+str(y) for x,y in zip(data_frame['dataset'], data_frame['subject_name'])]
        self.unique_subject_ids = list(set(self.subject_ids))
        # NOTE: we must sort the list if we want reproducible results while using it because set() has non-deterministic behaviour
        self.unique_subject_ids.sort()
        print(f'Unique subject IDs: {self.unique_subject_ids}')
    
    def __normalize(self, data, key):
        if key in self.scalar_bounds.keys():
            bounds = self.scalar_bounds
        elif key in self.kinematics_bounds.keys():
            bounds = self.kinematics_bounds
        elif key in self.kinetics_bounds.keys():
            bounds = self.kinetics_bounds
        else:
            raise Exception(f'Error while normalizing! Key {key} not found in any of the boundary value dictionaries!')
        return (data - bounds[key][0]) / (bounds[key][1] - bounds[key][0])
    
    # process all scalar-format data like subject demographics
    def __process_scalar_data(self,data_frame):
        
        # store the number of rows
        self.num_rows = len(data_frame.index)
        
        # process subject demographics
        body_mass = self.__series_to_tensor(data_frame['body_mass']).to(torch.float32)
        body_height = torch.tensor(self.__convert_height_to_meters(data_frame['body_height'])).to(torch.float32)
        age = self.__series_to_tensor(data_frame['age']).to(torch.float32)
        sex = self.__map_sex_to_binary(data_frame['sex']).to(torch.float32)
        
        if self.normalize_data:
            body_mass = self.__normalize(body_mass,'body_mass')
            body_height = self.__normalize(body_height,'body_height')
            age = self.__normalize(age,'age')
        
        # process gait info
        #gait_speed = self.__series_to_tensor(data_frame['gait_speed'])
        #gait_cycle_duration = self.__series_to_tensor(data_frame['gait_cycle_duration'])
        
        self.input_scalars = torch.cat((body_mass.unsqueeze(1), body_height.unsqueeze(1), age.unsqueeze(1), sex.unsqueeze(1)), 1)
        print(f'Scalar inputs: {self.input_scalars}')
    
    # process all time series format data like input kinematics and output kinetics
    def __process_time_series_data(self,data_frame):
        
        
        unique_cols = self.__find_unique_column_labels(data_frame)
        
        data = self.__time_series_data_to_tensor(data_frame,unique_cols)
        
        labels_jointangles = []
        labels_contactforces = []
        
        idx_jointangles = []
        idx_jointmoments = []
        idx_contactforces = []
        for i_col in range(len(unique_cols)):
            if '_jointangles_' in unique_cols[i_col]:
                idx_jointangles.append(i_col)
                labels_jointangles.append(unique_cols[i_col])
            elif '_jointmoments_' in unique_cols[i_col]:
                idx_jointmoments.append(i_col)
            elif '_contactforces_' in unique_cols[i_col]:
                idx_contactforces.append(i_col)
                labels_contactforces.append(unique_cols[i_col])
                break
            
        #print(f'Index of joint angles: {idx_jointangles}')
        #print(f'Index of joint moments: {idx_jointmoments}')
        
        # split the data tensor to inputs and targets knowing that the first 12 labels are for joint kinematics (inputs) and the remaining 9 for joint moments (targets)
        self.input_time_series = data[:,idx_jointangles,:]
        if len(idx_jointmoments) > 0:
            self.target_time_series = data[:,idx_jointmoments,:]
        elif len(idx_contactforces) > 0:
            self.target_time_series = data[:,idx_contactforces,:]
        else:
            raise Exception('Target time series could not be identified in Dataset::__process_time_series_data!')
        
        if self.normalize_data:
            for i, label in enumerate(labels_jointangles):
                for key in self.kinematics_bounds.keys():
                    if key in label:
                        self.input_time_series[:,i,:] =self.__normalize(self.input_time_series[:,i,:], key)
                        break
            for i, label in enumerate(labels_contactforces):
                for key in self.kinetics_bounds.keys():
                    if key in label:
                        self.target_time_series[:,i,:] =self.__normalize(self.target_time_series[:,i,:], key)
                        break

    # get a list of unique column labels of time series variables
    def __find_unique_column_labels(self, dataframe):
        # construct a list of unique column labels (i.e., "HipFlx_12" and "HipFlx_45" become just "HipFlx")
        cols = list(dataframe.columns.values)
        unique_cols = []
        max_length = 0
        for col in cols:
            # we look for "timeseries_", which is our indicator of time-series formatted data
            if col.find('timeseries_') != -1:
                # we use rfind to find the index of the last '_', which we assume is the underscore preceding the index of the value in the time series
                col_name = col[0:col.rfind('_')]
                length = int(col[col.rfind('_')+1:])
                if length > max_length:
                    max_length = length
                if col_name not in unique_cols:
                    unique_cols.append(col_name)
        self.sequence_length = max_length
        return unique_cols
    
    # insert the read time series into a single tensor
    def __time_series_data_to_tensor(self, dataframe, unique_cols):
        # read all time series data into a tensor
        data = torch.empty(len(dataframe.index), len(unique_cols), self.sequence_length)
        for s in range(len(dataframe.index)):
            for i_col in range(len(unique_cols)):
                for i in range(0,self.sequence_length):
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
            