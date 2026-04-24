import torch
import copy
from torch.utils.data import Dataset
import pandas as pd
from options import kinematics_bounds, kinetics_bounds, scalar_bounds, kinetics_variable, included_generalized_coordinates

class CustomTimeSeriesDataset(Dataset):
    """"
    A class that reads a dataset containing input and target data (including subject demographics scalars and time series of kinematics and kinetics) and provides it for Pytorch algorithms.
    Used for both the training and the test set.
    """
    
    def __init__(self,file: str):
        """
        When called using a file path as an argument, reads that filepath with pandas and extracts the data.
        """

        # max number of samples to read; if this is exceeded, nothing is read any longer
        # default 50000 should be enough; using a number like 300 will make prototyping faster
        self.max_rows = 50000
        
        # this will store the length of a single sequence of a single feature in a sample; None indicates it hasn't been set yet
        self.sequence_length = None
        
        # whether to normalize the data using pre-defined boundaries
        self.normalize_data = True
        # whether to center individual joint angle time series so that their mean is 0
        self.center_kinematics_time_series = True
        if self.normalize_data:
            self.__initialize_bounds()
        
        
        # only read the relevant columns from the CSV to save a little time (the columns defined through the included_generalized_coordinates and kinetics_variable variables in options.py)
        col_labels = ['dataset', 'subject_name', 'trial_name', 'row', 'age', 'sex', 'body_height', 'body_mass', 'leg']
        for gc in included_generalized_coordinates:
            col_labels.append(f'padded_timeseries_jointangles_{gc}')
        if kinetics_variable in ['summed', 'medial', 'lateral', 'patellofemoral']:
            col_labels.append(f'padded_timeseries_contactforces_kcf_{kinetics_variable}')
        elif kinetics_variable in ['ekfm', 'ekam']:
            col_labels.append(f'padded_timeseries_jointmoments_{kinetics_variable}')
        df = pd.read_csv(file, nrows=self.max_rows, usecols=lambda c: any([(x in c) for x in col_labels]))
        print(f'Loaded DataFrame: {df}')
        
        self.__process_scalar_data(df)
        self.__process_metadata(df)
        self.__process_time_series_data(df)
        
    def __len__(self) -> int:
        """
        Returns the number of samples (rows) in the data. For example, if the data contains 100 walking trials, returns 100.
        """
        return self.num_rows
        
    def __getitem__(self,idx: int) -> tuple:
        """"
        Given an integer index, returns a tuple containing the tensors of input subject demographics (scalars), input kinematics (time series), and target time kinetics (time series).
        In other words, for an index i starting at 0, returns the i'th data sample.
        """
        sample_input_scalars = self.input_scalars[idx,:]
        sample_input_time_series = self.input_time_series[idx,:,:]
        sample_target_time_series = self.target_time_series[idx,:,:]
        return sample_input_scalars, sample_input_time_series, sample_target_time_series
    
    # initialize the lower and upper bounds for inputs and outputs, used in normalizing the values
    def __initialize_bounds(self) -> None:
        """
        Populates the dictionaries containing the lower and upper bounds of various input and output variables. Note that while these map common values of the variables to range mostly between 0 and 1 using min-max normalization, it is not strict, i.e., some values may be below 0 or above 1.
        This loose min-max normalization is done to help the neural networks optimize its weights, which is more stable when inputs are not large numbers.
        """
        # kinematics bounds are based on the joint ranges of motion from the musculoskeletal model, but restricted further to map common values to [0, 1]
        self.kinematics_bounds = kinematics_bounds
        self.scalar_bounds = scalar_bounds
        self.kinetics_bounds = kinetics_bounds
    
    # return the length of a sequence (e.g., if you have 1500 samples (rows) of 3 different loading feature time series that are each 101 data points long, this returns 101)
    # in other words, the order is (row, feature, time point)
    # note that in networks, the order is different: (batch, time point, feature)
    def get_sequence_length(self) -> int:
        """
        Returns the length of an input or target sequence (a single time series). Current implementation assumes that inputs (kinematics time series) and targets (kinetics time series) have the same sequence length.
        For instance, if inputs and targets are time series with 250 time points, then this methods returns 250.
        """
        return self.sequence_length
    
    # get a subset (another dataset object) of this dataset that only contains data samples in the given indices
    def subset(self,idx_list: int):
        """
        Given a list of sample indices to include, returns a dataset that is a subset of the current dataset. Useful for picking indices for k-fold cross-validation.
        """
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
    def kfold(self,k: int):
        """
        Given the number of folds k, returns a list of lists containing the sample indices in the folds. A slightly more convenient way than calling get_split_indices() directly.
        """
        fractions = [1 for x in range(k)]
        idxs = self.get_split_indices(fractions)
        return idxs
    
    
    # get indices that are split according to a given list of fractions, while making sure no data rows of the same subject are placed in different subsets
    # note that all subjects in this context are also from different datasets (ensured through a "subject identifier" that contains the name of the dataset), i.e., if dataset A and dataset B both have a subject called "Subject1", then the code will not be tricked because the subject identifiers will be "A_Subject1" and "B_Subject1"
    def get_split_indices(self, fractions=(70,20,10)):
        """
        Given a tuple of fractions of different subsets, returns a list of lists containing the indices of those subsets. Useful for dividing the data into k folds or train-validation-test subsets.
        Ensures that the samples from a single subject can only be placed in a single subset, and tries to allocate data samples in the subsets evenly.
        """
        
        # inner function for inserting indices into a subset; because this is used in two different points in the outer function, we'll just define it once here
        def populate(target_index):
            # define subset_sizes and subset_indices as nonlocal so that we can modify them inside this inner function instead of just reading them
            nonlocal subset_sizes
            nonlocal subset_indices
            subset_sizes[target_index] += n_selected_rows
            subset_subjects[target_index] += 1
            for row in range(len(idx_rows)):
                if idx_rows[row]:
                    subset_indices[target_index].append(row)
        
        # calculate normalized fractions in case the user gives the fractions as percentage points or another format that doesn't add up to 1.0
        fractions_normalized = [float(x)/sum(fractions) for x in fractions]
        
        # we track the number of data rows in each of the subsets with subset_sizes, while subset_indices stores the corresponding row indices
        subset_sizes = [0 for x in fractions]
        subset_subjects = [0 for x in fractions]
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
        print(f'Final number of subjects per fraction: {subset_subjects}')
        print(f'Fractions of the final subset sizes: {[round(float(x)/sum(subset_sizes),2) for x in subset_sizes]}')
        return subset_indices
    
    # get the number of features (input and target vectors) as a 2-element tuple
    def get_num_features(self) -> tuple:
        """
        Returns a tuple of the number of features in the input and target data. For instance, if the kinematics input consists of the time series of 13 different joint angles and the kinetics input consists of the time series of 3 contact force variables, returns (13, 3).
        """
        n_input_features = len(self.input_time_series[0,:,0])
        n_target_features = len(self.target_time_series[0,:,0])
        return n_input_features, n_target_features
    
    # process information like dataset name
    def __process_metadata(self, data_frame):
        """
        Reads metadata such as the datasets and subjects associated with each data sample from the pandas DataFrame.
        """
        self.dataset_name = data_frame['dataset']
        unique_datasets = set([x for x in self.dataset_name])
        print(f'Unique datasets in the data: {unique_datasets}')
        
        self.subject_ids = [str(x)+'_'+str(y) for x,y in zip(data_frame['dataset'], data_frame['subject_name'])]
        self.unique_subject_ids = list(set(self.subject_ids))
        # NOTE: we must sort the list if we want reproducible results while using it because set() has non-deterministic behaviour
        self.unique_subject_ids.sort()
        print(f'Unique subject IDs: {self.unique_subject_ids}')
    
    def __normalize(self, data, key: str):
        """
        Implements min-max normalization to scale data so that the given minimum and maximum boundaries are mapped to 0 and 1. Does not enforce that all data must be in the range [0,1].
        For example, if mass is scaled between 50 and 150, then 50 kgs will be mapped to 0, 150 kgs will be mapped to 1, 100 kgs will be mapped to 0.5, values below 50 kgs will be mapped to negative numbers, and values above 150 kgs will be mapped to numbers greater than 1.
        """
        is_scalar = False
        is_kinematics = False
        #is_kinetics = False
        if key in self.scalar_bounds.keys():
            bounds = self.scalar_bounds
            is_scalar = True
        elif key in self.kinematics_bounds.keys():
            bounds = self.kinematics_bounds
            is_kinematics = True
        elif key in self.kinetics_bounds.keys():
            bounds = self.kinetics_bounds
            #is_kinetics = True
        else:
            raise Exception(f'Error while normalizing! Key {key} not found in any of the boundary value dictionaries!')
        
        # if the normalizable data is a scalar, we do standard normalization
        # otherwise, it is a time series and we do not want to offset the trailing zeros; hence, we save their indices and set them back to zero after normalization
        if is_scalar:
            scaled_data = (data - bounds[key][0]) / (bounds[key][1] - bounds[key][0])
        else:
            zeros = torch.abs(data) < 1e-12
            #if is_kinematics:
            scaled_data = (data - bounds[key][0]) / (bounds[key][1] - bounds[key][0])
            #elif is_kinetics:
            #    scaled_data = data / self.mean_of_target_maxima
            scaled_data[zeros] = 0.0
            if self.center_kinematics_time_series and is_kinematics:
                center = torch.mean(scaled_data[~zeros])
                scaled_data[~zeros] -= center
        
        return scaled_data
    
    # process all scalar-format data like subject demographics
    def __process_scalar_data(self,data_frame) -> None:
        """
        Reads subject demographics from the pandas DataFrame, scales them if required, and places them in a torch Tensor.
        Also reads the number of rows (samples) in the dataset from the pandas DataFrame so that it can be accessed later with len().
        """
        
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
    def __process_time_series_data(self,data_frame) -> None:
        """
        Reads input kinematics and target kinetics from the pandas DataFrame and normalizes them if required.
        """
        
        unique_cols = self.__find_unique_column_labels(data_frame)
        
        data = self.__time_series_data_to_tensor(data_frame,unique_cols)
        
        labels_jointangles = []
        labels_contactforces = []
        
        idx_jointangles = []
        idx_jointmoments = []
        idx_contactforces = []
        # loop through indices of unique columns found in the dataset file
        for i_col in range(len(unique_cols)):
            # if a column is a joint angle time series and contains a label defined in included_generalized_coordinates, append the index of the column to idx_jointangles and the name of the column to labels_jointangles so they can be used to construct the input time series tensor and normalize it according to label names
            if '_jointangles_' in unique_cols[i_col]:
                for gc in included_generalized_coordinates:
                    if gc in unique_cols[i_col]:
                        idx_jointangles.append(i_col)
                        labels_jointangles.append(unique_cols[i_col])
                        break
            elif f'_jointmoments_{kinetics_variable}' in unique_cols[i_col]:
                idx_jointmoments.append(i_col)
            elif f'_contactforces_kcf_{kinetics_variable}' in unique_cols[i_col]:
                idx_contactforces.append(i_col)
                labels_contactforces.append(unique_cols[i_col])
                print(unique_cols[i_col])
                #break # This break here tells us to stop reading columns after finding the first contact force column. Contact forces are last in column names, so despite breaking out of the loop, we'll have read the joint kinematics by then. Summed contact force should be the first contact force column, so we'll read only that instead of the others (medial/lateral/patellofemoral).
            
        #print(f'Index of joint angles: {idx_jointangles}')
        #print(f'Index of joint moments: {idx_jointmoments}')
        print(f'Index of contact forces: {idx_contactforces}, labels: {labels_contactforces}')
        
        # split the data tensor to inputs and targets knowing that the first labels are for joint kinematics (inputs) and the remaining for joint moments or contact forces (targets)
        self.input_time_series = data[:,idx_jointangles,:]
        if len(idx_jointmoments) > 0:
            self.target_time_series = data[:,idx_jointmoments,:]
        elif len(idx_contactforces) > 0:
            self.target_time_series = data[:,idx_contactforces,:]
        else:
            raise Exception('Target time series could not be identified in Dataset::__process_time_series_data!')
        
        # calculate the maximum knee joint loading during each stance phase, and print their mean; currently not used for anything, but could be used to scale kinetics
        self.mean_of_target_maxima = torch.mean(torch.max(self.target_time_series, dim=2)[0])
        print(f'target time series mean of maxima: {self.mean_of_target_maxima}')
        
        if self.normalize_data:
            for i, label in enumerate(labels_jointangles):
                for key in self.kinematics_bounds.keys():
                    if key in label:
                        self.input_time_series[:,i,:] = self.__normalize(self.input_time_series[:,i,:], key)
                        break
            for i, label in enumerate(labels_contactforces):
                for key in self.kinetics_bounds.keys():
                    if key in label:
                        self.target_time_series[:,i,:] = self.__normalize(self.target_time_series[:,i,:], key)
                        break

    # get a list of unique column labels of time series variables
    def __find_unique_column_labels(self, dataframe) -> list[str]:
        """
        Returns a list of unique column labels from the pandas DataFrame. Treats column labels with the same body but different suffix as the same label.
        For example, "knee_angle_r_1", "knee_angle_r_2", ..., "knee_angle_r_99", "knee_angle_r_100" are all treated as a unique column label named "knee_angle_r".
        """
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
    def __time_series_data_to_tensor(self, dataframe, unique_cols: list[str]):
        """
        Returns a torch Tensor with the time series data read from the pandas DataFrame.
        """
        # read all time series data into a tensor
        data = torch.empty(len(dataframe.index), len(unique_cols), self.sequence_length)
        # for each row (as indicated by DataFrame.index)
        for s in range(len(dataframe.index)):
            # for each unique column
            for i_col in range(len(unique_cols)):
                for i in range(0,self.sequence_length):
                    current_col = unique_cols[i_col] + '_' + str(i+1)
                    data[s, i_col, i] = dataframe[current_col][s]
        return data
    
    # if height is given in millimeters or centimeters, convert it to meters
    def __convert_height_to_meters(self,heights) -> list[float]:
        """
        Returns height from the subject demographics as meters after determining the suitable conversion.
        """
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
        """
        If sex from subject demographics is a string, maps it to binary integers [0=male, 1=female] and returns the binary integers in a tensor.
        """
        if sexes.dtype == 'object':
            return torch.tensor([int(sex=='F') for sex in sexes])
        # if sex is not a string (represented by the 'object' dtype), we assume it is already an integer and just convert to tensor
        else:
            return self.__series_to_tensor(sexes)
            