import torch
import torch.nn as nn
import torch.nn.functional as F

from options import lstm_hidden_size, lstm_num_layers, lstm_bidirectional

# mean square error function that puts less emphasis on values at the beginning and the end of the time series, to account for the high simulation nose in the beginning and end of contact force time series
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define the kernel size of the convolution filter
        self.window_size = 9
    
    # because the data may be padded, we use convolution to make sure the weights of the loss mitigate the effects of padding while also applying some mitigation at both ends of the padless time series
    def __update_weights_mask(self, targets):
        
        # create a mask that is 1 for non-zero values in the target time series, and 0 otherwise
        nonzeros = (targets != 0).float()
        
        # get the number of target channels or features
        target_shape = targets.shape        
        n_channels = target_shape[1]
        
        # construct the filters so their elements sum to 1 (normalized to 1)
        filters = torch.ones((n_channels, n_channels, self.window_size))/self.window_size
        padding_size = int((self.window_size-1)/2)
        
        # calculate the final weights mask
        self.weights_mask = F.conv1d(nonzeros, weight=filters, padding=padding_size)
        
        #print(f'Weights: {self.weights_mask}')
        #fig = plt.figure()
        #fig.add_subplot(121)
        #plt.plot(self.weights_mask[0,:,:].squeeze(0),'x')
        #fig.add_subplot(122)
        #plt.plot(targets[0,:,:].squeeze(0))
        #plt.show()
        
    def forward(self, inputs, targets):
        self.__update_weights_mask(targets)
        return torch.mean( ((inputs-targets)**2) * self.weights_mask )


# various network architectures that map input kinematic time series to output kinetic time series

# standard feedforward neural network, doesn't perform well
class KineticsFFN(nn.Module):
    def __init__(self, num_input_vectors, num_output_vectors, len_sequence, name='KineticsFFN'):
        super().__init__()
        
        self.model_name = name
        
        self.sequence_length = len_sequence
        
        self.num_output_vectors = num_output_vectors
        
        self.fc1 = nn.Linear(num_input_vectors*self.sequence_length, num_output_vectors*self.sequence_length)
        self.fc2 = nn.Linear(1024, num_output_vectors*self.sequence_length)
        self.flattener = nn.Flatten()
        
    def forward(self,inputs):
        
        scalars, time_series = inputs
        batch_size = scalars.shape[0]
        
        x = self.flattener(time_series)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        prediction = x.reshape(batch_size,self.sequence_length,self.num_output_vectors)
                
        return prediction


# convolutional neural network
class KineticsCNN(nn.Module):
    def __init__(self, num_input_vectors, num_output_vectors, kernel_size=1, name='KineticsCNN'):
        super().__init__()
        
        self.model_name = name
        
        
        
        self.c1 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_input_vectors,
                out_channels=num_output_vectors*8,
                kernel_size=kernel_size,
                stride=1,
                padding='same',
                dilation=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm1d(num_features=num_output_vectors*8),
            nn.ReLU()
        )
        
        self.c2 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_output_vectors*8,
                out_channels=num_output_vectors*16,
                kernel_size=kernel_size,
                stride=1,
                padding='same',
                dilation=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm1d(num_features=num_output_vectors*16),
            nn.ReLU()
        )
        
        self.c3 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_output_vectors*16,
                out_channels=num_output_vectors*32,
                kernel_size=kernel_size,
                stride=1,
                padding='same',
                dilation=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm1d(num_features=num_output_vectors*32),
            nn.ReLU()
        )
        
        self.c4 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_output_vectors*32,
                out_channels=num_output_vectors,
                kernel_size=kernel_size,
                stride=1,
                padding='same',
                dilation=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm1d(num_features=num_output_vectors),
            nn.ReLU()
        )

        
    def forward(self, inputs):
        scalars, time_series = inputs
        batch_size = scalars.shape[0]
        
        x = self.c1(time_series)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = x.permute(0,2,1)
        return x
        

# gated recurrent unit network
class KineticsGRU(nn.Module):
    def __init__(self, num_input_vectors, num_output_vectors, num_layers, name='KineticsGRU'):
        super().__init__()
        
        self.model_name = name
        
        self.gru = nn.GRU(
            input_size=num_input_vectors,
            hidden_size=num_output_vectors,
            batch_first=True,
            num_layers=num_layers,#4,
            bidirectional=lstm_bidirectional
        )
        
        # see explanation in forward()
        #self.coefficient = nn.Parameter(torch.rand((1), requires_grad=True), requires_grad=True)
        
    def forward(self,inputs):
        
        scalars, time_series = inputs
        
        # transform to the correct format that is defined by making batch_first=True in the LSTM constructor
        time_series_trans = time_series.permute(0,2,1)
                
        # pass the transposed input to the LSTM
        gru_out, temp = self.gru(time_series_trans)
        # lstm_out now contains the short-term memory values from each unrolled LSTM unit
        
        # we multiply all output values by a scalar coefficient to allow the prediction to match target values above 1 or below -1, in case the output of the main architecture returns values constrained to the range [-1,1]
        prediction = gru_out#*self.coefficient
        return prediction

# long short-term memory network
class KineticsLSTM(nn.Module):
    def __init__(self, num_input_vectors, num_output_vectors, name='KineticsLSTM'):
        super().__init__()
        
        self.model_name = name
        
        self.lstm = nn.LSTM(
            input_size=num_input_vectors,
            hidden_size=lstm_hidden_size,
            proj_size=num_output_vectors,
            batch_first=True,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional
        )
        
    def forward(self,inputs):
        
        scalars, time_series = inputs
        
        # transform to the correct format that is defined by making batch_first=True in the LSTM constructor
        time_series_trans = time_series.permute(0,2,1)
                
        # pass the transposed input to the LSTM
        lstm_out, temp = self.lstm(time_series_trans)
        # lstm_out now contains the short-term memory values from each unrolled LSTM unit
                
        prediction = lstm_out
        return prediction

# LSTM with demographic scalars
class DemographicKineticsLSTM(nn.Module):
    """
    The idea here is that unlike KineticsLSTM, this network incorporates information about scalar variables (mass, height, age, sex).
    They could be important for measures like moments and contact forces, where simple normalization of outputs by those values will not do. For instance, height does not affect the moments directly, but through its correlation with limb lengths.
    
    We could apply the scalars directly to the output time series by training a scaling factor for each time point and output vector (e.g., 101*9 scaling factors). However, because scaling factors of adjacent time points vary, this would result in a noisy output time series.
    Instead, the idea here is to find an effect of the scalar variables that affects the time series "smoothly". For this purpose, we fit N-th order polynomials to make adjustments to the LSTM output. This should fix the noise-like effect that is present with using individual scaling factors for each time point.
    This is also interpretable, because if it works, we can input subject demographics and get as output the N-th order polynomial approximated curve that separates the person representing those demogrpahics from the general kinetics time series.
    
    If N-th order polynomial fitting does not work, an alternative is to try Fourier series.
    """
    def __init__(self, num_input_vectors, num_output_vectors, name='DemographicKineticsLSTM'):
        super().__init__()
        
        self.model_name = name
        
        self.lstm = nn.LSTM(
            input_size=num_input_vectors,
            hidden_size=lstm_hidden_size,
            proj_size=num_output_vectors,
            batch_first=True,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional
        )
        self.fc = nn.Linear(4,num_output_vectors)
        
        
        #self.polynomial_degree = 1
        self.num_output_vectors = num_output_vectors
        
        #self.fc1 = nn.Linear(4,16)
        #self.fc2 = nn.Linear(16,(self.polynomial_degree+1)*num_output_vectors)
        
    def forward(self,inputs):
        
        scalars, time_series = inputs
        
        batch_size = scalars.shape[0]
        
        # transform to the correct format that is defined by making batch_first=True in the LSTM constructor
        time_series_trans = time_series.permute(0,2,1)
        
        # pass the transposed input to the LSTM
        lstm_out, temp = self.lstm(time_series_trans)
        # lstm_out now contains the short-term memory values from each unrolled LSTM unit
        
        x = self.fc(scalars)
        x = F.sigmoid(x)
        #print(f'shape of x: {x.shape}')
        x = x.unsqueeze(1)
        #print(f'shape of x: {x.shape}')
        #x = x.repeat(1,101,1)
        #print(f'shape of x: {x.shape}')
        
        
        # DIMC_COEFFS = (BATCH, POLYDEGREE, NUM_VECTORS=9)
        # DIM_Y = (BATCH, TIMEPOINTS=101, NUM_VECTORS=9)
        
        # polynomial construction
        #x = self.fc1(scalars)
        #x = F.sigmoid(x)
        #x = self.fc2(x)
        #x = F.sigmoid(x)
        #coefficients = x.reshape(batch_size, self.polynomial_degree+1, self.num_output_vectors)
        #y = torch.zeros((batch_size, 101, self.num_output_vectors))
        #for i in range(self.polynomial_degree+1):
        #    y += coefficients[:,i,:].reshape(batch_size, 1, self.num_output_vectors) * (torch.arange(1,102)**i).reshape(1,101,1).repeat(batch_size, 1, self.num_output_vectors)
            
        
        
        #print(f'shape of lstm_out: {lstm_out.shape}')
        
        output_scaled = lstm_out+x
        
        #print(f'shape of output: {output_scaled.shape}')
        
        prediction = output_scaled
        return prediction
