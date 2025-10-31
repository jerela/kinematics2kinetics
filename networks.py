import torch
import torch.nn as nn
import torch.nn.functional as F

from options import lstm_num_layers, lstm_bidirectional

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
        
        time_series = inputs
        batch_size = time_series.shape[0]
        
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
        
        
        
        self.c1 = nn.Conv1d(
            in_channels=num_input_vectors,
            out_channels=num_output_vectors*8,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn1 = nn.BatchNorm1d(num_features=num_output_vectors*8)
        self.relu = nn.ReLU()
        
        
        self.c2 = nn.Conv1d(
            in_channels=num_output_vectors*8,
            out_channels=num_output_vectors*16,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn2 = nn.BatchNorm1d(num_features=num_output_vectors*16)
        
        
        self.c3 = nn.Conv1d(
            in_channels=num_output_vectors*16,
            out_channels=num_output_vectors*32,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn3 = nn.BatchNorm1d(num_features=num_output_vectors*32)
        
        self.c4 = nn.Conv1d(
            in_channels=num_output_vectors*32,
            out_channels=num_output_vectors,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn4 = nn.BatchNorm1d(num_features=num_output_vectors)

        
    def forward(self, inputs):
        time_series = inputs
        batch_size = time_series.shape[0]
        
        x = self.c1(time_series)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.c3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.c4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = x.permute(0,2,1)
        return x
        

# convolutional neural network with 2D filter; note that this may not bring any additional information compared with 1D CNN, because the order of the rows (different kinematics time series) shouldn't carry useful information
class KineticsCNN2D(nn.Module):
    def __init__(self, num_input_vectors, num_output_vectors, kernel_width=1, name='KineticsCNN2D'):
        super().__init__()
        
        self.model_name = name
        
        # we use padding that ensures that the kernel moves along the width dimension keeping the width constant, but doesn't move along the height dimension
        padding_2d = (0, int((kernel_width-1)/2))
        
        self.c1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_output_vectors*8,
            kernel_size=(num_input_vectors, kernel_width),
            stride=1,
            padding=padding_2d,
            dilation=1,
            padding_mode='zeros'
        )
        self.bn1 = nn.BatchNorm1d(num_features=num_output_vectors*8)
        self.relu = nn.ReLU()
        
        
        self.c2 = nn.Conv1d(
            in_channels=num_output_vectors*8,
            out_channels=num_output_vectors*16,
            kernel_size=kernel_width,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn2 = nn.BatchNorm1d(num_features=num_output_vectors*16)
        
        self.c3 = nn.Conv1d(
            in_channels=num_output_vectors*16,
            out_channels=num_output_vectors*32,
            kernel_size=kernel_width,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn3 = nn.BatchNorm1d(num_features=num_output_vectors*32)
        
        self.c4 = nn.Conv1d(
            in_channels=num_output_vectors*32,
            out_channels=num_output_vectors,
            kernel_size=kernel_width,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn4 = nn.BatchNorm1d(num_features=num_output_vectors)
        
    def forward(self, inputs):
        time_series = inputs
        batch_size = time_series.shape[0]
        
        #print(f'SHAPE BEFORE: {time_series.shape}')
        
        
        # to treat the multiple-channel time series as image instead, we must add one dimension to denote the number of channels (1) in the image
        x = time_series.unsqueeze(1)
        
        #print(f'SHAPE UNSQUEEZED: {x.shape}')
        
        x = self.c1(x)
        
        #print(f'SHAPE AFTER: {x.shape}')
        
        # undoing the effects of the squeze ~20 lines above, we transform x back to a format that is 1D time series with multiple features, where each feature is its own channel
        x = x.squeeze(2)
        
        #print(f'SHAPE SQUEEZED: {x.shape}')
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.c3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.c4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        
        
        
        
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
        
        time_series = inputs
        
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
    def __init__(self, num_input_vectors, num_output_vectors, hidden_size=20, name='KineticsLSTM'):
        super().__init__()
        
        self.model_name = name
        
        self.lstm = nn.LSTM(
            input_size=num_input_vectors,
            hidden_size=hidden_size,
            proj_size=num_output_vectors,
            batch_first=True,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional
        )
        
        self.relu = nn.ReLU()
        
    def forward(self,inputs):
        
        time_series = inputs
        
        # transform to the correct format that is defined by making batch_first=True in the LSTM constructor
        time_series_trans = time_series.permute(0,2,1)
                
        # pass the transposed input to the LSTM
        lstm_out, temp = self.lstm(time_series_trans)
        # lstm_out now contains the short-term memory values from each unrolled LSTM unit
                
        prediction = lstm_out
        return prediction



# neural network that wraps around a time series predicting neural network and modifies its output to be more specific for subject demographics (body mass, body height, age, sex)
class DemographicScaler(nn.Module):
    def __init__(self, time_series_model, num_input_vectors, num_output_vectors, sequence_length, name=None):
        super().__init__()
        
        if name:
            self.model_name = name
        else:
            self.model_name = f'Demographic_{time_series_model.model_name}'
        self.num_output_vectors = num_output_vectors
        self.sequence_length = sequence_length
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4,16)
        self.fc2 = nn.Linear(16,64)
        self.fc3 = nn.Linear(64,num_output_vectors*self.sequence_length)

        # this refers to the model that eats the time series of kinematics (e.g., lower limb joint angles) and vomits out the time series of kinetics (e.g., knee contact forces)
        self.time_series_model = time_series_model
        
        # create parameters for the denoising filter based on the length of the sequence
        # the padding is calculated as a function of the filter width such that the dimensions of the output is the same as those of the input when convolving with the filter
        self.denoise_filter_width = round(self.sequence_length/15)
        if self.denoise_filter_width%2 == 0:
            self.denoise_filter_width += 1
        self.denoise_filter_padding = int((self.denoise_filter_width-1)/2)
        
    def forward(self, inputs):
        
        
        # MUOKKAA: KINEMATICS -> 1DCONV -> FULLY CONNECTED -> YHDISTYY SKALAARIEN FULLY CONNECTEDEIHIN JA LUO TÄTEN LOPULLISEN KERROINMASKIN
        
        scalars, time_series = inputs
        batch_size = scalars.shape[0]
        
        # predict the time series of kinetics from the time series of kinematics
        kinetics = self.time_series_model(time_series)
        
        # process scalar inputs into a "mask" that we can apply over the CNN-estimated curve
        x = self.fc1(scalars)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        
        x = x.reshape((batch_size,self.num_output_vectors,self.sequence_length))
        # mimic a low-pass filter by averaging in order to create the scaling mask
        x = F.avg_pool1d(x, kernel_size=self.denoise_filter_width, stride=1, padding=self.denoise_filter_padding, count_include_pad=False)

        # move the dimensions so that the order is (BATCH, SEQUENCE, FEATURES) where FEATURES can also be called CHANNELS (e.g., if the time series has 250 data points, those are in SEQUENCE, and if it has 4 loading features, those are in FEATURES)
        x = x.permute(0,2,1)        
        # scale each value in the kinetics time series by the scaling mask (x)
        x = kinetics * x
        
        
        return x



