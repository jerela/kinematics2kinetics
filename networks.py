import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
METHODS/ARCHITECTURES TO EXPLORE:
- LSTM-FCN PyTorch implementation: https://github.com/flaviagiammarino/lstm-fcn-pytorch/blob/main/lstm_fcn_pytorch/modules.py
- MLSTM-FCN PyTorch implementation: https://github.com/alexmelekhin/MLSTM-FCN-Pytorch/blob/main/src/model.py

"""
# mean square error function that puts less emphasis on values at the beginning and the end of the time series, to account for the high simulation nose in the beginning and end of contact force time series
class WeightedMSELoss(nn.Module):
    """
    A custom loss function that computes the mean square error between the input and the target. Customized to disregard zeros in the target data using a weights mask.
    Note that the mean is calculated for all elements regardless of shape. Therefore, even if there are several kinetics features or several data samples in the batch, one scalar is returned for loss.
    Note also that the number of trailing zeros (result of zero-padding) will make this error smaller than the "real" error is.
    This loss should be used during training to compare different hyperparameter configurations, when the absolute value of the loss (and its physical interpretation) is irrelevant and only the relative losses matter.
    """
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
    def __init__(self, num_input_vectors, num_output_vectors, hidden_size=20, num_layers=1, bidirectional=True, name='KineticsLSTM'):
        super().__init__()
        
        self.bidirectional = bidirectional
        
        self.model_name = name
        
        self.lstm = nn.LSTM(
            input_size=num_input_vectors,
            hidden_size=hidden_size,
            proj_size=num_output_vectors,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=bidirectional
        )
        
        self.lstm_scale = nn.Parameter(torch.randn(1))
        
    def forward(self,inputs):
        
        time_series = inputs
        
        # transform to the correct format that is defined by making batch_first=True in the LSTM constructor
        time_series_trans = time_series.permute(0,2,1)
                
        # pass the transposed input to the LSTM
        lstm_out, temp = self.lstm(time_series_trans)
        # lstm_out now contains the short-term memory values from each unrolled LSTM unit
        
        # if we use a bidirectional LSTM, the output will be a concatenation of the forward and hidden states so we want to sum them together to maintain our data shape
        if self.bidirectional:
            prediction = (lstm_out[:,:,0]+lstm_out[:,:,1]).unsqueeze(-1)
        else:
            prediction = lstm_out
        prediction = self.lstm_scale*prediction
        return prediction

# a network where convolutional blocks are first used to find features that are then fed to LSTM
class KineticsCNNLSTM(nn.Module):
    def __init__(self, num_input_vectors, num_output_vectors, kernel_size=1, hidden_size=20, lstm_num_layers=1, lstm_bidirectional=True, name='KineticsCNNLSTM'):
        super().__init__()
        
        self.model_name = name
        
        self.relu = nn.ReLU()
        
        self.lstm_bidirectional = lstm_bidirectional
        
        self.c1 = nn.Conv1d(
            in_channels=num_input_vectors,
            out_channels=num_output_vectors*4,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn1 = nn.BatchNorm1d(num_features=num_output_vectors*4)
        
        self.c2 = nn.Conv1d(
            in_channels=num_output_vectors*4,
            out_channels=num_output_vectors*8,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn2 = nn.BatchNorm1d(num_features=num_output_vectors*8)
        
        self.c3 = nn.Conv1d(
            in_channels=num_output_vectors*8,
            out_channels=num_output_vectors,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn3 = nn.BatchNorm1d(num_features=num_output_vectors)
        
        # num_input_vectors and num_output_vectors parameters are both given the value of num_output_vectors because the CNN has already transformed the input to the shape of the output
        self.lstm = KineticsLSTM(num_input_vectors=num_output_vectors, num_output_vectors=num_output_vectors, hidden_size=hidden_size, num_layers=lstm_num_layers, bidirectional=lstm_bidirectional)
        
    def forward(self,inputs):
        
        time_series = inputs
        
        # first we put the input through the convolutional part
        x = self.c1(time_series)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.c3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # then through an LSTM
        prediction = self.lstm(x)
        
        return prediction

# a squeeze-and-excitation block for the MLSTM-FCN model
class SqueezeAndExcitation(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(num_channels, 16, kernel_size=1)
        self.fc2 = nn.Conv1d(16, num_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x*inputs

# a multivariate LSTM-FCN constructed after https://arxiv.org/abs/1801.04503
class KineticsMLSTMFCN(nn.Module):
    def __init__(self, num_input_vectors, num_output_vectors, hidden_size, lstm_num_layers=1, lstm_bidirectional=True, name='KineticsMLSTMFCN'):
        super().__init__()
        
        self.model_name = name
        
        num_filters_c1 = num_input_vectors*8
        num_filters_c2 = num_input_vectors*16
        num_filters_c3 = num_input_vectors*8
        
        self.lstm = KineticsLSTM(num_input_vectors=num_input_vectors, num_output_vectors=num_output_vectors, hidden_size=hidden_size, num_layers=lstm_num_layers, bidirectional=lstm_bidirectional)
        
        self.c1 = nn.Conv1d(
            in_channels=num_input_vectors,
            out_channels=num_filters_c1,
            kernel_size=7,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.c2 = nn.Conv1d(
            in_channels=num_filters_c1,
            out_channels=num_filters_c2,
            kernel_size=5,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.c3 = nn.Conv1d(
            in_channels=num_filters_c2,
            out_channels=num_filters_c3,
            kernel_size=3,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )

        self.bn1 = nn.BatchNorm1d(num_filters_c1)
        self.bn2 = nn.BatchNorm1d(num_filters_c2)
        self.bn3 = nn.BatchNorm1d(num_filters_c3)

        self.se1 = SqueezeAndExcitation(num_filters_c1)
        self.se2 = SqueezeAndExcitation(num_filters_c2)

        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        x = inputs
        x1 = self.lstm(x)
        
        x2 = self.c1(x)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.se1(x2)
        
        x2 = self.c2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.se2(x2)
        
        x2 = self.c3(x2)
        x2 = self.bn3(x2)
        x2 = self.relu(x2)
        
        # global pooling by calculating the mean over all channels to get just one channel, so that it can be concatenated with the output of the LSTM
        x2 = torch.mean(x2,1)
        x2 = x2.unsqueeze(2)
        
        prediction = x1+x2
        return prediction



class PositionalEncoding(nn.Module):
    def __init__(self, num_input_vectors, sequence_length, dropout, batch_first=True):
        super().__init__()
        
        self.batch_first = batch_first
        
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_input_vectors, 2).float() * (-math.log(10000.0) / num_input_vectors))
        
        pe = torch.zeros(sequence_length, num_input_vectors)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.unsqueeze(0)
        
        self.dropout = nn.Dropout(p = dropout)
        
        if not batch_first:
            pe.transpose(1,0,2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x


class KineticsTransformer(nn.Module):
    def __init__(self, num_input_vectors, num_output_vectors, sequence_length, p_dropout, name='KineticsTransformer'):
        super().__init__()
        
        self.model_name = name
        
        n_features = (num_input_vectors//2)*2
        
        # positional encoder can't handle an uneven number of features so we have to make them even with conv
        self.c1 = nn.Conv1d(
            in_channels=num_input_vectors,
            out_channels=n_features,
            kernel_size=1,
            stride=1,
            padding='same',
            dilation=1,
            padding_mode='zeros'
        )
        self.bn1 = nn.BatchNorm1d(num_features=n_features)
        
        self.pe = PositionalEncoding(
            num_input_vectors = n_features,
            sequence_length = sequence_length,
            dropout = p_dropout,
            batch_first = True
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = n_features,
            nhead = 6,
            dim_feedforward = 2048,
            dropout = p_dropout,
            batch_first = True
        )
        
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=6
            )
        
        self.decoder = nn.Linear(
            in_features = n_features,
            out_features = num_output_vectors
        )
                
    def forward(self,inputs):        
        x = inputs
        x = self.c1(x)
        x = self.bn1(x)
        x = x.permute(0,2,1)
        x = self.pe(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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
        
        self.dropout = nn.Dropout(p=0.5)
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
        
        scalars, time_series = inputs
        batch_size = scalars.shape[0]
        
        # predict the time series of kinetics from the time series of kinematics
        kinetics = self.time_series_model(time_series)
        
        # process scalar inputs into a "mask" that we can apply over the estimated curve
        x = self.fc1(scalars)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
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



# The demographic scaling classes/functions below are experimental.

class GaussianFunction(nn.Module):
    def __init__(self, num_gaussians):
        super().__init__()
        self.num_gaussians = num_gaussians
    
    def forward(self, inputs, max_lengths):
        batch_size = max_lengths.shape[0]
        # create a range with 250 elements, from 0 to 249
        x = torch.arange(start=0, end=250, step=1).unsqueeze(0)
        # repeat the range for all samples in the batch
        x = x.repeat(batch_size,1)
        # reshape the ranges to (BATCH, SEQUENCE, 1)
        x = x.reshape(batch_size,250,1)
        # repeat dim 2 for the number of gaussian functions
        x = x.repeat(1,1,self.num_gaussians)
        
        # outputs of the hyperbolic tan are in the range [-1, 1], so we modify that by 2 to allow the mask to increase the magnitude of the kinetics curve
        # the center of the gaussian is scaled to [0,1] times information length
        # the width of the gaussian is scaled to [0,1] times information length
        a = (inputs[:,0:self.num_gaussians]*2).unsqueeze(1)
        b = ((inputs[:,self.num_gaussians:2*self.num_gaussians]+1)/2).unsqueeze(1)*(max_lengths.reshape(batch_size,1,1).repeat(1,1,self.num_gaussians))
        c = ((inputs[:,2*self.num_gaussians:3*self.num_gaussians]+1)/2).unsqueeze(1)*(max_lengths.reshape(batch_size,1,1).repeat(1,1,self.num_gaussians))
        
        out = a*torch.exp(-torch.square(x-b)/torch.square(c))
        # out shape: (batch, sequence, i_gaussian) -> dim 2 must be removed by summing
        return out.sum(dim=2)

# returns "information length", i.e., the number of data points in the time series that are not zero-padded
class InformationLengthMeasurer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,tensors):
        batch_size = tensors.shape[0]
        time_series = tensors[:,0,:]
        rightmost = torch.zeros(batch_size, dtype=torch.uint8)
        for b in range(batch_size):
            nonzeros = torch.abs(time_series[b,:]) > 1e-9
            idx_info = torch.nonzero(nonzeros)
            rightmost[b] = idx_info[-1]
        return rightmost

# Similar to DemographicScaler in the sense that is uses demographic info to apply a mask over the predicted kinetics; however, the mask is constructed from summed gaussian functions whose parameters are computed from demographic information
# training is quite unstable though, so this should be considered experimental only
class DemographicGaussian(nn.Module):
    def __init__(self, time_series_model, num_input_vectors, num_output_vectors, sequence_length, name=None):
        super().__init__()        
        
        if name:
            self.model_name = name
        else:
            self.model_name = f'DemographicGaussian_{time_series_model.model_name}'
        self.num_output_vectors = num_output_vectors
        self.sequence_length = sequence_length
        
        # number of gaussian functions to sum and include in the mask; the more gaussians, the more likely the training is unable to converge
        self.num_gaussians = 2
        
        # we include a hyperbolic tan activation to ensure the output is constrained in [-1,1]
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(4,16)
        self.fc2 = nn.Linear(16,64)
        self.fc3 = nn.Linear(64,num_output_vectors*self.num_gaussians*3)

        # a custom module that computes the values of the Gaussian function (points on a normal distribution)
        self.gaussian = GaussianFunction(self.num_gaussians)
        # gets the "information length" of a time series, i.e., the length of the time series without trailing padded zeros; used to scale the width and center of the Gaussians
        self.infolength = InformationLengthMeasurer()

        # this refers to the model that eats the time series of kinematics (e.g., lower limb joint angles) and vomits out the time series of kinetics (e.g., knee contact forces)
        self.time_series_model = time_series_model
        
    def forward(self, inputs):
        
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
        x = self.tanh(x)
        
        #print(f'x: {x}')
        
        # initialize the gaussian mask as a bunch of zeros
        gaussian_mask = torch.zeros(kinetics.size())
        # get the information length of the input time series (assuming it still has trailing zeros from zero-padding)
        information_length = self.infolength(time_series)
        
        for f in range(self.num_output_vectors):
            gaussian_mask[:,:,f] += self.gaussian(x,information_length)
            
        # scale each value in the kinetics time series by the gaussian mask
        x = kinetics * gaussian_mask
        
        return x