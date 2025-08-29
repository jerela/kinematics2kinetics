import torch
import torch.nn as nn
import torch.nn.functional as F

from options import lstm_hidden_size, lstm_num_layers, lstm_bidirectional

# a network that maps input kinematic time series to output kinetic time series
class KineticsLSTM(nn.Module):
    def __init__(self, num_input_vectors, num_output_vectors):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_input_vectors,
            hidden_size=lstm_hidden_size,
            proj_size=num_output_vectors,
            batch_first=True, num_layers=lstm_num_layers,
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

# a network that maps input demographic scalars and kinematic time series to output kinetic time series
class DemographicKineticsLSTM(nn.Module):
    """
    The idea here is that unlike KineticsLSTM, this network incorporates information about scalar variables (mass, height, age, sex).
    They could be important for measures like moments and contact forces, where simple normalization of outputs by those values will not do. For instance, height does not affect the moments directly, but through its correlation with limb lengths.
    
    We could apply the scalars directly to the output time series by training a scaling factor for each time point and output vector (e.g., 101*9 scaling factors). However, because scaling factors of adjacent time points vary, this would result in a noisy output time series.
    Instead, the idea here is to find an effect of the scalar variables that affects the time series "smoothly". For this purpose, we fit N-th order polynomials to make adjustments to the LSTM output. This should fix the noise-like effect that is present with using individual scaling factors for each time point.
    This is also interpretable, because if it works, we can input subject demographics and get as output the N-th order polynomial approximated curve that separates the person representing those demogrpahics from the general kinetics time series.
    
    If N-th order polynomial fitting does not work, an alternative is to try Fourier series.
    """
    def __init__(self, num_input_vectors, num_output_vectors):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_input_vectors,
            hidden_size=lstm_hidden_size,
            proj_size=num_output_vectors,
            batch_first=True,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional
        )
        #self.fc1 = nn.Linear(4,256)
        #self.fc2 = nn.Linear(256,101*num_output_vectors)
        

        
        self.polynomial_degree = 0
        self.num_output_vectors = num_output_vectors
        
        self.fc1 = nn.Linear(4,16)
        self.fc2 = nn.Linear(16,(self.polynomial_degree+1)*num_output_vectors)
        
    def forward(self,inputs):
        
        scalars, time_series = inputs
        
        batch_size = scalars.shape[0]
        
        # transform to the correct format that is defined by making batch_first=True in the LSTM constructor
        time_series_trans = time_series.permute(0,2,1)
        
        #x = self.fc1(scalars)
        #x = F.sigmoid(x)
        #x = self.fc2(x)
        #x = F.sigmoid(x)
        
        # DIMC_COEFFS = (BATCH, POLYDEGREE, NUM_VECTORS=9)
        # DIM_Y = (BATCH, TIMEPOINTS=101, NUM_VECTORS=9)
        
        x = self.fc1(scalars)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        coefficients = x.reshape(batch_size, self.polynomial_degree+1, self.num_output_vectors)
        #print(f'shape of coefficients: {coefficients.shape}')
        #print(f'shape of single degree coefficients: {coefficients[:,0,:].shape}')
        
        y = torch.zeros((batch_size, 101, self.num_output_vectors))
        
        #y += coefficients[:,0,:].reshape(batch_size, 1, self.num_output_vectors) * torch.ones((batch_size, 101, self.num_output_vectors))
        for i in range(self.polynomial_degree+1):
            #print(f'shape of y: {y.shape}')
            #print(f'shape of current coeffs: {self.coeffs[:,i].shape}')
            y += coefficients[:,i,:].reshape(batch_size, 1, self.num_output_vectors) * (torch.arange(1,102)**i).reshape(1,101,1).repeat(batch_size, 1, self.num_output_vectors)
            
        #print(f'shape of y: {y.shape}')
        
        # pass the transposed input to the LSTM
        lstm_out, temp = self.lstm(time_series_trans)
        # lstm_out now contains the short-term memory values from each unrolled LSTM unit
        
        #x = x.reshape(lstm_out.shape)
        #output_scaled = (x*lstm_out)
        #print(f'shape of lstm_out: {lstm_out.shape}')
        
        output_scaled = lstm_out+y
        
        prediction = output_scaled
        return prediction
