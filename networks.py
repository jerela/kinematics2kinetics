import torch
import torch.nn as nn

class KineticsLSTM(nn.Module):
    def __init__(self, num_input_vectors, num_output_vectors):
        super().__init__()
        
        hidden_size = 128
        self.lstm = nn.LSTM(input_size=num_input_vectors, hidden_size=128, proj_size=num_output_vectors, batch_first=True)
        #self.fc = nn.Linear(hidden_size,101)
        
    def forward(self,inputs):
        
        scalars, time_series = inputs
        
        # transform to the correct format that is defined by making batch_first=True in the LSTM constructor
        time_series_trans = time_series.permute(0,2,1)
        
        # pass the transposed input to the LSTM
        lstm_out, temp = self.lstm(time_series_trans)
        # lstm_out now contains the short-term memory values from each unrolled LSTM unit
        
        prediction = lstm_out
        return prediction
