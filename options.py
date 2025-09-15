# main/training settings
rng_seed=1
file_dataset='C:/Users/lavik/OneDrive/Documents/Polvineuro/Extracted/extracted_data_normalized.csv'
#file_dataset='C:/Users/lavik/OneDrive/Documents/Polvineuro/Extracted/extracted_data_padded.csv'
#file_dataset='C:/Users/lavik/OneDrive/Documents/Lencioni/processed_data.csv'
path_output = 'C:/Users/lavik/OneDrive/Documents/Polvineuro/PyTorch_output'

# training settings
batch_size=200
early_stopping_threshold=50
max_epochs=1000

workers=0

# network structure settings
lstm_hidden_size=128
lstm_num_layers=1
lstm_bidirectional=False

# learning rate settings
lr_initial=0.1

# visualization settings
plot_losses=False
plot_sample=False