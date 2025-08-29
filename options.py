# main/training settings
rng_seed=1
file_dataset='C:/Users/lavik/OneDrive/Documents/Lencioni/processed_data.csv'

# training settings
batch_size=200
training_threshold=1e-7
max_epochs=3000

# network structure settings
lstm_hidden_size=128
lstm_num_layers=1
lstm_bidirectional=False

# learning rate settings
lr_initial=0.1
lr_gamma=0.5
lr_step_size=100
