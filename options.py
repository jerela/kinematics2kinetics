# main/training settings
rng_seed=1
#file_dataset='C:/Users/lavik/OneDrive/Documents/Polvineuro/Extracted/extracted_data_normalized.csv'
#file_dataset='C:/Users/lavik/OneDrive/Documents/Polvineuro/Extracted/extracted_data_padded.csv'
file_dataset='C:/Users/lavik/OneDrive/Documents/Polvineuro/Extracted/extracted_data_healthy_padded.csv'
#file_dataset='C:/Users/lavik/OneDrive/Documents/Lencioni/processed_data.csv'
path_output = 'C:/Users/lavik/OneDrive/Documents/Polvineuro/PyTorch_output'

# training settings
batch_size=200
early_stopping_threshold=100
max_epochs=10000

workers=0

# network structure settings
lstm_num_layers=1
lstm_bidirectional=False

# learning rate settings
lr_initial=1e-2

# visualization settings
plot_losses=False
plot_sample=False

# test performance evaluation settings
path_test_data='C:/Users/lavik/OneDrive/Documents/Polvineuro/Extracted/test_data_padded.csv'
#path_trained_model='C:/Users/lavik/OneDrive/Documents/Polvineuro/PyTorch_output/Preliminary_testing/saves/checkpoint_Demographic_CNN_kernelsize9_fold3_epoch467_finished.pt'
path_trained_model='C:/Users/lavik/OneDrive/Documents/Polvineuro/PyTorch_output/Preliminary_testing/saves/checkpoint_Demographic_CNN_full_kernelsize_9_epoch546_finished.pt'