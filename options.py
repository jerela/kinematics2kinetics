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

# settings for min-max scaling data and undoing the scaling
kinematics_bounds = { # instead of using joint ranges of motion from the musculoskeletal model, let's assume plausible ranges during the stance phase
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
scalar_bounds = {
    'body_mass': (40.0, 150.0),
    'body_height': (1.4, 2.2),
    'age': (18.0, 80.0)
}
kinetics_bounds = {
    'kcf_summed': (0.0, 6000.0),
    'kcf_medial': (0.0, 4000.0),
    'kcf_lateral': (0.0,3000.0),
    'kcf_patellofemoral': (0.0,4000.0)
}
