# which knee contact force to use as the target: 'summed', 'medial', 'lateral' or 'patellofemoral'
kinetics_variable = 'summed'

# list of kinematics (generalized coordinate) variables to include in the input to the network
# "full" set of kinematics
included_generalized_coordinates = ['lumbar_extension', 'lumbar_rotation', 'lumbar_bending', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'hip_flexion_primary', 'hip_adduction_primary', 'hip_rotation_primary', 'knee_angle_primary', 'ankle_angle_primary', 'hip_flexion_secondary', 'hip_adduction_secondary', 'hip_rotation_secondary', 'knee_angle_secondary', 'ankle_angle_secondary']

# "lowerbody" set
#included_generalized_coordinates = ['hip_flexion_primary', 'hip_adduction_primary', 'hip_rotation_primary', 'knee_angle_primary', 'ankle_angle_primary', 'hip_flexion_secondary', 'hip_adduction_secondary', 'hip_rotation_secondary', 'knee_angle_secondary', 'ankle_angle_secondary']

# "stanceleg" set
#included_generalized_coordinates = ['hip_flexion_primary', 'hip_adduction_primary', 'hip_rotation_primary', 'knee_angle_primary', 'ankle_angle_primary']

# "sagittal" set
#included_generalized_coordinates = ['hip_flexion_primary', 'knee_angle_primary', 'ankle_angle_primary']

# "knee" set
#included_generalized_coordinates = ['knee_angle_primary']

# main/training settings
#rng_seed=1
rng_seed=5
file_dataset = 'DRIVE:/PATH/TO/TRAINING/DATA/extracted_data_healthy_padded.csv'
path_output = 'DRIVE:/PATH/TO/OUTPUT/DIRECTORY/PyTorch_output'

# training settings
batch_size=256
early_stopping_threshold=100
# 10 000 to be sure, although early stopping will halt training much earlier
max_epochs=10000

workers=0

accumulate_gradients = False

# learning rate settings
lr_initial=1e-2

# visualization settings
plot_losses=False
plot_sample=False

# path to the test data containing input kinematics and output kinetics of the test set; markerless uses kinematics from the markerless method, while reference uses kinematics from motion capture marker-based IK through OpenSim
# this is only used in main_test.py to evaluate the previously-trained network
#path_test_data='DRIVE:/PATH/TO/TEST/DATA/reference_test_data_padded.csv'
path_test_data='DRIVE:/PATH/TO/TEST/DATA/markerless_test_data_padded.csv'

# path to the trained model to use for making predictions in main_test.py
if kinetics_variable == 'summed':
    path_trained_model = 'DRIVE:/PATH/TO/MODELS/totKCF_full.pt'
elif kinetics_variable == 'medial':
    path_trained_model = 'DRIVE:/PATH/TO/MODELS/medKCF_full.pt'
elif kinetics_variable == 'lateral':
    path_trained_model = 'DRIVE:/PATH/TO/MODELS/latKCF_full.pt'
elif kinetics_variable == 'patellofemoral':
    path_trained_model = 'DRIVE:/PATH/TO/MODELS/patKCF_full.pt'
else:
    raise ValueError('Kinetics variable should be summed, medial, lateral, or patellofemoral, but none of those was identified in options.py')


path_output_predicted_time_series='DRIVE:/PATH/TO/OUTPUT/DIRECTORY/Predictions/'

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
