import os

DATA_FOLDER = '/home/faranio/Desktop/EDISS/Courses/1st Year. Period 1/Data Intensive Engineering ' \
              'I/Brain-Neuroactivity/legacy_data/'

RAW_DATA_FOLDER       = os.path.join(DATA_FOLDER, '01_raw')
CONVERTED_DATA_FOLDER = os.path.join(DATA_FOLDER, '02_converted')
LABELS_DATA_FOLDER    = os.path.join(DATA_FOLDER, '03_labels')
MODELS_DATA_FOLDER    = os.path.join(DATA_FOLDER, '05_models')
POSE_DATA_FOLDER      = os.path.join(DATA_FOLDER, '06_pose_labels')
POSE_TABLES_FOLDER    = os.path.join(DATA_FOLDER, '07_pose_tables')
GRAPHS_FOLDER         = os.path.join(DATA_FOLDER, '08_graphs')
CORNER_DATA_FOLDER    = os.path.join(DATA_FOLDER, '09_borders_labels')
CORNER_TABLES_FOLDER  = os.path.join(DATA_FOLDER, '10_borders_tables')
DATASET_FOLDER        = os.path.join(DATA_FOLDER, '11_dataset')

CLOSENESS_THRESHOLDS = {
	'Lower_Body_To_Body_Is_Close': 0.1,
	'Mean_Body_Position_Is_Close': 0.1,
	'Nose_To_Nose_Is_Close'      : 0.05,
	'Nose_To_Body_Is_Close'      : 0.1,
	'Nose_To_Back_Is_Close'      : 0.05,
	'Nose_To_Tail_End_Is_Close'  : 0.05
}

KEYPOINTS = [
	'Nose_1',
	'Upper_Body_1',
	'Lower_Body_1',
	'Back_1',
	'Tail_Middle_1',
	'Tail_End_1',
	'Left_Hand_1',
	'Right_Hand_1',
	'Left_Leg_1',
	'Right_Leg_1',
	'Nose_2',
	'Body_2',
	'Back_2',
	'Tail_Middle_2',
	'Tail_End_2'
]

MAIN_FEATURES = [
	'Lower_Body_To_Body_Distance_12',
	'Mean_Body_Position_Distance_12',
	'Nose_To_Nose_Distance_12',
	'Nose_To_Body_Distance_12',
	'Nose_To_Back_Distance_12',
	'Nose_To_Tail_End_Distance_12'
]

MAIN_FEATURES_AND_CLOSENESS = [
	'Lower_Body_To_Body_Distance_12',
	'Mean_Body_Position_Distance_12',
	'Nose_To_Nose_Distance_12',
	'Nose_To_Body_Distance_12',
	'Nose_To_Back_Distance_12',
	'Nose_To_Tail_End_Distance_12',
	'Lower_Body_To_Body_Is_Close',
	'Mean_Body_Position_Is_Close',
	'Nose_To_Nose_Is_Close',
	'Nose_To_Body_Is_Close',
	'Nose_To_Back_Is_Close',
	'Nose_To_Tail_End_Is_Close'
]

FEATURES_AND_CLUSTER_NUMS = {
    'Lower_Body_To_Body_Distance_12': 2,
    'Mean_Body_Position_Distance_12': 3,
    'Nose_To_Nose_Distance_12'      : 2,
    'Nose_To_Body_Distance_12'      : 4,
    'Nose_To_Back_Distance_12'      : 2,
    'Nose_To_Tail_End_Distance_12'  : 2,
    'Lower_Body_To_Body_Is_Close'   : 2,
    'Mean_Body_Position_Is_Close'   : 2,
    'Nose_To_Nose_Is_Close'         : 2,
    'Nose_To_Body_Is_Close'         : 2,
    'Nose_To_Back_Is_Close'         : 2,
    'Nose_To_Tail_End_Is_Close'     : 2
}
