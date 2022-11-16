from functools import cmp_to_key

import cv2
import h5py
import numpy as np
import pandas as pd
import pickle
import re

from src.config import *


def read_keypoints(file_name):
    h5_file = h5py.File(file_name, 'r')
    scores = list(h5_file['instance_scores'][0])
    node_names = [str(x)[2:-1] for x in h5_file['node_names'][:]]
    point_scores = h5_file['point_scores'][0].transpose()
    x, y = h5_file['tracks'][0]
    x, y = x.transpose(), y.transpose()
    return scores, node_names, point_scores, x, y


def get_minimum_frame_count():
    minimum_frame_count = 1e10
    
    for video_file_name in os.listdir(CONVERTED_DATA_FOLDER):
        corner_keypoints_path = os.path.join(CORNER_TABLES_FOLDER, f'{video_file_name}.predictions.h5')
        corner_scores, _, _, _, _ = read_keypoints(corner_keypoints_path)
        
        if minimum_frame_count > len(corner_scores):
            minimum_frame_count = len(corner_scores)
    
    return minimum_frame_count


def calculate_units_per_cm(video_file_name):
    borders_keypoints_path = os.path.join(CORNER_TABLES_FOLDER, f'{video_file_name}.predictions.h5')
    border_tables = h5py.File(borders_keypoints_path, 'r')
    
    border_x, border_y = border_tables['tracks'][0]
    border_x, border_y = border_x.transpose(), border_y.transpose()
    
    border_x_mean, border_y_mean = np.nanmean(border_x, axis=0), np.nanmean(border_y, axis=0)
    
    left_upper_corner = (border_x_mean[0], border_y_mean[0])
    right_upper_corner = (border_x_mean[1], border_y_mean[1])
    
    units_per_cm = np.linalg.norm(np.array(right_upper_corner) - np.array(left_upper_corner)) / ARENA_LENGTH
    return units_per_cm


def get_video_config(video_file_name):
    units_per_cm = calculate_units_per_cm(video_file_name)
    cap = cv2.VideoCapture(os.path.join(CONVERTED_DATA_FOLDER, video_file_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bin_length_in_seconds = 60
    interval = int(bin_length_in_seconds * fps)
    duration_in_seconds = frame_count / fps
    
    config = {
        'bin_length': bin_length_in_seconds,
        'duration': duration_in_seconds,
        'fps': fps,
        'frame_count': frame_count,
        'interval': interval,
        'units_per_cm': units_per_cm
    }
    
    return config


def derive_df(video_file_name):
    pose_keypoints_path = os.path.join(POSE_TABLES_FOLDER, f'{video_file_name}.predictions.h5')
    pose_tables = h5py.File(pose_keypoints_path, 'r')
    pose_node_names = [str(x)[2:-1] for x in pose_tables['node_names'][:]]
    pose_x, pose_y = pose_tables['tracks'][0]
    pose_x, pose_y = pose_x.transpose(), pose_y.transpose()
    
    columns = []
    
    for node_name in pose_node_names:
        columns.append(f'{node_name}_x')
        columns.append(f'{node_name}_y')
    
    df = pd.DataFrame(pose_x, columns=[i + '_x' for i in pose_node_names]).join(
        pd.DataFrame(pose_y, columns=[i + '_y' for i in pose_node_names])
    )
    
    df = df[columns]
    return df


def add_auxiliary_features(df, config):
    df['Mean_Body_Position_1_x'] = df[['Nose_1_x', 'Upper_Body_1_x', 'Lower_Body_1_x', 'Back_1_x']].mean(axis=1)
    df['Mean_Body_Position_1_y'] = df[['Nose_1_y', 'Upper_Body_1_y', 'Lower_Body_1_y', 'Back_1_y']].mean(axis=1)
    
    df['Mean_Body_Position_2_x'] = df[['Nose_2_x', 'Body_2_x', 'Back_2_x']].mean(axis=1)
    df['Mean_Body_Position_2_y'] = df[['Nose_2_y', 'Body_2_y', 'Back_2_y']].mean(axis=1)
    
    df['Mean_Body_Position_Distance_12'] = np.linalg.norm(
        df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']].values - df[
            ['Mean_Body_Position_2_x', 'Mean_Body_Position_2_y']].values, axis=1)
    
    df['Close_Proximity'] = (df['Mean_Body_Position_Distance_12'] / config['units_per_cm']) <= PROXIMITY
    df['Close_Proximity'] = df['Close_Proximity'].astype(int)
    
    return df


def total_distance(pos1, pos2, units_per_cm):
    distances = np.linalg.norm(pos2.values - pos1.values, axis=1)
    distances = distances[~np.isnan(distances)]
    distance_in_cm = sum(distances) / units_per_cm
    return distance_in_cm


def total_time_of_interaction(col, duration, frame_count):
    total_time = duration * col.sum() / frame_count
    return total_time


def frequency_of_interaction(interaction_time, length):
    frequency = interaction_time / length
    return frequency


def latency_to_first_interaction(col, duration, frame_count):
    frames = 0
    
    for idx, val in col.items():
        if val == 0:
            frames += 1
        else:
            break
    
    latency = frames / frame_count * duration
    
    return latency


def time_stamped_interaction(col, duration, frame_count):
    time_series = []
    
    for k, v in col.items():
        if v != 0:
            val = v * k / frame_count * duration
            val = round(val, 2)
            time_series.append(val)
    
    return time_series


def derive_features(df, config):
    time_of_interaction = total_time_of_interaction(
        df['Close_Proximity'],
        config['duration'],
        config['frame_count']
    )
    first_interaction_latency = latency_to_first_interaction(
        df['Close_Proximity'],
        config['duration'],
        config['frame_count']
    )
    time_stamped_active_interaction = time_stamped_interaction(
        df['Close_Proximity'],
        config['duration'],
        config['frame_count']
    )
    interaction_frequency = frequency_of_interaction(
        time_of_interaction,
        config['duration']
    )
    distance_traveled = total_distance(
        df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']],
        df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']].shift(),
        config['units_per_cm']
    )
    results = {
        'distance_moved': distance_traveled,
        'first_interaction_latency': first_interaction_latency,
        'interaction_frequency': interaction_frequency,
        'time_of_active_interaction': time_of_interaction,
        'time_stamped_active_interaction': time_stamped_active_interaction
    }
    
    return results


def video_name_sort(a, b):
    regex = r'\d+'
    a_nums, b_nums = re.findall(regex, a['Video']), re.findall(regex, b['Video'])
    
    for i, j in zip(a_nums, b_nums):
        if int(i) > int(j):
            return 1
        if int(i) < int(j):
            return -1
    
    return 0


def read_dataset(interpolation=False):
    file_path = os.path.join(DATASET_FOLDER, 'new_dataset.pickle')
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)
    else:
        dataset = []
        min_frame_count = get_minimum_frame_count()
        
        for video_file_name in os.listdir(CONVERTED_DATA_FOLDER):
            config = get_video_config(video_file_name)
            df = derive_df(video_file_name)
            df = add_auxiliary_features(df, config)
            
            if interpolation:
                df = df.interpolate()
            
            df = df.iloc[:min_frame_count, :]
            
            features = derive_features(df, config)
            
            record = {
                'Video': video_file_name,
                'Config': config,
                'Data': df,
                'Features': features
            }
            
            dataset += [record]
        
        with open(file_path, 'wb') as file:
            pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    dataset = sorted(dataset, key=cmp_to_key(video_name_sort))
    
    first_experiment_dataset = []
    second_experiment_dataset = []
    
    for i in range(len(dataset)):
        if '_v3_' in dataset[i]['Video']:
            second_experiment_dataset += [dataset[i]]
        else:
            first_experiment_dataset += [dataset[i]]
    
    return first_experiment_dataset, second_experiment_dataset
