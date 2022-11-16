import cv2
import h5py
import numpy as np
import os
import pandas as pd

from src.config import *


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
	cap = cv2.VideoCapture(os.path.join(CONVERTED_DATA_FOLDER, video_file_name))
	fps = cap.get(cv2.CAP_PROP_FPS)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration_in_seconds = frame_count / fps
	bin_length_in_seconds = 60
	interval = int(bin_length_in_seconds * fps)
	units_per_cm = calculate_units_per_cm(video_file_name)
	
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
	
	if frames == len(col):
		return -1
	
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


def derive_features(df, config, is_bin=True, show=True):
	time_of_interaction = total_time_of_interaction(df['Close_Proximity'], config['duration'],
	                                                config['frame_count'])
	first_interaction_latency = latency_to_first_interaction(df['Close_Proximity'], config['duration'],
	                                                         config['frame_count'])
	time_stamped_active_interaction = time_stamped_interaction(df['Close_Proximity'], config['duration'],
	                                                           config['frame_count'])
	
	if is_bin:
		interaction_frequency = frequency_of_interaction(time_of_interaction, config['bin_length'])
	else:
		interaction_frequency = frequency_of_interaction(time_of_interaction, config['duration'])
	
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
	
	if show:
		if is_bin:
			bin_duration = config['bin_length']
		else:
			bin_duration = config['duration']
		
		print("Frequency of interaction (interaction/second): %4.2f" % interaction_frequency)
		print("Latency to first interaction              (s): %4.2f" % first_interaction_latency)
		print("Total time of active interaction          (s): %4.2f" % time_of_interaction)
		print("Bin duration time                         (s): %4.2f" % bin_duration)
		print("Total distance moved by the subject      (cm): %4.2f\n" % distance_traveled)
		# print("Time stamped active interaction              : %s" % time_stamped_active_interaction)
		
	return results


def print_derived_features(df, config, show_small_bin=True):
	print('\n')
	
	for i in range(0, len(df), config['interval']):
		temp_df = df[i:i + config['interval']]
		
		if len(temp_df) < config['interval'] and not show_small_bin:
			continue
			
		derive_features(temp_df, config, is_bin=True, show=True)
		print('*' * 80)
	
	print("TOTAL VALUES FOR THE WHOLE VIDEO:\n")
	derive_features(df, config, is_bin=False, show=True)


def main(video_file_name):
	config = get_video_config(video_file_name)
	df = derive_df(video_file_name)
	df = add_auxiliary_features(df, config)
	print_derived_features(df, config, False)


if __name__ == "__main__":
	main(video_file_name='Trial_1_v1_SI_w6_control_4037.mp4')
