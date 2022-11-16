from functools import cmp_to_key
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import silhouette_score, TimeSeriesKMeans

import h5py
import numpy as np
import pandas as pd
import pickle
import re

from src.config import *


def video_name_sort(a, b):
    regex = r'\d+'
    a_nums, b_nums = re.findall(regex, a['Video']), re.findall(regex, b['Video'])
    
    for i, j in zip(a_nums, b_nums):
        if int(i) > int(j):
            return 1
        if int(i) < int(j):
            return -1
    
    return 0


def read_keypoints(file_name):
    h5_file = h5py.File(file_name, 'r')
    scores = list(h5_file['instance_scores'][0])
    node_names = [str(x)[2:-1] for x in h5_file['node_names'][:]]
    point_scores = h5_file['point_scores'][0].transpose()
    x, y = h5_file['tracks'][0]
    x, y = x.transpose(), y.transpose()
    return scores, node_names, point_scores, x, y


def create_dataframe_from_keypoints(node_names, point_scores, x, y):
    columns = []

    for node_name in node_names:
        columns.append(f'{node_name}_x')
        columns.append(f'{node_name}_y')
        columns.append(f'{node_name}_score')

    df = pd.DataFrame(point_scores, columns=[x + '_score' for x in node_names]).join(
        pd.DataFrame(x, columns=[i + '_x' for i in node_names])
    ).join(
        pd.DataFrame(y, columns=[i + '_y' for i in node_names])
    )

    df = df[columns]
    
    return df


def preprocess_corner_keypoints(x, y):
    corner_x_mean, corner_y_mean = np.nanmean(x, axis=0), np.nanmean(y, axis=0)
    corners = []

    left_upper_corner = (corner_x_mean[0], corner_y_mean[0])
    right_upper_corner = (corner_x_mean[1], corner_y_mean[1])
    right_lower_corner = (corner_x_mean[2], corner_y_mean[2])
    left_lower_corner = (corner_x_mean[3], corner_y_mean[3])
    
    return left_upper_corner, right_upper_corner, right_lower_corner, left_lower_corner


def add_mean_body_position(df):
    df['Mean_Body_Position_1_x'] = df[['Nose_1_x', 'Upper_Body_1_x', 'Lower_Body_1_x', 'Back_1_x']].mean(axis=1)
    df['Mean_Body_Position_1_y'] = df[['Nose_1_y', 'Upper_Body_1_y', 'Lower_Body_1_y', 'Back_1_y']].mean(axis=1)
    
    df['Mean_Body_Position_2_x'] = df[['Nose_2_x', 'Body_2_x', 'Back_2_x']].mean(axis=1)
    df['Mean_Body_Position_2_y'] = df[['Nose_2_y', 'Body_2_y', 'Back_2_y']].mean(axis=1)
    
    return df


def add_corner_distances(df, corners):
    corner_mapping = {
        'Left_Upper_Corner_Distance_1': 0,
        'Right_Upper_Corner_Distance_1': 1,
        'Right_Lower_Corner_Distance_1': 2,
        'Left_Lower_Corner_Distance_1': 3
    }
    
    for feature_name, corner_idx in corner_mapping.items():
        df[feature_name] = np.linalg.norm(
            df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']].values - corners[corner_idx],
            axis=1
        )
    
    return df


def add_border_distances(df, corners):
    borders_mapping = {
        'Upper_Border_Distance_1': (1, 0),
        'Left_Border_Distance_1': (3, 0),
        'Right_Border_Distance_1': (1, 2),
        'Lower_Border_Distance_1': (3, 2)
    }
    
    for feature_name, corner_indices in borders_mapping.items():
        idx1, idx2 = corner_indices
        
        df[feature_name] = np.linalg.norm(
            np.expand_dims(
                np.cross(
                    np.array(corners[idx1]) - np.array(corners[idx2]),
                    corners[idx2] - df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']].values
                ), axis=1
            ), axis=1
        ) / np.linalg.norm(np.array(corners[idx1]) - np.array(corners[idx2]))
    
    return df


def add_distance(df, feature1, feature2, single=True):
    new_feature_name = f'{feature1}_To_{feature2}_Distance_1'
    idx = '1'
    
    if not single:
        idx = '2'
        new_feature_name += idx
        
    first_values = df[[f'{feature1}_1_x', f'{feature1}_1_y']].values
    second_values = df[[f'{feature2}_{idx}_x', f'{feature2}_{idx}_y']].values
    
    df[new_feature_name] = np.linalg.norm(first_values - second_values, axis=1)
    
    return df


def add_body_parts_distances(df):
    df = add_distance(df, "Nose", "Tail_End", single=True)
    df = add_distance(df, "Nose", "Left_Hand", single=True)
    df = add_distance(df, "Nose", "Right_Hand", single=True)
    df = add_distance(df, "Nose", "Nose", single=False)
    df = add_distance(df, "Nose", "Body", single=False)
    df = add_distance(df, "Nose", "Back", single=False)
    df = add_distance(df, "Nose", "Tail_End", single=False)
    df = add_distance(df, "Lower_Body", "Body", single=False)
    df = add_distance(df, "Mean_Body_Position", "Mean_Body_Position", single=False)
    
    return df


def calculate_total_distance(df):
    pos1 = df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']]
    pos2 = df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']].shift()
    
    distances = np.linalg.norm(pos2.values - pos1.values, axis=1)
    distances = distances[~np.isnan(distances)]
    
    total_distance = sum(distances)
    return total_distance


def preprocess_pose_keypoints(node_names, point_scores, x, y, corners):
    df = create_dataframe_from_keypoints(node_names, point_scores, x, y)
    df = add_mean_body_position(df)
    df = add_corner_distances(df, corners)
    df = add_border_distances(df, corners)
    df = add_body_parts_distances(df)
    
    return df


def get_minimum_size():
    minimum_size = 1e9
    
    for video_file_name in os.listdir(CONVERTED_DATA_FOLDER):
        corner_keypoints_path = os.path.join(CORNER_TABLES_FOLDER, f'{video_file_name}.predictions.h5')
        corner_scores, corner_node_names, corner_point_scores, corner_x, corner_y = read_keypoints(
            corner_keypoints_path)
        
        if minimum_size > len(corner_scores):
            minimum_size = len(corner_scores)
        
    return minimum_size


def read_dataset(window=None, normalization=False, interpolation=False, show=False):
    print("[INFO] Reading the dataset...")
    
    file_path = os.path.join(DATASET_FOLDER, 'dataset.pickle')
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)
    else:
        dataset = []
        min_size = get_minimum_size()
        
        for video_file_name in os.listdir(CONVERTED_DATA_FOLDER):
            corner_keypoints_path = os.path.join(CORNER_TABLES_FOLDER, f'{video_file_name}.predictions.h5')
            pose_keypoints_path = os.path.join(POSE_TABLES_FOLDER, f'{video_file_name}.predictions.h5')
    
            corner_scores, corner_node_names, corner_point_scores, corner_x, corner_y = read_keypoints(
                corner_keypoints_path)
            pose_scores, pose_node_names, pose_point_scores, pose_x, pose_y = read_keypoints(pose_keypoints_path)
    
            corners = preprocess_corner_keypoints(corner_x, corner_y)
            data = preprocess_pose_keypoints(pose_node_names, pose_point_scores, pose_x, pose_y, corners)
            
            distance = calculate_total_distance(data)
            
            if show:
                print('-' * 80)
                print(f"Number of frames: {data.shape[0]}")
                print(f"Number of features: {data.shape[1]}\n")
                
                print(f"Left upper corner coordinates: {corners[0]}")
                print(f"Right upper corner coordinates: {corners[1]}")
                print(f"Right lower corner coordinates: {corners[2]}")
                print(f"Left lower corner coordinates: {corners[3]}")
                
            if interpolation:
                data = data.interpolate()
    
            data = data.iloc[:min_size, :]
            
            if window:
                data = data.rolling(window).mean().iloc[::window, :]
                
            if normalization:
                scaler = MinMaxScaler()
                data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
                
            for col_name in MAIN_FEATURES:
                if not normalization:
                    scaler = MinMaxScaler()
                    temp_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
                    temp_data.index = data.index
                else:
                    temp_data = data.copy()
                
                new_col = '_'.join(col_name.split('_')[:-2]) + '_Is_Close'
                data[new_col] = temp_data[col_name] <= CLOSENESS_THRESHOLDS[new_col]
                data[new_col] = data[new_col].astype(float)
            
            record = {
                'Video': video_file_name,
                'Data': data,
                'Distance': distance,
                'Corners': corners
            }
    
            dataset.append(record)
            
        with open(file_path, 'wb') as file:
            pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("The dataset is read!")

    first_exp_dataset = []
    second_exp_dataset = []
    video_idx_to_name = {}

    dataset = sorted(dataset, key=cmp_to_key(video_name_sort))

    for i in range(len(dataset)):
        if '_v3_' in dataset[i]['Video']:
            second_exp_dataset.append(dataset[i])
        else:
            first_exp_dataset.append(dataset[i])
    
        video_idx_to_name[i] = dataset[i]['Video']

    return first_exp_dataset, second_exp_dataset, video_idx_to_name


def find_best_number_of_clusters(X, start=2, end=20, metric='dtw'):
    numbers_of_clusters = {}

    print(f"Metric - {metric}")
    
    for k in range(start, end):
        clusterer = TimeSeriesKMeans(n_clusters=k, metric=metric, n_jobs=-1, random_state=0)
        clusterer.fit(X)

        score = silhouette_score(X, clusterer.labels_, metric=metric)

        print(f'Number of clusters - {k}, Silhouette score - {score}')
        numbers_of_clusters[k] = score
        
    best_num_clusters = max(numbers_of_clusters, key=numbers_of_clusters.get)
    
    print(f"Best Silhouette score - {max(numbers_of_clusters.values())}, Best number of clusters - {best_num_clusters}")


def main():
    first_dataset, second_dataset, video_idx_to_name = read_dataset(window=20, normalization=False, interpolation=True)

    for main_feature in MAIN_FEATURES:
        X = []
        videos = {}

        for i in range(len(first_dataset)):
            X.append(first_dataset[i]['Data'][main_feature][1:])
            videos[i] = first_dataset[i]['Video']

        X = pd.concat(X, axis=1).transpose().fillna(method='bfill')
        X = X.values[:, :, np.newaxis]

        print(f"\nFeature - {main_feature}")

        find_best_number_of_clusters(X, 2, 20, metric='euclidean')

        # clusterer = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', n_jobs=-1)
        # clusterer.fit(X)
        #
        # dtw_score = silhouette_score(X, clusterer.labels_, metric='dtw')

        # print('*' * 70)
        # print(f"\n[FEATURE]            ---> {main_feature}")
        # print(f"[NUMBER OF CLUSTERS] ---> {n_clusters}")
        # print(f"[SILHOUETTE SCORE]   ---> {dtw_score}")
        #
        # unique_labels = np.unique(clusterer.labels_)
        # video_mapping = {label: [] for label in unique_labels}
        #
        # for i in range(len(videos)):
        #     video_mapping[clusterer.labels_[i]].append(i)
        #
        # for label, videos_list in video_mapping.items():
        #     print(f"\nCluster #{label}:")
        #
        #     for i in videos_list:
        #         print(f"- {videos[i]}")


if __name__ == "__main__":
    main()
