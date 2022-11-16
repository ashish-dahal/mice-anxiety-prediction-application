from shapely import speedups
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

speedups.disable()

from itertools import permutations
from pathlib import Path
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.config import data_path
from src.pipeline.utils import create_barplot, get_video_config, plot_timeline, read_pose_keypoints, save_features


class YMazeExperimentFeatureEngineering:
    """
    A class that derives time-series and aggregate features for a video recording with a Y-Maze Experiment Type.
    """
    def __init__(self, pose_keypoints_path: str, box_keypoints_path: str, thresholds_config: dict, video_config: dict,
                 save_path: str):
        """
        :param pose_keypoints_path:
            Path to resultant pose keypoints of video recordings.
        :param box_keypoints_path:
            Path to resultant box keypoints of video recordings.
        :param video_config:
            Video recordings information such as units per cm, FPS, duration, etc.
        """
        self.pose_keypoints_path = pose_keypoints_path
        self.box_keypoints_path = box_keypoints_path
        self.thresholds_config = thresholds_config
        self.video_config = video_config
        self.save_path = save_path
        self.experiment_type = "YM"

    def run_feature_engineering(self) -> tuple:
        """
        Derives features for a single video recording.
        :return:
            A Pandas DataFrame with aggregate features for a single video recording and a path for graphs and charts.
        """
        aggregate_features = self.__derive_features()
        file_name = Path(self.pose_keypoints_path).name
        path = str(Path(self.save_path).joinpath(
            f"{self.experiment_type}/{str(file_name).split('.')[0]}")
        )
        return aggregate_features, path

    def __derive_features(self) -> Union[pd.DataFrame, None]:
        """
        Derive time-series and summary features for every video recording based on detected keypoints for the Y-Maze
        Experiment.
        :return:
            Pandas DataFrame with summary features for the classification or None.
        """
        if self.pose_keypoints_path is None or len(self.pose_keypoints_path) == 0:
            print("[Feature Engineering] Warning! The pose keypoints path is empty!")
            return None

        if self.box_keypoints_path is None or len(self.box_keypoints_path) == 0:
            print("[Feature Engineering] Warning! The box keypoints path is empty!")
            return None

        df = read_pose_keypoints(self.pose_keypoints_path, self.video_config)
        regions = self.__get_regions()
        df = self.__add_auxiliary_features(df, regions)
        df = df.interpolate()

        time_series_features = self.__calculate_time_series_features(df)
        aggregate_features = self.__calculate_aggregate_features(df)

        self.__plot_graphs(df, time_series_features, aggregate_features)

        time_series_features = self.__format_time_series_features(time_series_features)
        formatted_aggregate_features = self.__format_aggregate_features(aggregate_features)

        save_features(self.pose_keypoints_path, self.save_path, self.experiment_type, time_series_features,
                      formatted_aggregate_features)
        return aggregate_features

    @staticmethod
    def __format_time_series_features(time_series_features: pd.DataFrame) -> pd.DataFrame:
        """
        Format time series features for saving as Excel.
        :param time_series_features:
            Pandas DataFrame of time series features.
        :return:
            Formatted Pandas DataFrame of time series features.
        """
        time_series_features.drop(columns=["mice_location"], inplace=True)
        time_series_features.rename(columns={
            "Timestamp": "Timestamp (s)",
            "in_arm1": "In Arm 1",
            "in_arm2": "In Arm 2",
            "in_arm3": "In Arm 3",
            "in_center": "In Center",
        }, inplace=True)
        return time_series_features

    @staticmethod
    def __format_aggregate_features(aggregate_features: pd.DataFrame) -> list:
        """
        Format aggregate features for saving as Excel.
        :param aggregate_features:
            Pandas DataFrame of aggregate features.
        :return:
            List of Pandas DataFrames with aggregate features.
        """
        main_features = [
            'distance_traveled',
            'time_spent_arm1',
            'time_spent_arm2',
            'time_spent_arm3',
            'time_spent_center',
            'entries_arm1',
            'entries_arm2',
            'entries_arm3',
            'entries_center',
            'total_alternation',
            'alternation_index'
        ]
    
        results = []
    
        def __remove_suffix(column_name):
            for name in main_features:
                if name in column_name:
                    return name
        
            return None
    
        for i in range(0, len(aggregate_features.columns), len(main_features)):
            temp_df = aggregate_features.iloc[:, i:i + len(main_features)]
            columns = [__remove_suffix(x) for x in list(temp_df.columns)]
            temp_df.columns = columns
        
            temp_df = temp_df[main_features]
            temp_df.rename(columns={
                'distance_traveled': "Total Distance (cm)",
                'time_spent_arm1': "Arm 1 Time (s)",
                'time_spent_arm2': "Arm 2 Time (s)",
                'time_spent_arm3': "Arm 3 Time (s)",
                'time_spent_center': "Center Time (s)",
                'entries_arm1': "Arm 1 Entries",
                'entries_arm2': "Arm 2 Entries",
                'entries_arm3': "Arm 3 Entries",
                'entries_center': "Center Entries",
                'total_alternation': "Total Alternations",
                'alternation_index': "Alternation Index"
            }, inplace=True)
        
            results += [temp_df]
    
        return results

    def __get_regions(self) -> list:
        """
        Get border coordinates for each arm and the center of the Y-Maze.
        :return:
            List of lists of tuples with x and y coordinates of borders for each arm and the center.
        """
        print("[Feature Engineering] Getting border coordinates...")

        if os.path.splitext(self.box_keypoints_path)[1] == ".h5":
            border_tables = h5py.File(self.box_keypoints_path, 'r')
            border_x, border_y = border_tables['tracks'][0]
            border_x, border_y = border_x.transpose(), border_y.transpose()
    
            best_score_idx = np.argmax(border_tables["instance_scores"][0])
            border_x_mean, border_y_mean = border_x[best_score_idx], border_y[best_score_idx]
            border_y_mean = self.video_config["height"] - border_y_mean
        else:
            border_x_mean, border_y_mean = [], []
    
            with open(self.box_keypoints_path, 'r') as f:
                num_corners = int(f.readline())
        
                for i in range(num_corners):
                    x, y = list(map(int, f.readline().split(',')))
                    border_x_mean.append(x)
                    border_y_mean.append(y)
    
        # Assigning coordinates to each border nodes
        a1_l = (border_x_mean[0], border_y_mean[0])
        a1_r = (border_x_mean[1], border_y_mean[1])
        a2_l = (border_x_mean[2], border_y_mean[2])
        a2_r = (border_x_mean[3], border_y_mean[3])
        a3_l = (border_x_mean[4], border_y_mean[4])
        a3_r = (border_x_mean[5], border_y_mean[5])
        c1 = (border_x_mean[6], border_y_mean[6])
        c2 = (border_x_mean[7], border_y_mean[7])
        c3 = (border_x_mean[8], border_y_mean[8])
    
        arm1 = [c1, a1_l, a1_r, c2]
        arm2 = [c2, a2_l, a2_r, c3]
        arm3 = [c3, a3_l, a3_r, c1]
        center = [c1, c2, c3]
    
        return [arm1, arm2, arm3, center]

    def __add_auxiliary_features(self, df: pd.DataFrame, regions: list) -> pd.DataFrame:
        """
        Derive auxiliary features for calculating time-series and aggregate features.
        :param df:
            Pandas DataFrame of pose keypoints.
        :param regions:
            List of tuples with x and y coordinates of each arm and the center of the Y-Maze.
        :return:
            Pandas DataFrame with newly added auxiliary features.
        """
        df['Mean_Body_Position_x'] = df[['nose_x', 'neck_x', 'hip_x', 'tail_base_x']].mean(axis=1)
        df['Mean_Body_Position_y'] = df[['nose_y', 'neck_y', 'hip_y', 'tail_base_y']].mean(axis=1)

        df['mice_location'] = df[['nose_x', 'nose_y', 'Mean_Body_Position_x',
                                  'Mean_Body_Position_y', 'tail_base_x', 'tail_base_y']].apply(
            lambda x: self.__check_arm(x, regions), axis=1
        )

        # Creating separate time series columns for each maze region to record whether or not the mice is in that region
        df['in_arm1'] = df['mice_location'].map(lambda x: True if x == "ARM1" else False)
        df['in_arm2'] = df['mice_location'].map(lambda x: True if x == "ARM2" else False)
        df['in_arm3'] = df['mice_location'].map(lambda x: True if x == "ARM3" else False)
        df['in_center'] = df['mice_location'].map(lambda x: True if x == "CENTER" else False)
        
        return df

    def __calculate_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a set of requested time-series features.
        :param df:
            A Pandas DataFrame with pose keypoints and auxiliary features.
        :return:
            A Pandas DataFrame of time-series features for the video recording.
        """
        timestamp_features = df.iloc[:, -5:]
        timestamp_features["Timestamp"] = self.video_config["duration"] / len(df) * timestamp_features.index
        
        cols = timestamp_features.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        timestamp_features = timestamp_features[cols]
        
        return timestamp_features

    def __calculate_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a set of requested aggregate features.
        :param df:
            A Pandas DataFrame with pose keypoints and auxiliary features.
        :return:
            A Pandas DataFrame of aggregate features for the video recording.
        """
        time_spent_arm1, time_spent_arm2, time_spent_arm3, time_spent_center = self.__time_spent(df)
        entries_arm1, entries_arm2, entries_arm3, entries_center = self.__calc_entries(df)
        total_entries = sum([entries_arm1, entries_arm2, entries_arm3, entries_center])
        total_alternation = self.__calc_alterations(df)
        alternation_index = total_alternation * 100 / (total_entries - 2)
        distance_traveled = self.__total_distance(
            df[['Mean_Body_Position_x', 'Mean_Body_Position_y']],
            df[['Mean_Body_Position_x', 'Mean_Body_Position_y']].shift()
        )
    
        results = {
            'time_spent_arm1': [time_spent_arm1],
            'time_spent_arm2': [time_spent_arm2],
            'time_spent_arm3': [time_spent_arm3],
            'time_spent_center': [time_spent_center],
            'entries_arm1': [entries_arm1],
            'entries_arm2': [entries_arm2],
            'entries_arm3': [entries_arm3],
            'entries_center': [entries_center],
            'total_alternation': [total_alternation],
            'alternation_index': [alternation_index],
            'distance_traveled': [distance_traveled]
        }

        INTERVAL = int(self.video_config["interval"])

        for i in range(0, len(df), INTERVAL):
            temp_df = df[i:i + INTERVAL]
            
            results["time_spent_arm1{0}".format(i)], results["time_spent_arm2{0}".format(i)], \
                results["time_spent_arm3{0}".format(i)], results["time_spent_center{0}".format(i)] = self.__time_spent(
                temp_df
            )
            results["entries_arm1{0}".format(i)], results["entries_arm2{0}".format(i)], \
                results["entries_arm3{0}".format(i)], results["entries_center{0}".format(i)] = self.__calc_entries(
                temp_df
            )
            total_entries = sum([
                results["entries_arm1{0}".format(i)],
                results["entries_arm2{0}".format(i)],
                results["entries_arm3{0}".format(i)],
                results["entries_center{0}".format(i)]
            ])
            results["total_alternation{0}".format(i)] = self.__calc_alterations(temp_df)
            results["alternation_index{0}".format(i)] = results["total_alternation{0}".format(i)] * 100 / (total_entries
                                                                                                           - 2)
            results["distance_traveled{0}".format(i)] = self.__total_distance(
                temp_df[['Mean_Body_Position_x', 'Mean_Body_Position_y']],
                temp_df[['Mean_Body_Position_x', 'Mean_Body_Position_y']].shift()
            )

        results = pd.DataFrame(results)
        return results

    def __plot_graphs(self, df: pd.DataFrame, time_series_features: pd.DataFrame, aggregate_features: pd.DataFrame):
        """
        Plot useful graphs and charts for the user.
        :param df:
            Pandas DataFrame with auxiliary features.
        :param time_series_features:
            Pandas DataFrame with time-series features.
        :param aggregate_features:
            Pandas DataFrame with derived aggregate features.
        """
        self.__plot_contour_graph(df)
        plot_timeline(
            time_series_features,
            ["in_arm1", "in_arm2", "in_arm3", "in_center"],
            ["tab:blue", "tab:red", "tab:green", "tab:orange"],
            ["Arm 1", "Arm 2", "Arm 3", "Center"],
            "Region",
            "Mouse movement timeline",
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        create_barplot(
            aggregate_features,
            ["time_spent_arm1", "time_spent_arm2", "time_spent_arm3", "time_spent_center"],
            "red",
            "Time spent in regions",
            "Region",
            "Time (s)",
            ["Arm 1", "Arm 2", "Arm 3", "Center"],
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        create_barplot(
            aggregate_features,
            ["entries_arm1", "entries_arm2", "entries_arm3", "entries_center"],
            "blue",
            "Number of entries into regions",
            "Region",
            "Frequency",
            ["Arm 1", "Arm 2", "Arm 3", "Center"],
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        
    def __plot_contour_graph(self, df: pd.DataFrame):
        """
        Plot the movement contour graph of the target mouse.
        :param df:
            Pandas DataFrame with auxiliary features.
        """
        title = "Contour plot of mouse movement"
        file_name = Path(self.pose_keypoints_path).name
        path = str(Path(self.save_path).joinpath(
            f"{self.experiment_type}/{str(file_name).split('.')[0]}")
        )

        if not os.path.exists(path):
            os.makedirs(path)

        df_copy = df.copy()
        df_copy.index = df_copy.index / len(df_copy) * self.video_config["duration"]
    
        fig, ax = plt.subplots(figsize=(10, 10))
        regions = self.__get_regions()
        border = []
    
        for sub_list in regions:
            border.extend(sub_list)
    
        xs, ys = zip(*border)
        ax.plot(xs, ys, color="black", zorder=-1, label="Maze")
        ax = sns.kdeplot(df[f'Mean_Body_Position_x'].values, y=df[f'Mean_Body_Position_y'], cmap="mako", shade=True,
                         thresh=0.2, legend=True)
        ax.set_xlabel('X coordinates', fontsize=16)
        ax.set_ylabel('Y coordinates', fontsize=16)
        fig.suptitle(title, fontsize=20)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(Path(path).joinpath(f"{title}.png")))
        plt.show()
        
    @staticmethod
    def __check_arm(row: pd.Series, regions: list) -> str:
        """
        Get the string name of the position of the mouse inside the Y-Maze.
        :param row:
            Row containing information about the x and y coordinates of the mouse's position.
        :param regions:
            List of tuples with x and y coordinates of each arm and the center of the Y-Maze.
        :return:
            String indicating the current position of the mouse (ARM1, ARM2, ARM3, CENTER).
        """
        arm1, arm2, arm3, center = regions
        x, y = row['Mean_Body_Position_x'], row['Mean_Body_Position_y']
        mean_body_position = Point(x, y)
        poly_arm1 = Polygon(arm1)
        poly_arm2 = Polygon(arm2)
        poly_arm3 = Polygon(arm3)
        poly_center = Polygon(center)
    
        if poly_arm1.contains(mean_body_position):
            return "ARM1"
        if poly_arm2.contains(mean_body_position):
            return "ARM2"
        if poly_arm3.contains(mean_body_position):
            return "ARM3"
        if poly_center.contains(mean_body_position):
            return "CENTER"
        
    @staticmethod
    def __entries_counter(col: pd.Series) -> int:
        """
        Count the number of times the mouse entered into the arm.
        :param col:
            Column for a given arm or center of the Y-Maze.
        :return:
            Integer denoting the number of times the mouse entered into the provided region.
        """
        # Assign the mouse position, True means mouse is in the arm
        col = list(col)
        is_in_arm = col[0]
    
        # Start counter, if mouse is already in the arm, value is 1, if not 0.
        entries = 1 if is_in_arm else 0
    
        # Loop through mouse position in each video frame
        for val in col[1:]:
            # Check for position change
            if is_in_arm != val:
                # If changed, assign the new position
                is_in_arm = val
                # If the value is True, it means mouse entered so increment entry counter
                if is_in_arm:
                    entries += 1
    
        return entries
    
    def __calc_entries(self, df: pd.DataFrame) -> list:
        """
        Calculate the number of entries done by the mouse into each arm and the center of the Y-Maze.
        :param df:
            Pandas DataFrame with derived features.
        :return:
            List of integer values denoting the number of entries for arm 1, arm 2, arm 3, and center respectively.
        """
        entries_arm1 = self.__entries_counter(df['in_arm1'])
        entries_arm2 = self.__entries_counter(df['in_arm2'])
        entries_arm3 = self.__entries_counter(df['in_arm3'])
        entries_center = self.__entries_counter(df['in_center'])
        return [entries_arm1, entries_arm2, entries_arm3, entries_center]

    def __time_spent(self, df: pd.DataFrame) -> list:
        """
        Calculate time spent in each arm and the center of the Y-Maze.
        :param df:
            Pandas DataFrame with derived features.
        :return:
            List of float values indicating the time spent in each region in seconds.
        """
        time_spent_arm1 = sum(df['in_arm1']) / self.video_config["fps"]
        time_spent_arm2 = sum(df['in_arm2']) / self.video_config["fps"]
        time_spent_arm3 = sum(df['in_arm3']) / self.video_config["fps"]
        time_spent_center = sum(df['in_center']) / self.video_config["fps"]
        return [time_spent_arm1, time_spent_arm2, time_spent_arm3, time_spent_center]

    @staticmethod
    def __calc_alterations(df: pd.DataFrame) -> int:
        """
        Calculate the total number of alterations between arms of the Y-Maze done by the mouse.
        :param df:
            Pandas DataFrame with derived features.
        :return:
            The integer value denoting the total number of alterations performed in a video recording.
        """
        travel_sequence = ""
    
        for position in df['mice_location']:
            if position == 'ARM1':
                travel_sequence += "A"
            elif position == 'ARM2':
                travel_sequence += "B"
            elif position == 'ARM3':
                travel_sequence += "C"
            elif position == 'CENTER':
                travel_sequence += "O"
    
        # removing consecutive duplicates in zone
        travel_sequence = "".join([letter for index, letter in enumerate(
            travel_sequence) if index == 0 or letter != travel_sequence[index - 1]])
    
        # removing center to get the arms visit only
        travel_sequence = travel_sequence.replace('O', '')
    
        # All possible zone alternations via permutation function
        perms = [''.join(p) for p in permutations('ABC')]
    
        total_alternation = 0
        for perm in perms:
            alternations = travel_sequence.count(perm)
            total_alternation += alternations
    
        return total_alternation

    def __total_distance(self, pos1: pd.Series, pos2: pd.Series) -> float:
        """
        Calculate the total distance traveled by the mouse inside the Y-Maze.
        :param pos1:
            The previous position of the mouse in x and y coordinates.
        :param pos2:
            The current position of the mouse in x and y coordinates.
        :return:
            Total distance traveled by the mouse inside the Y-Maze in centimeters.
        """
        distances = np.linalg.norm(pos2.values - pos1.values, axis=1)
        distances = distances[~np.isnan(distances)]
        distance_in_cm = sum(distances) / self.video_config["units_per_cm"]
        return distance_in_cm


if __name__ == "__main__":
    current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    feature_engineering = YMazeExperimentFeatureEngineering(
        pose_keypoints_path=str(current_path.joinpath("data/pose_estimation/h5_files/YM/pose/Trial     1_YM_6w_Treated_mpgb99664ff-2930-47e5-b641-125a54e759bb.mp4.slp.h5")),
        box_keypoints_path=str(current_path.joinpath("data/pose_estimation/h5_files/YM/border/Trial     1_YM_6w_Treated_mpgb99664ff-2930-47e5-b641-125a54e759bb_grayscale_short.mp4.slp.h5")),
        thresholds_config={},
        video_config=get_video_config(
            str(current_path.joinpath("videos/Trial     1_YM_6w_Treated.mpg")),
            str(current_path.joinpath("data/pose_estimation/h5_files/YM/border/Trial     1_YM_6w_Treated_mpgb99664ff-2930-47e5-b641-125a54e759bb_grayscale_short.mp4.slp.h5")),
            {},
            "YM",
            -1
        ),
        save_path=str(data_path.joinpath("data/derived_features"))
    )
    features = feature_engineering.run_feature_engineering()
