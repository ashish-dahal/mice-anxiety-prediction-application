from shapely import speedups
from shapely.geometry import Point, Polygon

speedups.disable()

from pathlib import Path
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

from src.config import data_path
from src.pipeline.utils import create_barplot, get_video_config, plot_timeline, read_pose_keypoints, save_features

warnings.filterwarnings('ignore')


class ElevatedPlusMazeFeatureEngineering:
    """
    A class that derives time-series and aggregate features for a video recording with an Elevated Plus Maze Experiment
    Type.
    """
    def __init__(self, pose_keypoints_path: str, box_keypoints_path: str, thresholds_config: dict, video_config: dict,
                 save_path: str):
        """
        :param pose_keypoints_path:
            Path to resultant pose keypoints of video recordings.
        :param box_keypoints_path:
            Path to resultant maze keypoints of video recordings.
        :param thresholds_config:
            A dictionary of threshold values for the experiment.
        :param video_config:
            Video recordings information such as units per cm, FPS, duration, etc.
        :param save_path:
            A path to the folder where to save graphs and features.
        """
        self.pose_keypoints_path = pose_keypoints_path
        self.box_keypoints_path = box_keypoints_path
        self.thresholds_config = thresholds_config
        self.video_config = video_config
        self.save_path = save_path
        self.experiment_type = "EPM"

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
        Derive time-series and summary features for every video recording based on detected keypoints for the Elevated
        Plus Maze Experiment.
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
        maze_coords = self.__get_maze_coords()
        df = self.__add_auxiliary_features(maze_coords, df)
        
        time_series_features = self.__calculate_time_series_features(df)
        aggregate_features = self.__calculate_aggregate_features(df)

        self.__plot_graphs(maze_coords, df, time_series_features, aggregate_features)

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
        time_series_features = time_series_features[[
            "Timestamp",
            "is_in_open",
            "is_in_closed",
            "is_in_center",
            "nose_in_center",
            "is_dipping"]]
        time_series_features.rename(columns={
            "Timestamp": "Timestamp (s)",
            "is_in_open": "In Open Arm",
            "is_in_closed": "In Closed Arm",
            "is_in_center": "In Center",
            "nose_in_center": "Nose in Center",
            "is_dipping": "Nose is Dipping"
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
            "distance_traveled_open",
            "distance_traveled_closed",
            "frequency_of_entry_to_open",
            "frequency_of_entry_to_closed",
            "time_in_open",
            "time_in_closed",
            "time_head_dipping",
            "latency_to_enter_open"
        ]
    
        results = []
    
        def __remove_suffix(column_name):
            for name in main_features:
                if name in column_name:
                    return name
                
            if "open_frequency" in column_name:
                return "frequency_of_entry_to_open"
            elif "closed_frequency" in column_name:
                return "frequency_of_entry_to_closed"
            elif "latency_to_open" in column_name:
                return "latency_to_enter_open"
                
            return None
    
        for i in range(0, len(aggregate_features.columns), len(main_features)):
            temp_df = aggregate_features.iloc[:, i:i + len(main_features)]
            columns = [__remove_suffix(x) for x in list(temp_df.columns)]
            temp_df.columns = columns

            temp_df = temp_df[main_features]
            temp_df.rename(columns={
                "distance_traveled_open": "Open Arm Distance (cm)",
                "distance_traveled_closed": "Closed Arm Distance (cm)",
                "frequency_of_entry_to_open": "Open Arm Frequency of Entry (Hz)",
                "frequency_of_entry_to_closed": "Closed Arm Frequency of Entry (Hz)",
                "time_in_open": "Open Arm Time (s)",
                "time_in_closed": "Closed Arm Time (s)",
                "time_head_dipping": "Head Dipping Time (s)",
                "latency_to_enter_open": "Open Arm First Entry Latency (s)"
            }, inplace=True)

            results += [temp_df]
    
        return results
    
    def __get_maze_coords(self) -> list:
        """
        Calculate coordinates of the maze.
        :return:
            List of x and y coordinates of the maze.
        """
        print("[Feature Engineering] Getting border coordinates...")
        
        # Defining maze
        if os.path.splitext(self.box_keypoints_path)[1] == ".h5":
            maze_tables = h5py.File(self.box_keypoints_path, 'r')
            maze_x, maze_y = maze_tables['tracks'][0]
            maze_x, maze_y = maze_x.transpose(), maze_y.transpose()
    
            best_score_idx = np.argmax(maze_tables["instance_scores"][0])
            maze_x_mean, maze_y_mean = maze_x[best_score_idx], maze_y[best_score_idx]
            maze_y_mean = self.video_config["height"] - maze_y_mean
        else:
            maze_x_mean, maze_y_mean = [], []
    
            with open(self.box_keypoints_path, 'r') as f:
                num_corners = int(f.readline())
        
                for i in range(num_corners):
                    x, y = list(map(int, f.readline().split(',')))
                    maze_x_mean.append(x)
                    maze_y_mean.append(y)
    
        c1 = (maze_x_mean[0], maze_y_mean[0])
        c2 = (maze_x_mean[1], maze_y_mean[1])
        c3 = (maze_x_mean[2], maze_y_mean[2])
        c4 = (maze_x_mean[3], maze_y_mean[3])
        o1 = (maze_x_mean[4], maze_y_mean[4])
        o2 = (maze_x_mean[5], maze_y_mean[5])
        o3 = (maze_x_mean[6], maze_y_mean[6])
        o4 = (maze_x_mean[7], maze_y_mean[7])
        o5 = (maze_x_mean[8], maze_y_mean[8])
        o6 = (maze_x_mean[9], maze_y_mean[9])
        o7 = (maze_x_mean[10], maze_y_mean[10])
        o8 = (maze_x_mean[11], maze_y_mean[11])
        
        return [c1, c2, c3, c4, o1, o2, o3, o4, o5, o6, o7, o8]
    
    @staticmethod
    def __add_auxiliary_features(maze_coords: list, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive auxiliary features for calculating time-series and aggregate features.
        :param maze_coords:
            List of x and y coordinates of the maze.
        :param df:
            Pandas DataFrame of pose keypoints.
        :return:
            Pandas DataFrame with newly added auxiliary features.
        """
        c1, c2, c3, c4, o1, o2, o3, o4, o5, o6, o7, o8 = maze_coords
        
        # Calculating Mean Body Positions of mice
        df['Mean_Body_Position_x'] = df[['Nose_x', 'Body_1_x', 'Body_2_x', 'Back_x']].mean(axis=1)
        df['Mean_Body_Position_y'] = df[['Nose_y', 'Body_1_y', 'Body_2_y', 'Back_y']].mean(axis=1)
    
        df['body_position'] = list(zip(df.Mean_Body_Position_x, df.Mean_Body_Position_y))
        df['nose_position'] = list(zip(df.Nose_x, df.Nose_y))
        df['is_in_closed'] = ''
        df['nose_in_center'] = ''
        df['is_dipping'] = ''
    
        polygonA = Polygon([c1, c2, o1, o8])
        polygonB = Polygon([o5, o4, c3, c4])
        polygonC = Polygon([o8, o1, o4, o5])
        polygonE = Polygon([o1, o2, o3, o4])
        polygonF = Polygon([o8, o5, o6, o7])
    
        visited = []
        nose_visited = []
        dip_visited = []
    
        # For body
        for i in range(0, len(df['body_position'])):
            if df['body_position'][i] not in visited:
                point = Point(df['body_position'][i][0], df['body_position'][i][1])
                visited.append(df['body_position'][i])
            
                if point.within(polygonA) | point.within(polygonB):
                    df['is_in_closed'][i] = 'Close'
                elif point.within(polygonC):
                    df['is_in_closed'][i] = 'Center'
                else:
                    df['is_in_closed'][i] = 'Open'

        for i in range(0, len(df['nose_position'])):
            if df['nose_position'][i] not in nose_visited:
                nose_point = Point(df['nose_position'][i][0], df['nose_position'][i][1])
                nose_visited.append(df['nose_position'][i])
                if nose_point.within(polygonC):
                    df['nose_in_center'][i] = 1
                else:
                    df['nose_in_center'][i] = 0
    
        # For dipping
        for i in range(0, len(df['nose_position'])):
            if df['nose_position'][i] not in dip_visited:
                nose_point = Point(df['nose_position'][i][0], df['nose_position'][i][1])
                dip_visited.append(df['nose_position'][i])
            
                if nose_point.within(polygonA) | nose_point.within(polygonB) | nose_point.within(
                        polygonC) | nose_point.within(polygonE) | nose_point.within(polygonF):
                    df['is_dipping'][i] = 0
                else:
                    df['is_dipping'][i] = 1
                    
        return df
    
    def __calculate_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a set of requested time-series features.
        :param df:
            A Pandas DataFrame with pose keypoints and auxiliary features.
        :return:
            A Pandas DataFrame of time-series features for the video recording.
        """
        num_columns_to_drop = 24
        df_timestamp = df.copy()
        df_timestamp["Timestamp"] = self.video_config["duration"] / len(df) * df_timestamp.index
        df_timestamp.drop(df_timestamp.iloc[:, :num_columns_to_drop], axis=1, inplace=True)
        df_timestamp['is_in_closed'] = np.where(df['is_in_closed'] == 'Close', True, False)
        df_timestamp['is_in_open'] = np.where(df['is_in_closed'] == 'Open', True, False)
        df_timestamp['is_in_center'] = np.where(df['is_in_closed'] == 'Center', True, False)
        df_timestamp['nose_in_center'] = np.where(df_timestamp['nose_in_center'] == 1, True, False)
        df_timestamp['is_dipping'] = np.where(df_timestamp['is_dipping'] == 1, True, False)
        df_timestamp = df_timestamp[['Timestamp', 'is_in_closed', 'is_in_open', 'is_in_center', 'is_dipping',
                                     'nose_in_center']]
        return df_timestamp
    
    def __calculate_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a set of requested aggregate features.
        :param df:
            A Pandas DataFrame with pose keypoints and auxiliary features.
        :return:
            A Pandas DataFrame with aggregate features for the whole video recording and its bins.
        """
        time_in_open, time_in_closed = self.__time_spent(df['is_in_closed'], int(self.video_config['fps']))
        time_head_dipping = self.__head_dipping_time(df['is_dipping'], int(self.video_config['fps']))
        open_frequency = self.__frequency_of_entry(time_in_open, self.video_config['duration'])
        closed_frequency = self.__frequency_of_entry(time_in_closed, self.video_config['duration'])
        latency_to_open = self.__latency_to_enter_open(
            df['is_in_closed'], self.video_config['duration'],
            int(self.video_config['frame_count'])
        )
        distance_traveled_closed = self.__total_distance(
            df[['Mean_Body_Position_x', 'Mean_Body_Position_y']],
            df[['Mean_Body_Position_x', 'Mean_Body_Position_y']].shift().where(df['is_in_closed'] == 'Close'),
            self.video_config['units_per_cm']
        )
        distance_traveled_open = self.__total_distance(
            df[['Mean_Body_Position_x', 'Mean_Body_Position_y']],
            df[['Mean_Body_Position_x', 'Mean_Body_Position_y']].shift().where(df['is_in_closed'] == 'Open'),
            self.video_config['units_per_cm']
        )
    
        results = {
            'time_in_open': [time_in_open],
            'time_in_closed': [time_in_closed],
            'latency_to_enter_open': [latency_to_open],
            'time_head_dipping': [time_head_dipping],
            'frequency_of_entry_to_open': [open_frequency],
            'frequency_of_entry_to_closed': [closed_frequency],
            'distance_traveled_closed': [distance_traveled_closed],
            'distance_traveled_open': [distance_traveled_open]
        }
    
        INTERVAL = int(self.video_config['interval'])
    
        for i in range(0, len(df), INTERVAL):
            temp_df = df[i:i + INTERVAL]
        
            results["time_in_open{0}".format(i)] = self.__time_spent(
                temp_df['is_in_closed'],
                int(self.video_config['fps'])
            )[0]
            results["time_in_closed{0}".format(i)] = self.__time_spent(
                temp_df['is_in_closed'],
                int(self.video_config['fps'])
            )[1]
            results["latency_to_open{0}".format(i)] = self.__latency_to_enter_open(
                temp_df['is_in_closed'],
                self.video_config['bin_length'],
                INTERVAL
            )
            results["time_head_dipping{0}".format(i)] = self.__head_dipping_time(
                temp_df['is_dipping'],
                int(self.video_config['fps'])
            )
            results["open_frequency{0}".format(i)] = self.__frequency_of_entry(
                results["time_in_open{0}".format(i)],
                self.video_config['bin_length']
            )
            results["closed_frequency{0}".format(i)] = self.__frequency_of_entry(
                results["time_in_closed{0}".format(i)],
                self.video_config['bin_length']
            )
            results["distance_traveled_closed{0}".format(i)] = self.__total_distance(
                temp_df[['Mean_Body_Position_x', 'Mean_Body_Position_y']],
                temp_df[['Mean_Body_Position_x', 'Mean_Body_Position_y']].shift().where(
                    temp_df['is_in_closed'] == 'Close'
                ),
                self.video_config['units_per_cm']
            )
            results["distance_traveled_open{0}".format(i)] = self.__total_distance(
                temp_df[['Mean_Body_Position_x', 'Mean_Body_Position_y']],
                temp_df[['Mean_Body_Position_x', 'Mean_Body_Position_y']].shift().where(
                    temp_df['is_in_closed'] == 'Open'
                ),
                self.video_config['units_per_cm']
            )
            
        results = pd.DataFrame(results)
        return results
    
    def __plot_graphs(self, maze_coords: list, df: pd.DataFrame, time_series_features: pd.DataFrame,
                      aggregate_features: pd.DataFrame):
        """
        Plot useful graphs and charts for the user.
        :param maze_coords:
            List of x and y coordinates of the maze.
        :param df:
            Pandas DataFrame with auxiliary features.
        :param time_series_features:
            Pandas DataFrame with derived time-series features for the video recording.
        :param aggregate_features:
            Pandas DataFrame with derived aggregate features for the video recording.
        """
        self.__plot_nose_movement(df, maze_coords)
        plot_timeline(
            time_series_features,
            ["is_in_closed", "is_in_open", "is_in_center"],
            ["tab:red", "tab:green", "tab:blue"],
            ["Closed Arm", "Open Arm", "Center"],
            "Region",
            "Mouse movement timeline",
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        plot_timeline(
            time_series_features,
            ["is_dipping", "nose_in_center"],
            ["tab:green", "tab:blue"],
            ["Dipping", "Center"],
            "Region",
            "Mouse's nose movement timeline",
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        create_barplot(
            aggregate_features,
            ["time_in_open", "time_in_closed", "time_head_dipping"],
            "red",
            "Time spent by the mouse",
            "Action",
            "Time (s)",
            ["Open Arm", "Closed Arm", "Head Dipping"],
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        create_barplot(
            aggregate_features,
            ["frequency_of_entry_to_open", "frequency_of_entry_to_closed"],
            "blue",
            "Frequency of entry to arms",
            "Arm",
            "Frequency",
            ["Open Arm", "Closed Arm"],
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        create_barplot(
            aggregate_features,
            ["distance_traveled_open", "distance_traveled_closed"],
            "green",
            "Distance traveled by the mouse",
            "Arm",
            "Distance (cm)",
            ["Open Arm", "Closed Arm"],
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        
    def __plot_nose_movement(self, df: pd.DataFrame, maze_coords: list):
        """
        Plot the movement of the nose of the target mouse.
        :param df:
            Pandas DataFrame with auxiliary features.
        :param maze_coords:
            List of x and y coordinates of the maze.
        """
        title = "Movement of nose of the mouse"
        file_name = Path(self.pose_keypoints_path).name
        path = str(Path(self.save_path).joinpath(
            f"{self.experiment_type}/{str(file_name).split('.')[0]}")
        )
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        df_copy = df.copy()
        c1, c2, c3, c4, o1, o2, o3, o4, o5, o6, o7, o8 = maze_coords
    
        plt.figure(figsize=(10, 10))
        new_maze_coords = [o8, c1, c2, o1]
        xs, ys = zip(*new_maze_coords)
        plt.plot(xs, ys, color="black", zorder=-1, label="Closed Arms", linewidth=3)
        new_maze_coords = [o4, c3, c4, o5]
        xs, ys = zip(*new_maze_coords)
        plt.plot(xs, ys, color="black", zorder=-1, linewidth=3)
        new_maze_coords = [o1, o2, o3, o4]
        xs, ys = zip(*new_maze_coords)
        plt.plot(xs, ys, color="black", zorder=-1, label="Open Arms", linestyle="dashed", linewidth=3)
        new_maze_coords = [o5, o6, o7, o8]
        xs, ys = zip(*new_maze_coords)
        plt.plot(xs, ys, color="black", zorder=-1, linestyle="dashed", linewidth=3)
        plt.plot(df_copy["Nose_x"].values, df_copy["Nose_y"].values, label="Target", color="blue")
        plt.title(title, fontsize=20)
        plt.legend()
        plt.savefig(str(Path(path).joinpath(f"{title}.png")))
        plt.show()

    @staticmethod
    def __total_distance(pos1: pd.Series, pos2: pd.Series, units_per_cm: float):
        """
        Calculate the total distance traveled by the target mouse.
        :param pos1:
            Current central x and y coordinates of the mouse.
        :param pos2:
            Next central x and y coordinates of the mouse.
        :param units_per_cm:
            Units of coordinates in a single centimeter.
        :return:
            Total distance traveled in centimeters.
        """
        distances = np.linalg.norm(pos2.values - pos1.values, axis=1)
        distances = distances[~np.isnan(distances)]
        distance_in_cm = sum(distances) / units_per_cm
        return distance_in_cm

    @staticmethod
    def __time_spent(col: pd.Series, fps: int) -> tuple:
        """
        Calculate total time spent in an open and closed arm in seconds.
        :param col:
            Column of the Pandas DataFrame denoting whether the mouse is in closed or open arm.
        :param fps:
            The number of frames per second in a video recording.
        :return:
            Total time spent in open and closed arms in seconds.
        """
        time_open = (col == 'Open').sum() / fps
        time_closed = (col == 'Close').sum() / fps
        return time_open, time_closed

    @staticmethod
    def __frequency_of_entry(time: float, length: float) -> float:
        """
        Calculate the fraction of time spent in an arm.
        :param time:
            The total time of spent in an arm in seconds.
        :param length:
            The duration of the video recording in seconds.
        :return:
            The fraction of time spent in a given arm.
        """
        frequency = time / length
        return frequency

    @staticmethod
    def __head_dipping_time(col: pd.Series, fps: int) -> float:
        """
        Total time spent by the mouse dipping its head.
        :param col:
            Column of Pandas DataFrame denoting if the mouse was dipping over the open arm of the maze.
        :param fps:
            Frames per second in a video recording.
        :return:
            Total time spent by the mouse dipping its head over the open arm of the maze in seconds.
        """
        time_spent_dipping = (col == 1).sum() / fps
        return time_spent_dipping

    @staticmethod
    def __latency_to_enter_open(col, duration: float, frame_count: int) -> float:
        """
        Calculate the latency before entering the open arm.
        :param col:
            The column denoting whether the mouse is in the open arm or not.
        :param duration:
            The duration of the video recording in seconds.
        :param frame_count:
            The total number of frames in the video recording.
        :return:
            The latency before the first enter into open arm in the video recording in seconds.
        """
        frames = 0
        
        for idx, val in col.items():
            if val != 'Open':
                frames += 1
            else:
                break
                
        if frames == len(col):
            return -1
        
        latency = frames / frame_count * duration
        return latency


if __name__ == "__main__":
    current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    feature_engineering = ElevatedPlusMazeFeatureEngineering(
        pose_keypoints_path=str(current_path.joinpath(
            "temp/pose/Trial_1_v1_EPM_w2_control_1_2593.mp4.predictions.h5"
        )),
        box_keypoints_path=str(current_path.joinpath(
            "temp/border/Trial_1_v1_EPM_w2_control_1_2593.mp4.predictions.h5"
        )),
        thresholds_config={},
        video_config=get_video_config(
            str(current_path.joinpath("temp/videos/Trial_1_v1_EPM_w2_control_1_2593.mp4")),
            str(current_path.joinpath("temp/border/Trial_1_v1_EPM_w2_control_1_2593.mp4.predictions.h5")),
            {},
            "EPM",
            -1
        ),
        save_path=str(data_path.joinpath("data/derived_features"))
    )
    features = feature_engineering.run_feature_engineering()
