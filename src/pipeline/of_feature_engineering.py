from shapely import speedups
from shapely.geometry import Point, Polygon

speedups.disable()

from pathlib import Path
from typing import Union

import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from src.config import config, data_path
from src.pipeline.utils import create_barplot, get_video_config, plot_timeline, read_pose_keypoints, save_features


class OpenFieldExperimentFeatureEngineering:
    """
    A class that derives time-series and aggregate features for a video recording with an Open Field Experiment Type.
    """
    def __init__(self, pose_keypoints_path: str, box_keypoints_path: str, thresholds_config: dict, video_config: dict,
                 save_path: str):
        """
        :param pose_keypoints_path:
            Path to resultant pose keypoints of video recordings.
        :param box_keypoints_path:
            Path to resultant box keypoints of video recordings.
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
        self.experiment_type = "OF"

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
        Derive time-series and summary features for every video recording based on detected keypoints for the Open Field
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
        inner_corners = self.__calculate_inner_corners()
        df = self.__add_auxiliary_features(df)
        df = df.interpolate()

        time_series_features = self.__calculate_time_series_features(df, inner_corners)
        aggregate_features = self.__calculate_aggregate_features(df, inner_corners)

        cols = aggregate_features.columns.tolist()
        cols = cols[-3:] + cols[:-3]
        aggregate_features = aggregate_features[cols]

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
        time_series_features = time_series_features[["Timestamp", "Center", "Outer", "Immobile", "Walk", "Run"]]
        time_series_features.rename(columns={
            "Timestamp": "Timestamp (s)",
            "Walk": "Is Walking",
            "Center": "In Inner Zone",
            "Outer": "In Outer Zone",
            "Run": "Is Running"
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
            "DistanceInner",
            "DistanceOuter",
            "DistanceTraveled",
            "TimeCenter",
            "TimeOuter",
            "VelocityInner",
            "VelocityOuter",
            "Velocity",
            "FrequencyToEnterZones",
            "LatencyToFirstEnter",
            "Immobility",
            "TurnLeft",
            "TurnRights",
            "RunTurn"
        ]
        
        results = []
        
        def __remove_suffix(column_name):
            if "_" not in column_name:
                return column_name
            
            return "_".join(column_name.split("_")[:-1])
        
        for i in range(0, len(aggregate_features.columns), len(main_features)):
            temp_df = aggregate_features.iloc[:, i:i+len(main_features)]
            columns = [__remove_suffix(x) for x in list(temp_df.columns)]
            temp_df.columns = columns

            temp_df = temp_df[main_features]
            temp_df.rename(columns={
                "DistanceInner": "Inner Zone Distance (cm)",
                "DistanceOuter": "Outer Zone Distance (cm)",
                "DistanceTraveled": "Total Distance (cm)",
                "TimeCenter": "Inner Zone Time (s)",
                "TimeOuter": "Outer Zone Time (s)",
                "VelocityInner": "Inner Zone Average Velocity (cm/s)",
                "VelocityOuter": "Outer Zone Average Velocity (cm/s)",
                "Velocity": "Average Velocity (cm/s)",
                "FrequencyToEnterZones": "Zone Change Frequency (Hz)",
                "LatencyToFirstEnter": "First Zone Change Latency (s)",
                "Immobility": "Immobility Time (s)",
                "TurnLeft": "Number of Turns to Left",
                "TurnRights": "Number of Turns to Right",
                "RunTurn": "Number of Running and Turning"
            }, inplace=True)

            results += [temp_df]
            
        return results
    
    def __calculate_inner_corners(self) -> list:
        """
        Calculate inner zone corners' x and y coordinates.
        :return:
            A list of inner corners coordinates.
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
                    
        points = list(map(list, zip(*[border_x_mean, border_y_mean])))
        distances = [np.linalg.norm(np.array(points[0]) - np.array(points[i])) for i in range(1, len(points))]
        diag_point = np.argmax(distances) + 1
        chosen_points = {0, diag_point}

        top_left = points[0]
        bottom_right = points[diag_point]
        
        for i in range(len(points)):
            if i not in chosen_points:
                top_right = points[i]
                chosen_points.add(i)
                break

        for i in range(len(points)):
            if i not in chosen_points:
                bottom_left = points[i]
                chosen_points.add(i)
                break
    
        # Calculating center point of the box
        centroid = (((border_x_mean[0] + border_x_mean[1] + border_x_mean[2] + border_x_mean[3]) / 4),
                    ((border_y_mean[0] + border_y_mean[1] + border_y_mean[2] + border_y_mean[3]) / 4))
    
        outer_box_dist = tuple(map(lambda i, j: abs(i - j), centroid, top_right))
        divisornodecimals = 2
        inner_box_dist = tuple(map(lambda x: abs(x / divisornodecimals), outer_box_dist))

        top_left_inner = tuple(map(lambda i, j: abs(i - j), top_left, inner_box_dist))
        top_right_inner = tuple(map(lambda i, j: abs(i - j), top_right, inner_box_dist))
        bottom_right_inner = tuple(map(lambda i, j: abs(i - j), bottom_right, inner_box_dist))
        bottom_left_inner = tuple(map(lambda i, j: abs(i - j), bottom_left, inner_box_dist))
        
        xs, ys = zip(*[top_left, top_right, bottom_right, bottom_left])
        plt.plot(xs, ys)
        plt.show()
        
        return [top_left_inner, top_right_inner, bottom_right_inner, bottom_left_inner]

    @staticmethod
    def __add_auxiliary_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive auxiliary features for calculating time-series and aggregate features.
        :param df:
            Pandas DataFrame of pose keypoints.
        :return:
            Pandas DataFrame with newly added auxiliary features.
        """
        df = df.rename(columns={"body_1_x": "Mean_Body_Position_x", "body_1_y": "Mean_Body_Position_y"})
        
        # df['Mean_Body_Position_x'] = df[['nose_x', 'head_x', 'body_1_x', 'body_2_x', 'back_x']].mean(axis=1)
        # df['Mean_Body_Position_y'] = df[['nose_y', 'head_y', 'body_1_y', 'body_2_y', 'back_y']].mean(axis=1)

        # Immobility
        # Threshold: if change in mean position is > 1, than the mouse is mobile
        df['is_immobile_x'] = df['Mean_Body_Position_x'].diff()
        df['is_immobile_y'] = df['Mean_Body_Position_y'].diff()
        df['is_immobile_x'] = df['is_immobile_x'].abs()
        df['is_immobile_y'] = df['is_immobile_y'].abs()
        
        return df
        
    def __calculate_time_series_features(self, df: pd.DataFrame, inner_corners: list) -> pd.DataFrame:
        """
        Calculate a set of requested time-series features.
        :param df:
            A Pandas DataFrame with pose keypoints and auxiliary features.
        :param inner_corners:
            A list of tuples with x and y coordinates of inner zone corners.
        :return:
            A Pandas DataFrame of time-series features for the video recording.
        """
        video_length = self.video_config["duration"]
        frame_size_sec = video_length / len(df)
        
        cols = ['Timestamp']
        df_timestamp = pd.DataFrame(columns=cols, index=range(len(df)))

        frame_temp = frame_size_sec
        
        for a in range(len(df)):
            df_timestamp.loc[a].Timestamp = frame_temp
            frame_temp += frame_size_sec
            
        df_timestamp['Center'] = df.apply(lambda x: self.__is_in_center(x, inner_corners), axis=1)
        df_timestamp['Outer'] = df.apply(lambda x: self.__is_in_outer(x, inner_corners), axis=1)
        df_timestamp['Immobile'] = df.apply(self.__immobility_check, axis=1)

        df_timestamp_travel_prev = df[["Mean_Body_Position_x", "Mean_Body_Position_y"]]
        df_timestamp_travel_next = df[["Mean_Body_Position_x", "Mean_Body_Position_y"]].diff()
        distances = np.linalg.norm(df_timestamp_travel_next.values - df_timestamp_travel_prev.values, axis=1)
        distances[0] = 0
        distances /= self.video_config["units_per_cm"]
        df_timestamp["Diff"] = distances

        df_timestamp["Run"] = df_timestamp.apply(self.__check_run, axis=1)
        df_timestamp["Walk"] = (~df_timestamp["Run"] & ~df_timestamp["Immobile"])
        
        df_timestamp = df_timestamp[["Timestamp", "Walk", "Center", "Outer", "Immobile", "Run"]]
        
        return df_timestamp
        
    def __calculate_aggregate_features(self, df: pd.DataFrame, inner_corners: list) -> pd.DataFrame:
        """
        Calculate a set of requested aggregate features.
        :param df:
            A Pandas DataFrame with pose keypoints and auxiliary features.
        :param inner_corners:
            A list of tuples with x and y coordinates of inner zone corners.
        :return:
            A Pandas DataFrame of aggregate features for the video recording.
        """
        distance_traveled = self.__total_distance(
            df[['Mean_Body_Position_x', 'Mean_Body_Position_y']],
            df[['Mean_Body_Position_x', 'Mean_Body_Position_y']].shift(),
            self.video_config['units_per_cm']
        )

        data = {'DistanceTraveled_Total': [distance_traveled]}
        df_summary = pd.DataFrame(data)

        # Time in center
        df['is_in_center'] = df.apply(lambda x: self.__is_in_center(x, inner_corners), axis=1)
        num_frames_in_center = df.is_in_center.sum()

        fps = int(self.video_config["fps"])
        time_in_center = num_frames_in_center / fps

        df_summary['TimeCenter_Total'] = time_in_center
        
        # Time in outer part of the arena
        false_count = (~df.is_in_center).sum()
        time_not_in_center = false_count / fps
        df_summary['TimeOuter_Total'] = time_not_in_center

        # Distance inner
        distance_traveled_center = self.__total_distance(
            df.query('is_in_center==True')[['Mean_Body_Position_x', 'Mean_Body_Position_y']],
            df.query('is_in_center==True')[['Mean_Body_Position_x', 'Mean_Body_Position_y']].shift(),
            self.video_config['units_per_cm']
        )

        df_summary['DistanceInner_Total'] = distance_traveled_center
        distance_traveled_not_center = distance_traveled - distance_traveled_center
        df_summary['DistanceOuter_Total'] = distance_traveled_not_center
        time = time_in_center + time_not_in_center

        velocity = self.__total_velocity(distance_traveled, time)
        df_summary['Velocity_Total'] = velocity

        velocity_center = self.__total_velocity(distance_traveled_center, time_in_center)
        df_summary['VelocityInner_Total'] = velocity_center
        velocity_outer = self.__total_velocity(distance_traveled_not_center, time_not_in_center)
        df_summary['VelocityOuter_Total'] = velocity_outer

        df['change_zone'] = df['is_in_center'].diff()
        frequency_to_enter = df.change_zone.sum()
        df_summary['FrequencyToEnterZones_Total'] = frequency_to_enter / self.video_config["duration"]

        df['is_immobile_boolean'] = df.apply(self.__immobility_check, axis=1)
        
        # Count true instances for mice being in center in dataframe of trial
        temp_im = df.is_immobile_boolean.sum()
        time_immobile = temp_im / fps
        df_summary['Immobility_Total'] = time_immobile

        # Latency to first enter zones
        temp_df_2 = df
        temp_df_2 = temp_df_2.iloc[1:]

        list_latency = temp_df_2['change_zone'].tolist()
        counter = 0
        numb = list_latency[1]

        for x in range(len(list_latency)):
            if numb == list_latency[x]:
                counter = counter + 1
            else:
                break

        latency = counter / fps
        df_summary['LatencyToFirstEnter_Total'] = latency

        # BINS
        binX = 1
        binY = 1500
        total_turn_left, total_turn_right = 0, 0
        total_run_turn = 0
        
        while binY < len(df):
            very_temp_df = df.iloc[binX:binY]
            distance_traveled_bin = self.__total_distance(
                very_temp_df[['Mean_Body_Position_x', 'Mean_Body_Position_y']],
                very_temp_df[['Mean_Body_Position_x', 'Mean_Body_Position_y']].shift(),
                self.video_config['units_per_cm']
            )
            distance_traveled_center = self.__total_distance(
                very_temp_df.query('is_in_center==True')[['Mean_Body_Position_x', 'Mean_Body_Position_y']],
                very_temp_df.query('is_in_center==True')[['Mean_Body_Position_x', 'Mean_Body_Position_y']].shift(),
                self.video_config['units_per_cm']
            )
            distance_traveled_not_center = distance_traveled_bin - distance_traveled_center
            time_in_center_bin = very_temp_df.is_in_center.sum()
            frequency_to_enter = very_temp_df.change_zone.sum() / self.video_config["duration"]
            false_count = (~very_temp_df.is_in_center).sum()
            time_not_in_center_bin = false_count / fps
            time_in_center_bin = time_in_center_bin / fps
            total_time = time_in_center_bin + time_not_in_center_bin
            velocity = self.__total_velocity(distance_traveled_bin, total_time)
            velocity_center = self.__total_velocity(distance_traveled_center, time_in_center_bin)
            velocity_outer = self.__total_velocity(distance_traveled_not_center, time_not_in_center_bin)
            temp_im = very_temp_df.is_immobile_boolean.sum()
            time_immobile = temp_im / fps

            temp_df_2 = very_temp_df
            temp_df_2 = temp_df_2.iloc[1:]

            list_latency = temp_df_2['change_zone'].tolist()
            counter = 0
            numb = list_latency[1]

            for x in range(len(list_latency)):
                if numb == list_latency[x]:
                    counter = counter + 1
                else:
                    break
                    
            latency = counter / fps

            turn_right = 0
            turn_left = 0

            temp_df = very_temp_df[['Mean_Body_Position_x', 'Mean_Body_Position_y']]
            temp_df = temp_df.rolling(fps).mean()[::fps]
            temp_df = temp_df.iloc[1:]

            q = 2
            w = 4

            while w < len(temp_df.index):
                line1 = temp_df.iloc[:q].values
                line2 = temp_df.iloc[q:w].values
    
                line1 = line1 / np.linalg.norm(line1)
                line2 = line2 / np.linalg.norm(line2)
    
                angle1 = math.atan2(line1[0][1] - line1[1][1], line1[0][0] - line1[1][0])
                angle2 = math.atan2(line2[0][1] - line2[1][1], line2[0][0] - line2[1][0])
                angleDegrees = (angle2 - angle1) * 360 / (2 * math.pi)
                
                if angleDegrees < 0:
                    final_ang = angleDegrees + 360
                else:
                    final_ang = angleDegrees
    
                if final_ang > 180:
                    turn_right = turn_right + 1
                if final_ang < 180:
                    turn_left = turn_left + 1
    
                q = q + 2
                w = w + 2

            df_summary['TimeCenter_' + str(binY)] = time_in_center_bin
            df_summary['TimeOuter_' + str(binY)] = time_not_in_center_bin
            df_summary['DistanceTraveled_' + str(binY)] = distance_traveled_bin
            df_summary['DistanceInner_' + str(binY)] = distance_traveled_center
            df_summary['DistanceOuter_' + str(binY)] = distance_traveled_not_center
            df_summary['Velocity_' + str(binY)] = velocity
            df_summary['VelocityInner_' + str(binY)] = velocity_center
            df_summary['VelocityOuter_' + str(binY)] = velocity_outer
            df_summary['FrequencyToEnterZones_' + str(binY)] = frequency_to_enter
            df_summary['Immobility_' + str(binY)] = time_immobile
            df_summary['LatencyToFirstEnter_' + str(binY)] = latency
            df_summary['TurnRights_' + str(binY)] = turn_right
            df_summary['TurnLeft_' + str(binY)] = turn_left
            temp = df_summary.apply(lambda row: self.__is_running(row, binY), axis=1)
            df_summary['RunTurn_' + str(binY)] = df_summary.apply(lambda row: self.__runturn(row, temp, binY), axis=1)
            
            total_turn_left += turn_left
            total_turn_right += turn_right
            total_run_turn += df_summary["RunTurn_" + str(binY)]
            
            binX += 1500
            binY += 1500

        df_summary['TurnLeft'] = total_turn_left
        df_summary['TurnRights'] = total_turn_right
        df_summary['RunTurn'] = total_run_turn
        
        return df_summary
    
    def __plot_graphs(self, df: pd.DataFrame, time_series_features: pd.DataFrame, aggregate_features: pd.DataFrame):
        """
        Plot useful graphs and charts for the user.
        :param df:
            Pandas DataFrame with auxiliary features.
        :param time_series_features:
            Pandas DataFrame with derived time-series features for the video recording.
        :param aggregate_features:
            Pandas DataFrame with derived aggregate features for the video recording.
        """
        self.__plot_mouse_movement(df)
        plot_timeline(
            time_series_features,
            ["Immobile", "Run", "Walk"],
            ["tab:red", "tab:blue", "tab:green"],
            ["Immobile", "Running", "Walking"],
            "Action",
            "Mouse mobility timeline",
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        plot_timeline(
            time_series_features,
            ["Center", "Outer"],
            ["tab:red", "tab:blue"],
            ["Center", "Outer"],
            "Zone",
            "Mouse movement by zones timeline",
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        create_barplot(
            aggregate_features,
            ["DistanceInner_Total", "DistanceOuter_Total"],
            "green",
            "Distance traveled by the mouse",
            "Zone",
            "Distance (cm)",
            ["Center", "Outer"],
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        create_barplot(
            aggregate_features,
            ["TimeCenter_Total", "TimeOuter_Total"],
            "red",
            "Time the mouse spent in zones",
            "Zone",
            "Time (s)",
            ["Center", "Outer"],
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        create_barplot(
            aggregate_features,
            ["Velocity_Total", "VelocityInner_Total", "VelocityOuter_Total"],
            "blue",
            "Average velocity of the mouse",
            "Zone",
            "Average Velocity (cm/s)",
            ["Both", "Center", "Outer"],
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
    
    def __plot_mouse_movement(self, df: pd.DataFrame):
        """
        Plot the movement of the mouse through its central body position.
        :param df:
            Pandas DataFrame with auxiliary features.
        """
        title = "Movement of the mouse"
        file_name = Path(self.pose_keypoints_path).name
        path = str(Path(self.save_path).joinpath(
            f"{self.experiment_type}/{str(file_name).split('.')[0]}")
        )
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        df_copy = df.copy()
    
        plt.figure(figsize=(10, 10))
        plt.plot(df_copy["Mean_Body_Position_x"].values, df_copy["Mean_Body_Position_y"].values, label="Target")
        plt.title(title, fontsize=20)
        plt.legend()
        plt.savefig(str(Path(path).joinpath(f"{title}.png")))
        plt.show()
    
    @staticmethod
    def __total_distance(pos1: pd.Series, pos2: pd.Series, units_per_cm: float) -> float:
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
    def __immobility_check(row: pd.Series) -> bool:
        """
        Check if the mouse is moving or not.
        :param row:
            Row containing x and y coordinates of the mouse's immobility.
        :return:
            Boolean value indicating if the mouse is moving or not.
        """
        if row['is_immobile_x'] < 3 and row['is_immobile_y'] < 3:
            val = True
        else:
            val = False
        return val
    
    @staticmethod
    def __total_velocity(distance_traveled: float, time: float) -> float:
        """
        Calculating average velocity for a video recording.
        :param distance_traveled:
            Total distance traveled by the mouse.
        :param time:
            Duration of the video recording.
        :return:
            Average velocity of the mouse in the video recording.
        """
        velocityT = distance_traveled / time
        return velocityT
    
    @staticmethod
    def __is_in_center(row: pd.Series, corners: list) -> bool:
        """
        Checking whether the mouse is in the central zone or not.
        :param row:
            Row having x and y coordinates of the mouse's body position.
        :param corners:
            List of tuples with x and y coordinates of the central zone's corners.
        :return:
            Boolean value indicating if the mouse is in the center or not.
        """
        body_coordinates = (row['Mean_Body_Position_x'], row['Mean_Body_Position_y'])
        p1 = Point(body_coordinates)
        poly = Polygon(corners)
        temp2 = p1.within(poly)
        return temp2

    def __is_running(self, row: pd.Series, x: int) -> int:
        """
        Checking if the mouse is running or not.
        :param row:
            Row containing velocity value of the mouse.
        :param x:
            Value for differentiating the bin of the video recording.
        :return:
            Integer value denoting if the mouse is running (1) or not (0).
        """
        if len(self.thresholds_config) > 0:
            velocity_threshold = self.thresholds_config["velocity_threshold"]
        else:
            velocity_threshold = config["experiments"][self.experiment_type]["velocity_threshold"]
            
        if float(row['Velocity_' + str(x)]) > velocity_threshold:
            val = 1
        else:
            val = 0
            
        return val
    
    def __check_run(self, row: pd.Series) -> bool:
        """
        Checking if the mouse is running or not (boolean output).
        :param row:
            Row containing the value of the difference in mouse's position between frames.
        :return:
            Boolean value indicating if the mouse is running (True) or walking (False).
        """
        if len(self.thresholds_config) > 0:
            velocity_threshold = self.thresholds_config["velocity_threshold"]
        else:
            velocity_threshold = config["experiments"][self.experiment_type]["velocity_threshold"]
        
        if not row["Immobile"] and row['Diff'] > velocity_threshold:
            val = True
        else:
            val = False
            
        return val
    
    @staticmethod
    def __is_in_outer(row: pd.Series, corners: list) -> bool:
        """
        Checking if the mouse is in the outer zone of the arena.
        :param row:
            Row having x and y coordinates of the mouse's body position.
        :param corners:
            A list of tuples with x and y coordinates of the arena corners.
        :return:
            Boolean value indicating if the mouse is in the outer region of the arena or not.
        """
        body_coordinates = (row['Mean_Body_Position_x'], row['Mean_Body_Position_y'])
        p1 = Point(body_coordinates)
        poly = Polygon(corners)
        temp2 = p1.within(poly)
        return not temp2
    
    @staticmethod
    def __runturn(row: pd.Series, running: pd.Series, x: int) -> int:
        """
        Checking if the mouse is running and turning or not.
        :param row:
            Row having values of running and turning (left or right).
        :param running:
            Pandas Series with values denoting whether the mouse was running or not.
        :param x:
            Value for differentiating the bin of the video recording.
        :return:
            Integer value denoting if the mouse is running and turning (1) or not (0).
        """
        if float(running) == 1 and (float(row['TurnLeft_' + str(x)]) > 0 or float(row['TurnRights_' + str(x)]) > 0):
            val = 1
        else:
            val = 0
            
        return val


if __name__ == "__main__":
    current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    feature_engineering = OpenFieldExperimentFeatureEngineering(
        pose_keypoints_path=str(current_path.joinpath(
            "temp/pose/Trial1_v1_OF_w2_Control1_2593_pose.mp4.predictions.analysis.h5"
        )),
        box_keypoints_path=str(current_path.joinpath(
            "temp/border/Trial1_v1_OF_w2_Control1_2593_border.mp4.predictions.analysis.h5"
        )),
        thresholds_config={},
        video_config=get_video_config(
            str(current_path.joinpath("temp/videos/Trial1_v1_OF_w2_Control1_2593.mp4")),
            str(current_path.joinpath(
                "temp/border/Trial1_v1_OF_w2_Control1_2593_border.mp4.predictions.analysis.h5"
            )),
            {},
            "OF",
            -1,
        ),
        save_path=str(data_path.joinpath("data/derived_features"))
    )
    features = feature_engineering.run_feature_engineering()
