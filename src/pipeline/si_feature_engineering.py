from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from src.config import config, data_path
from src.pipeline.utils import get_video_config, plot_timeline, read_pose_keypoints, save_features


class SocialInteractionFeatureEngineering:
    """
    A class that derives time-series and aggregate features for a video recording with a Social Interaction Experiment
    Type.
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
        self.experiment_type = "SI"

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
        Derive time-series and summary features for every video recording based on detected keypoints for the Social
        Interaction Experiment.
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
        df = self.__add_auxiliary_features(df)
        df = df.interpolate()
        
        time_series_features = self.__calculate_time_series_features(df)
        aggregate_features = self.__calculate_aggregate_features(df)

        self.__plot_graphs(df, time_series_features)

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
        time_series_features.rename(columns={
            "Timestamp": "Timestamp (s)",
            "Close_Proximity": "Active Interaction"
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
            "distance_moved",
            "interaction_frequency",
            "time_of_active_interaction",
            "first_interaction_latency"
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
                "distance_moved": "Total Distance (cm)",
                "interaction_frequency": "Active Interaction Frequency (Hz)",
                "time_of_active_interaction": "Active Interaction Time (s)",
                "first_interaction_latency": "First Active Interaction Latency (s)"
            }, inplace=True)
        
            results += [temp_df]
    
        return results
    
    def __add_auxiliary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive auxiliary features for calculating time-series and aggregate features.
        :param df:
            Pandas DataFrame of pose keypoints.
        :return:
            Pandas DataFrame with newly added auxiliary features.
        """
        df['Mean_Body_Position_1_x'] = df[['Nose_1_x', 'Upper_Body_1_x', 'Lower_Body_1_x', 'Back_1_x']].mean(axis=1)
        df['Mean_Body_Position_1_y'] = df[['Nose_1_y', 'Upper_Body_1_y', 'Lower_Body_1_y', 'Back_1_y']].mean(axis=1)
        
        df['Mean_Body_Position_2_x'] = df[['Nose_2_x', 'Body_2_x', 'Back_2_x']].mean(axis=1)
        df['Mean_Body_Position_2_y'] = df[['Nose_2_y', 'Body_2_y', 'Back_2_y']].mean(axis=1)
        
        df['Mean_Body_Position_Distance_12'] = np.linalg.norm(
            df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']].values - df[
                ['Mean_Body_Position_2_x', 'Mean_Body_Position_2_y']].values, axis=1)
        
        if len(self.thresholds_config) > 0:
            proximity = self.thresholds_config["proximity"]
        else:
            proximity = config["experiments"][self.experiment_type]["proximity"]
        
        df['Close_Proximity'] = (df['Mean_Body_Position_Distance_12'] / self.video_config['units_per_cm']) <= \
                                proximity
        
        return df
    
    def __calculate_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a set of requested time-series features.
        :param df:
            A Pandas DataFrame with pose keypoints and auxiliary features.
        :return:
            A Pandas DataFrame of time-series features for the video recording.
        """
        df_timestamp = df.copy()
        df_timestamp = df_timestamp.loc[:, ["Close_Proximity"]]
        df_timestamp.rename(columns={"Close_Proximity": "Active_Interaction"})
        df_timestamp["Timestamp"] = self.video_config["duration"] / len(df) * df_timestamp.index
        
        cols = df_timestamp.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df_timestamp = df_timestamp[cols]
        
        return df_timestamp
    
    def __calculate_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a set of requested aggregate features.
        :param df:
            A Pandas DataFrame with pose keypoints and auxiliary features.
        :return:
            A Pandas DataFrame of aggregate features for the video recording.
        """
        time_of_interaction = self.__total_time_of_interaction(
            df['Close_Proximity'].astype(int),
            self.video_config['duration'],
            int(self.video_config['frame_count'])
        )
        first_interaction_latency = self.__latency_to_first_interaction(
            df['Close_Proximity'].astype(int),
            self.video_config['duration'],
            int(self.video_config['frame_count'])
        )
        interaction_frequency = self.__frequency_of_interaction(
            time_of_interaction,
            self.video_config['duration']
        )
        distance_traveled = self.__total_distance(
            df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']],
            df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']].shift(),
            self.video_config['units_per_cm']
        )
        results = {
            'distance_moved': [distance_traveled],
            'first_interaction_latency': [first_interaction_latency],
            'interaction_frequency': [interaction_frequency],
            'time_of_active_interaction': [time_of_interaction]
        }

        INTERVAL = int(self.video_config["interval"])

        for i in range(0, len(df), INTERVAL):
            temp_df = df[i:i + INTERVAL]
            
            results["time_of_active_interaction{0}".format(i)] = self.__total_time_of_interaction(
                temp_df['Close_Proximity'].astype(int),
                self.video_config['bin_length'],
                len(temp_df)
            )
            results["distance_moved{0}".format(i)] = self.__total_distance(
                temp_df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']],
                temp_df[['Mean_Body_Position_1_x', 'Mean_Body_Position_1_y']].shift(),
                self.video_config['units_per_cm']
            )
            results["first_interaction_latency{0}".format(i)] = self.__latency_to_first_interaction(
                temp_df["Close_Proximity"].astype(int),
                self.video_config['bin_length'],
                len(temp_df)
            )
            results["interaction_frequency{0}".format(i)] = self.__frequency_of_interaction(
                results["time_of_active_interaction{0}".format(i)],
                self.video_config['bin_length']
            )
        
        results = pd.DataFrame(results)
        return results
    
    def __plot_graphs(self, df: pd.DataFrame, time_series_features: pd.DataFrame):
        """
        Plot useful graphs and charts for the user.
        :param df:
            Pandas DataFrame with auxiliary features.
        :param time_series_features:
            Pandas DataFrame with time-series features.
        """
        self.__plot_distance_between_mice(df)
        self.__plot_distance_with_active_interaction(df)
        self.__plot_mice_movement(df)
        df_copy = time_series_features.copy()
        df_copy["Far_Proximity"] = ~df_copy["Close_Proximity"]
        plot_timeline(
            df_copy,
            ["Close_Proximity", "Far_Proximity"],
            ["tab:green", "tab:red"],
            ["Active Interaction", "No Interaction"],
            "Interaction",
            "Mice interaction timeline",
            self.save_path,
            self.experiment_type,
            self.pose_keypoints_path
        )
        
    def __plot_distance_between_mice(self, df: pd.DataFrame):
        """
        Plot the graph of distance between the bodies of the mice.
        :param df:
            Pandas DataFrame with time-series features.
        """
        title = "Distance between body centers of mice"
        file_name = Path(self.pose_keypoints_path).name
        path = str(Path(self.save_path).joinpath(
            f"{self.experiment_type}/{str(file_name).split('.')[0]}")
        )

        if not os.path.exists(path):
            os.makedirs(path)
            
        df_copy = df.copy()
        df_copy.index = df_copy.index / len(df_copy) * self.video_config["duration"]
        
        plt.figure(figsize=(15, 4))
        (df_copy["Mean_Body_Position_Distance_12"] / self.video_config['units_per_cm']).plot(color="blue")
        plt.title(title, fontsize=20)
        plt.xlim([0, max(df_copy.index)])
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Distance (cm)", fontsize=16)
        plt.grid(axis="y")
        plt.savefig(str(Path(path).joinpath(f"{title}.png")))
        plt.show()
        
    def __plot_distance_with_active_interaction(self, df: pd.DataFrame):
        """
        Plot the distance graph along with active interaction graph.
        :param df:
            Pandas DataFrame with auxiliary features.
        """
        title = "Distance between body centers of mice with active interaction"
        file_name = Path(self.pose_keypoints_path).name
        path = str(Path(self.save_path).joinpath(
            f"{self.experiment_type}/{str(file_name).split('.')[0]}")
        )

        if not os.path.exists(path):
            os.makedirs(path)
            
        df_copy = df.copy()
        df_copy.index = df_copy.index / len(df_copy) * self.video_config["duration"]
        
        plt.figure(figsize=(15, 4))
        (df_copy["Mean_Body_Position_Distance_12"] / self.video_config['units_per_cm']).plot(color="blue")
        (df_copy["Close_Proximity"] * max(df_copy["Mean_Body_Position_Distance_12"] /
                                          self.video_config["units_per_cm"])).astype(int).plot(color="red")
        plt.title(title, fontsize=20)
        plt.xlim([0, max(df_copy.index)])
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Distance (cm)", fontsize=16)
        plt.grid(axis="y")
        plt.savefig(str(Path(path).joinpath(f"{title}.png")))
        plt.show()
        
    def __plot_mice_movement(self, df: pd.DataFrame):
        """
        Plot the movement of target and non-target mice.
        :param df:
            Pandas DataFrame with auxiliary features.
        """
        title = "Movement of mice"
        file_name = Path(self.pose_keypoints_path).name
        path = str(Path(self.save_path).joinpath(
            f"{self.experiment_type}/{str(file_name).split('.')[0]}")
        )

        if not os.path.exists(path):
            os.makedirs(path)
            
        df_copy = df.copy()
        df_copy.index = df_copy.index / len(df_copy) * self.video_config["duration"]
        
        plt.figure(figsize=(10, 10))
        plt.plot(df_copy["Mean_Body_Position_1_x"].values, df_copy["Mean_Body_Position_1_y"].values, label="Target")
        plt.plot(df_copy["Mean_Body_Position_2_x"].values, df_copy["Mean_Body_Position_2_y"].values, label="Non-target")
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
    def __total_time_of_interaction(col: pd.Series, duration: float, frame_count: int) -> float:
        """
        Calculate the total time of active interaction between mice.
        :param col:
            The column indicating the proximity of the mice to each other.
        :param duration:
            The total duration of the video recording in seconds.
        :param frame_count:
            The total number of frames in the video recording.
        :return:
            Total time of active interaction in seconds.
        """
        total_time = duration * col.sum() / frame_count
        return total_time
    
    @staticmethod
    def __frequency_of_interaction(interaction_time: float, length: float) -> float:
        """
        Calculate the frequency of active interaction between mice.
        :param interaction_time:
            The total time of active interaction between mice in seconds.
        :param length:
            The duration of the video recording in seconds.
        :return:
            The frequency of active interaction in number of interactions per second.
        """
        frequency = interaction_time / length
        return frequency
    
    @staticmethod
    def __latency_to_first_interaction(col: pd.Series, duration: float, frame_count: int) -> float:
        """
        Calculate the latency before the first active interaction.
        :param col:
            The column denoting the proximity between the mice.
        :param duration:
            The duration of the video recording in seconds.
        :param frame_count:
            The total number of frames in the video recording.
        :return:
            The latency before the first active interaction in the video recording in seconds.
        """
        frames = 0
        
        for idx, val in col.items():
            if val == 0:
                frames += 1
            else:
                break
        
        latency = frames / frame_count * duration
        
        return latency


if __name__ == "__main__":
    current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    feature_engineering = SocialInteractionFeatureEngineering(
        pose_keypoints_path=str(current_path.joinpath(
            "legacy_data/07_pose_tables/Trial_1_v1_SI_w2_control_2593.mp4.predictions.h5"
        )),
        box_keypoints_path=str(current_path.joinpath(
            "legacy_data/10_borders_tables/Trial_1_v1_SI_w2_control_2593.mp4.predictions.h5"
        )),
        thresholds_config={},
        video_config=get_video_config(
            str(current_path.joinpath("legacy_data/02_converted/Trial_1_v1_SI_w2_control_2593.mp4")),
            str(current_path.joinpath(
                "legacy_data/10_borders_tables/Trial_1_v1_SI_w2_control_2593.mp4.predictions.h5"
            )),
            {},
            "SI",
            -1
        ),
        save_path=str(data_path.joinpath("data/derived_features"))
    )
    features = feature_engineering.run_feature_engineering()
