from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import ntpath
import numpy as np
import os
import pandas as pd
import random
import warnings

from src.config import config

warnings.filterwarnings("ignore")


def click_event(event, x: int, y: int, flags, params: dict):
    """
    Label the coordinates of corners of the arena.
    :param event:
        Action performed by the mouse.
    :param x:
        X-coordiante of the mouse cursor's position.
    :param y:
        Y-coordinate of the mouse cursor's position.
    :param flags:
        The default parameter for the function.
    :param params:
        List of parameters that includes the video frame, number of corners and list of labeled coordinates.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(params["coords"]) < params["num_corners"]:
            params["coords"].append((x, y))
            cv2.circle(params["image"], (x, y), radius=3, color=(0, 0, 255), thickness=-1)
            
            image_copy = params["image"].copy()

            if len(params["coords"]) < params["num_corners"]:
                if params["experiment_type"] in ("OF", "SI"):
                    cv2.putText(
                        image_copy, "Click on the next corner of the arena in clockwise direction.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                    )
                elif params["experiment_type"] == "EPM":
                    if len(params["coords"]) in (1, 3):
                        cv2.putText(
                            image_copy, "Click on the right corner of the same closed arm of the maze.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                        )
                    elif len(params["coords"]) == 2:
                        cv2.putText(
                            image_copy, "Click on the left corner of the next closed arm of the maze.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                        )
                    elif len(params["coords"]) == 3:
                        cv2.putText(
                            image_copy,
                            "Click on the inner corner adjacent to the right corner of the first closed arm.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                        )
                    else:
                        cv2.putText(
                            image_copy,
                            "Click on the next corner of the maze (in clockwise direction).", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                        )
                elif params["experiment_type"] == "YM":
                    if len(params["coords"]) % 2 == 1 and len(params["coords"]) < 6:
                        cv2.putText(
                            image_copy, "Click on the right corner of the current arm of the maze.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                        )
                    elif len(params["coords"]) % 2 == 0 and len(params["coords"]) < 6:
                        cv2.putText(
                            image_copy, "Click on the left corner of next arm of the maze (in clockwise direction).",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                        )
                    elif len(params["coords"]) == 6:
                        cv2.putText(
                            image_copy, "Click on the inner corner adjacent to the first labeled corner of the maze.",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                        )
                    else:
                        cv2.putText(
                            image_copy,
                            "Click on the next inner corner of the maze (in clockwise direction).", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                        )

            cv2.imshow("Labeling frame", image_copy)
        else:
            cv2.imshow("Labeling frame", params["image"])


def label_corners_manually(experiment_type: str, video_path: str, output_path: Path) -> str:
    """
    Label the corners of the maze manually.
    :param experiment_type:
        Type of the experiment to identify the instructions and a number of corners.
    :param video_path:
        Path to the video file to extract frames for manual labeling.
    :param output_path:
        Resultant folder for the file that will have manually labeled coordinates of the corners of the arena.
    :return:
        Resultant path of the output file.
    """
    num_corners = 0
    
    if experiment_type in ("OF", "SI"):
        num_corners = 4
    elif experiment_type == "EPM":
        num_corners = 12
    elif experiment_type == "YM":
        num_corners = 9
        
    vidcap = cv2.VideoCapture(video_path)
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    while True:
        random_frame_number = random.randint(0, total_frames)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
        success, image = vidcap.read()
    
        params = {
            "experiment_type": experiment_type,
            "num_corners": num_corners,
            "coords": [],
            "image": image
        }

        cv2.putText(image, "If corners are not visible, press 'c' to change frame.",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Press 'e' when all corners are labeled.",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        image_copy = image.copy()

        if experiment_type in ("OF", "SI"):
            cv2.putText(image_copy, "Click on the upper left corner of the arena.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif experiment_type == "EPM":
            cv2.putText(image_copy, "Click on left corner of the closed arm of the maze.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif experiment_type == "YM":
            cv2.putText(image_copy, "Click on left corner of the first arm of the maze.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Labeling frame", image_copy)
        cv2.setMouseCallback("Labeling frame", click_event, params)
        cv2.waitKey(0)
        
        if len(params["coords"]) == params["num_corners"]:
            cv2.destroyAllWindows()
            break
            
    result_file = output_path / f"{ntpath.basename(video_path)}.txt"
    
    print("[Pose Estimation] Corner coordinates are saved in the following file:", result_file)
    
    with open(result_file, 'w') as f:
        f.write(str(params["num_corners"]) + '\n')
        
        for tup in params["coords"]:
            x, y = tup
            f.write(str(x) + ',' + str(y) + '\n')
        
    return str(result_file)


def calculate_units_per_cm(borders_keypoints_path: str, arena_length: int, experiment_type: str) -> float:
    """
    Get the units per centimeter for a video recording.
    :param borders_keypoints_path:
        Path to the H5 file with borders keypoints.
    :param arena_length:
        Original length of the arena in centimeters.
    :param experiment_type:
        A string of an abbreviation of an experiment type. Each experiment has different shape of the arena.
    :return:
        Float value indicating the number of units per 1 centimeter (for calculations).
    """
    if os.path.splitext(borders_keypoints_path)[1] == ".h5":
        border_tables = h5py.File(borders_keypoints_path, 'r')
        border_x, border_y = border_tables['tracks'][0]
        border_x, border_y = border_x.transpose(), border_y.transpose()
    
        best_score_idx = np.argmax(border_tables["instance_scores"][0])
        border_x_mean, border_y_mean = border_x[best_score_idx], border_y[best_score_idx]
    else:
        border_x_mean, border_y_mean = [], []
        
        with open(borders_keypoints_path, 'r') as f:
            num_corners = int(f.readline())
            
            for i in range(num_corners):
                x, y = list(map(int, f.readline().split(',')))
                border_x_mean.append(x)
                border_y_mean.append(y)
    
    point1, point2 = None, None
    
    if experiment_type in ["SI", "OF"]:
        points = list(map(list, zip(*[border_x_mean, border_y_mean])))
        distances = [np.linalg.norm(np.array(points[0]) - np.array(points[i])) for i in range(1, len(points))]
        diag_point = np.argmax(distances) + 1
        
        point1 = points[0]
        
        for i in range(1, len(points)):
            if i != diag_point:
                point2 = points[i]
                break
    elif experiment_type == "EPM":
        points = list(map(list, zip(*[border_x_mean, border_y_mean])))
        distances = [np.linalg.norm(np.array(points[0]) - np.array(points[i])) for i in range(1, len(points))]
        diag_point = np.argmin(distances) + 1
        del points[diag_point]
        distances = [np.linalg.norm(np.array(points[0]) - np.array(points[i])) for i in range(1, len(points))]
        diag_point = np.argmin(distances) + 1
        
        point1 = points[0]
        point2 = points[diag_point]
    elif experiment_type == "YM":
        points = list(map(list, zip(*[border_x_mean, border_y_mean])))
        distances = [np.linalg.norm(np.array(points[0]) - np.array(points[i])) for i in range(1, len(points))]
        diag_point = np.argmin(distances) + 1
        del points[diag_point]
        distances = [np.linalg.norm(np.array(points[0]) - np.array(points[i])) for i in range(1, len(points))]
        diag_point = np.argmin(distances) + 1
    
        point1 = points[0]
        point2 = points[diag_point]

    units_per_cm = np.linalg.norm(np.array(point1) - np.array(point2)) / arena_length
    return units_per_cm


def get_video_config(video_path: str, borders_keypoints_path: str, thresholds_config: dict, experiment_type: str,
                     num_frames: int) -> dict:
    """
    Get the video recording configurations such as FPS, duration, etc.
    :param video_path:
        Path to the video recording.
    :param borders_keypoints_path:
        Path to the file with borders keypoints.
    :param thresholds_config:
        A dictionary of threshold values for the experiment.
    :param experiment_type:
        A string abbreviation of the type of the experiment in a video recording.
    :param num_frames:
        A number of frames to consider from the start of the video recording.
    :return:
        Dictionary with video recording configurations.
    """
    if len(thresholds_config) > 0:
        arena_length = thresholds_config["arena_length"]
    else:
        arena_length = config["experiments"][experiment_type]["arena_length"]

    units_per_cm = calculate_units_per_cm(borders_keypoints_path, arena_length, experiment_type)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if num_frames == -1:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        frame_count = num_frames
        
    if len(thresholds_config) > 0:
        bin_length = thresholds_config["bin_length"]
    else:
        bin_length = config["pipeline"]["pose_estimation"]["bin_length"]
    
    interval = int(bin_length * fps)
    duration_in_seconds = frame_count / fps
    height = int(cap.get(4))
    
    video_config = {
        'bin_length': bin_length,
        'duration': duration_in_seconds,
        'fps': fps,
        'frame_count': frame_count,
        'interval': interval,
        'units_per_cm': units_per_cm,
        'height': height
    }
    
    return video_config


def create_barplot(df: pd.DataFrame, columns: list, color: str, title: str, xlabel: str, ylabel: str, labels: list,
                   save_path: str, experiment_type: str, pose_keypoints_path: str):
    """
    Create a bar-plot.
    :param df:
        Pandas DataFrame with aggregate features.
    :param columns:
        Names of columns to target for the bar-plot.
    :param color:
        Color of the bar-plot.
    :param title:
        Title of the bar-plot.
    :param xlabel:
        Name of the x-axis label.
    :param ylabel:
        Name of the y-axis label.
    :param labels:
        Labels of ticks on x-axis.
    :param save_path:
        A path to the folder where to save graphs and features.
    :param experiment_type:
        A string abbreviation denoting the experiment type.
    :param pose_keypoints_path:
        Path to the pose keypoints file.
    """
    file_name = Path(pose_keypoints_path).name
    path = str(Path(save_path).joinpath(f"{experiment_type}/{str(file_name).split('.')[0]}"))
    
    if not os.path.exists(str(path)):
        os.makedirs(str(path))
    
    y = df[columns].values[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(columns, y, color=color)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticklabels(labels)
    plt.grid(axis="y")
    plt.savefig(str(Path(path).joinpath(f"{title}.png")))
    plt.show()


def plot_timeline(df: pd.DataFrame, columns: list, colors: list, labels: list, ylabel: str, title: str, save_path: str,
                  experiment_type: str, pose_keypoints_path: str):
    """
    Plot a timeline.
    :param df:
        Pandas DataFrame with time-series features.
    :param columns:
        List of columns to consider from the DataFrame.
    :param colors:
        List of colors to draw the timeline.
    :param labels:
        List of labels to name the y-ticks of the timeline.
    :param ylabel:
        Name of the y-label.
    :param title:
        Title of the timeline.
    :param save_path:
        A path to the folder where to save graphs and features.
    :param experiment_type:
        A string abbreviation denoting the experiment type.
    :param pose_keypoints_path:
        Path to the pose keypoints file.
    """
    file_name = Path(pose_keypoints_path).name
    path = str(Path(save_path).joinpath(f"{experiment_type}/{str(file_name).split('.')[0]}"))
    
    if not os.path.exists(str(path)):
        os.makedirs(str(path))
    
    df_copy = df.copy()
    df_copy.set_index("Timestamp", inplace=True)
    data = {k: [] for k in columns}
    current_val = ''
    start = 0
    
    for idx, row in df_copy.iterrows():
        if current_val == '':
            for key in columns:
                if row[key]:
                    current_val = key
                    break
                    
            if current_val == '':
                current_val = "Other"
                
        change = False
        
        for key in columns:
            if row[key]:
                change = True
                
                if current_val != key:
                    finish = float(idx)
                    
                    if current_val not in data:
                        data[current_val] = []
                        
                    data[current_val].append((start, finish - start))
                    start = float(idx)
                    current_val = key
                    break
                
        if not change:
            if current_val != "Other":
                finish = float(idx)
                data[current_val].append((start, finish - start))
                start = float(idx)
                current_val = "Other"
    
    data[current_val].append((start, float(df_copy.index[-1]) - start))
    
    if len(data.keys()) > len(columns):
        colors += ["tab:grey"]
        labels += ["Other"]
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    for key, color, pos in zip(data.keys(), colors, range(10, 10*len(data.keys())+1, 10)):
        ax.broken_barh(data[key], (pos, 9), facecolors=color)
        
    ax.set_yticks([15+10*i for i in range(len(data.keys()))])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=20)
    plt.savefig(str(Path(path).joinpath(f"{title}.png")))
    plt.show()


def read_pose_keypoints(path: str, video_config: dict) -> pd.DataFrame:
    """
    Read the file containing pose keypoints.
    :param path:
        Path to the H5 formatted file of pose keypoints.
    :param video_config:
        Dictionary of video configurations.
    :return:
        Pandas DataFrame representation of the pose keypoints.
    """
    print("[Feature Engineering] Reading pose keypoints...")
    pose_tables = h5py.File(path, 'r')
    pose_node_names = [str(x)[2:-1] for x in pose_tables['node_names'][:]]
    pose_x, pose_y = pose_tables['tracks'][0]
    pose_x, pose_y = pose_x.transpose(), pose_y.transpose()
    pose_y = video_config["height"] - pose_y 
    
    columns = []
    
    for node_name in pose_node_names:
        columns.append(f'{node_name}_x')
        columns.append(f'{node_name}_y')
    
    df = pd.DataFrame(pose_x, columns=[i + '_x' for i in pose_node_names]).join(
        pd.DataFrame(pose_y, columns=[i + '_y' for i in pose_node_names])
    )
    
    df = df.loc[:video_config["frame_count"], columns]
    return df


def save_features(pose_keypoints_path: str, save_path: str, experiment_type: str, time_series_features: pd.DataFrame,
                  aggregate_features: list):
    """
    Save the time series and aggregate features for a single video recording in an Excel file in separate sheets.
    :param pose_keypoints_path:
        Path to the file with predicted keypoints for the pose.
    :param save_path:
        A path to the folder where to save features.
    :param experiment_type:
        A string abbreviation denoting the experiment type of a video recording.
    :param time_series_features:
        Derived time-series features for a single video recording.
    :param aggregate_features:
        Derived aggregate features for a single video recording with bins.
    """
    file_name = Path(pose_keypoints_path).name
    features_folder = Path(save_path).joinpath(f"{experiment_type}/{str(file_name).split('.')[0]}")
    
    if not os.path.exists(str(features_folder)):
        os.makedirs(str(features_folder))
        
    excel_path = str(features_folder.joinpath(str(file_name).split(".")[0] + ".xlsx"))
    writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")
    time_series_features.to_excel(writer, sheet_name="Time-series Features", index=False, engine="xlsxwriter")
    
    for i in range(len(aggregate_features)):
        if i == 0:
            aggregate_features[i].to_excel(
                writer,
                sheet_name="Aggregate Features",
                index=False,
                engine="xlsxwriter"
            )
        else:
            aggregate_features[i].to_excel(
                writer,
                sheet_name="Aggregate Features (Bin {0})".format(i),
                index=False,
                engine="xlsxwriter"
            )

    workbook = writer.book
    format = workbook.add_format()
    format.set_align('center')
    format.set_align('vcenter')
    
    for i in range(len(time_series_features.columns)):
        col_idx = chr(ord('A') + i)
        writer.sheets["Time-series Features"].set_column(f"{col_idx}:{col_idx}", 17, format)
        
    for i in range(len(aggregate_features)):
        for j in range(len(aggregate_features[i].columns)):
            col_idx = chr(ord('A') + j)
            
            if i == 0:
                writer.sheets["Aggregate Features"].set_column(f"{col_idx}:{col_idx}", 35, format)
            else:
                writer.sheets["Aggregate Features (Bin {0})".format(i)].set_column(f"{col_idx}:{col_idx}", 35, format)
    
    writer.save()
