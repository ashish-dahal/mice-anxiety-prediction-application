from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import Union

import cv2
import ntpath
import os
import streamlit as st
import subprocess
import uuid

from src.config import config, data_path
from src.pipeline.utils import get_video_config, label_corners_manually


class PoseEstimation:
    """
    A class that performs the pose tracking and arena corners estimation for video recordings.
    """
    VIDEO_PATH = ""
    POSE_MODEL_PATH = ""
    POSE_SLP_PATH = ""
    POSE_H5_PATH = ""
    BORDER_MODEL_PATH = ""
    BORDER_SLP_PATH = ""
    BORDER_H5_PATH = ""
    EXPERIMENT_TYPE = ""
    use_gpu = ""

    __VIDEO_CONFIG = {}

    def __init__(self, video_path: str, experiment_type: str, thresholds_config: dict, use_gpu=False, num_frames=-1,
                 clean_cache=False, box_keypoints_path=None):
        """
        Initialize the variables.
        :param video_path:
            A path to the video recording.
        :param experiment_type:
            A string abbreviation denoting the experiment types of video recordings.
        :param thresholds_config:
            A dictionary of threshold values for the experiment.
        :param use_gpu:
            A boolean flag denoting whether to use the GPU or not during pose tracking and arena corners estimation.
        :param num_frames:
            A number of frames from the video to consider.
        :param clean_cache:
            Boolean value indicating whether to delete converted video recordings or not.
        :param box_keypoints_path:
            Path to the file with manually labeled corners for the OF and SI experiments.
        """
        # Set video path
        self.VIDEO_PATH = video_path
        experiment_types = ["OF", "EPM", "SI", "YM"]

        # Check if the experiment type is valid
        if experiment_type.upper() not in experiment_types:
            raise Exception("[Pose Estimation] Warning! Invalid experiment type provided.")
        else:
            # Set experiment type
            self.EXPERIMENT_TYPE = experiment_type
            
        h5_files_folder  = data_path.joinpath(config["pipeline"]["pose_estimation"]["h5_files_folder"])
        models_folder    = data_path.joinpath(config["pipeline"]["pose_estimation"]["models_folder"])
        slp_files_folder = data_path.joinpath(config["pipeline"]["pose_estimation"]["slp_files_folder"])
        
        self.thresholds_config = thresholds_config

        for exp_type in experiment_types:
            paths = [f"{exp_type}/pose/", f"{exp_type}/border/"]
            
            for path in paths:
                exp_path = slp_files_folder.joinpath(path)
    
                if not os.path.exists(exp_path):
                    os.makedirs(exp_path)
    
                exp_path = h5_files_folder.joinpath(path)
    
                if not os.path.exists(exp_path):
                    os.makedirs(exp_path)

        # Define paths to model, slp file and h5 file based on the experiment type
        self.POSE_MODEL_PATH = models_folder.joinpath(f"{experiment_type}/pose/")
        self.POSE_SLP_PATH = slp_files_folder.joinpath(f"{experiment_type}/pose/")
        self.POSE_H5_PATH = h5_files_folder.joinpath(f"{experiment_type}/pose/")
        self.BORDER_MODEL_PATH = models_folder.joinpath(f"{experiment_type}/border/")
        self.BORDER_SLP_PATH = slp_files_folder.joinpath(f"{experiment_type}/border/")
        self.BORDER_H5_PATH = h5_files_folder.joinpath(f"{experiment_type}/border/")

        # Set GPU usage flag
        self.use_gpu = use_gpu
        
        self.clean_cache = clean_cache
        self.num_frames = num_frames
        self.box_keypoints_path = box_keypoints_path

    def __convert_to_mp4(self) -> Union[None, str]:
        """
        Convert popular video formats to mp4 format. Supported formats: 'mpg', 'mpeg', 'avi', 'mov', 'mkv', 'webm',
        'flv', 'wmv', 'ogv'.
        :param video_path (str):
            Path to a video file
        :param output_path (str, optional):
            Path where the converted video should be saved. Defaults to the path to current directory
        :return:
            int: -1 if video could not be converted
            int: 0 if video is already in MP4 format
            str: MP4 file path if video converted successfully
        """
        print("[Pose Estimation] Converting the video recording to MP4 format...")
        VALID_VIDEO_FORMATS = ('mpg', 'mpeg', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'ogv')

        # Initialize moviepy VideoFileClip
        try:
            clip = VideoFileClip(self.VIDEO_PATH)
        except OSError:
            raise OSError(f"[Pose Estimation] Warning! Unable to convert the video: {self.VIDEO_PATH}")

        # Extract name and extension
        _, old_ext = os.path.splitext(self.VIDEO_PATH)
        file_name = Path(self.VIDEO_PATH).name

        # Check if video format is supported
        if old_ext[1:].lower() in VALID_VIDEO_FORMATS:
            output_folder = data_path.joinpath(config["pipeline"]["pose_estimation"]["converted_videos_folder"])

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Save video as mp4 with unique filename
            unique_id = str(uuid.uuid4())
            output_path = str(output_folder.joinpath(f"{str(file_name) + '_' + old_ext[1:] + unique_id}.mp4"))
            clip.write_videofile(output_path)
            self.VIDEO_PATH = output_path
            print(f"[Pose Estimation] Conversion of the video to MP4 format is finished successfully!")
            return output_path
        else:
            print(f'[Pose Estimation] Note. Video not converted. The video is already in MP4 Format: {self.VIDEO_PATH}')
            return None
        
    def __cut_video(self) -> str:
        """
        Gets the small grayscaled portion from of the video for border estimation.
        :return:
            Path to the resultant short grayscale video file.
        """
        print("[Pose Estimation] Cut the video for further border estimation...")
        source = cv2.VideoCapture(self.VIDEO_PATH)
        width, height = int(source.get(3)), int(source.get(4))
        size = (width, height)

        _, old_ext = os.path.splitext(self.VIDEO_PATH)
        file_name = Path(self.VIDEO_PATH).name

        output_folder = data_path.joinpath(config["pipeline"]["pose_estimation"]["converted_videos_folder"])

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_path = str(output_folder.joinpath(f"{str(file_name)}_grayscale_short{old_ext}"))
        fps = source.get(cv2.CAP_PROP_FPS)
        result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size, 0)
        frames = 0

        while True:
            ret, img = source.read()
    
            if not ret:
                break

            frames += 1
            
            if frames % 100 != 0:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result.write(gray)

        source.release()
        print("[Pose Estimation] The video was cropped successfully!")
        return output_path

    def __track_keypoint(self, model_path: Path, output_path: Path) -> Union[str, None]:
        """
        Track video keypoints using SLEAP software.
        :param model_path (str):
            Path to the tracking model
        :param output_path (str):
            Path to where the output should be saved
        :return:
            str: Path to the .slp file if keypoint tracking is successful
            None: None if keypoint tracking is unsuccessful
        """
        # Get the filename
        file = ntpath.basename(self.VIDEO_PATH)

        # Set if inference will be done is CPU only mode or with GPU based on the use_gpu flag
        gpu_flag = "" if self.use_gpu else "--cpu"

        # Set output path and filename
        output_file = str(output_path.joinpath(f"{file}.slp"))

        # Prepare keypoint tracking command string (--frames 0-10)
        if self.num_frames == -1:
            string = f'sleap-track \"{self.VIDEO_PATH}\" --tracking.tracker simple -m \"{str(model_path)}\" ' \
                     f'-o \"{output_file}\" {gpu_flag} '
        else:
            string = f'sleap-track \"{self.VIDEO_PATH}\" --tracking.tracker simple --frames 0-{self.num_frames} ' \
                     f'-m \"{str(model_path)}\" -o \"{output_file}\" {gpu_flag} '
        # Run keypoint tracking
        result = subprocess.getoutput(string)
        print(string)
        print("result::: ", result)
        
        if "Aborted (core dumped)" in result:
            raise Exception("[Pose Estimation] Error! Not enough GPU/RAM memory for automatic corner/pose detection!")

        # Return the path to the .slp file
        return output_file

    @staticmethod
    def __slp_to_h5(file_path: str, output_path: Path) -> Union[bool, str]:
        """
        Convert .slp file to .h5 file.
        :param file_path (str):
            Path to the .slp file
        :param output_path (str):
            Path to where the .h5 file should be saved
        :return:
            str: Path to .h5 file if the conversion is successful
            bool: False if conversion is unsuccessful
        """
        # Get the filename
        file = ntpath.basename(file_path)

        # Set output path and filename
        output_file = str(output_path.joinpath(f"{file}.h5"))

        # Prepare conversion command string
        string = f'sleap-convert \"{file_path}\" -o \"{output_file}\" --format analysis'

        # Convert .slp to .h5 format
        result = subprocess.getoutput(string)
        print(string)
        print("result::: ", result)

        # Return the path to the .h5 file
        return output_file
    
    def __clean_cache(self, paths: list):
        """
        Delete videos in case the clean cache flag is set to True.
        :param paths:
            List of paths to video files that need to be deleted.
        """
        if not self.clean_cache:
            return
        
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
            
        print("[Pose Estimation] Cache is successfully cleared!")

    def run_pose_estimation(self, status_message=None) -> Union[list, None]:
        """
        Run pose tracking on a video. Steps: Video Conversion, Pose Tracking, Convert SLP to H5.
        :param status_message:
            GUI container for showing progress information.
        :return:
            None: If error occurs during the pipeline steps
            List: [pose_h5, border_h5] H5 files for pose and border tracking if pipeline executes successfully
        """
        # 1. Convert the video to mp4
        if status_message:
            with status_message.container():
                st.warning("Status: [Pose Estimation] Converting the video recording to MP4 format...")
                
        mp4_path = self.__convert_to_mp4()
        
        # 2. Create small video for border estimation
        if status_message:
            with status_message.container():
                st.warning("Status: [Pose Estimation] Cutting the video for further border estimation...")
                
        short_grayscale_video_path = None
                
        if self.EXPERIMENT_TYPE in ("EPM", "YM"):
            short_grayscale_video_path = self.__cut_video()
            original_video_path = self.VIDEO_PATH
            self.VIDEO_PATH = short_grayscale_video_path

            # 3. Run border estimation on the video
            print("[Pose Estimation] Performing the border estimation...")
            
            if status_message:
                with status_message.container():
                    st.warning("Status: [Pose Estimation] Performing the automatic corner detection...")
                    
            try:
                border_slp = self.__track_keypoint(self.BORDER_MODEL_PATH, self.BORDER_SLP_PATH)
                border_h5 = self.__slp_to_h5(border_slp, self.BORDER_H5_PATH)
            except:
                border_h5 = label_corners_manually(self.EXPERIMENT_TYPE, short_grayscale_video_path, self.BORDER_SLP_PATH)
                
            self.VIDEO_PATH = original_video_path
        else:
            if self.box_keypoints_path is not None:
                border_h5 = self.box_keypoints_path
            else:
                border_h5 = label_corners_manually(self.EXPERIMENT_TYPE, self.VIDEO_PATH, self.BORDER_SLP_PATH)

        # 4. Run pose tracking on the video
        print("[Pose Estimation] Performing the pose tracking...")

        if status_message:
            with status_message.container():
                st.warning("Status: [Pose Estimation] Performing the automatic pose tracking...")

        try:
            pose_slp = self.__track_keypoint(self.POSE_MODEL_PATH, self.POSE_SLP_PATH)
        except Exception as e:
            raise Exception(e)

        # 5. Convert pose slp file to h5
        print("[Pose Estimation] Converting pose .slp to .h5 file...")

        if status_message:
            with status_message.container():
                st.warning("Status: [Pose Estimation] Converting pose .slp to .h5 file...")

        try:
            pose_h5 = self.__slp_to_h5(pose_slp, self.POSE_H5_PATH)
        except:
            raise Exception("[Pose Estimation] Error! The conversion of pose .SLP file to .H5 file has failed!")

        print("Done!")
        
        ##################################################################################################################
        # with h5py.File(pose_h5, "r") as f:
        #     instance_scores = list(f['instance_scores'])
        #     average_score = np.mean(instance_scores)
        #
        #     if average_score < config["pipeline"]["pose_estimation"]["video_threshold"]:
        #         print("[Pose Estimation] Error! The prediction accuracy for mice is very low. "
        #               "The video might not be of correct experiment type.")
        #         return None
        ##################################################################################################################

        # 6. Set video configurations
        print("[Pose Estimation] Getting video configurations...")

        if status_message:
            with status_message.container():
                st.warning("Status: [Pose Estimation] Getting video configurations...")

        self.__VIDEO_CONFIG = get_video_config(self.VIDEO_PATH, border_h5, self.thresholds_config, self.EXPERIMENT_TYPE,
                                               self.num_frames)
        print("Done!")

        # 7. Cleaning cache
        cache_paths = []
        
        if short_grayscale_video_path:
            cache_paths.append(short_grayscale_video_path)

        if mp4_path:
            cache_paths.append(mp4_path)

        self.__clean_cache(cache_paths)

        # Return h5 files for pose and border and video configurations
        return [pose_h5, border_h5, self.__VIDEO_CONFIG]
