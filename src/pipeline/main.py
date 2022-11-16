from pathlib import Path
from PIL import Image

import os
import streamlit as st

from src.config import data_path
from src.pipeline.classification import Classification
from src.pipeline.feature_engineering import FeatureEngineering
from src.pipeline.pose_estimation import PoseEstimation


def run_pipeline(video_paths: list, experiment_type: str, thresholds_config: dict, save_path: str, use_gpu: bool,
                 num_frames: int, clean_cache: bool):
    """
    Run the whole pipeline (Pose Estimation, Feature Engineering, Classification).
    :param video_paths:
        List of paths to the video files.
    :param experiment_type:
        A string abbreviation denoting the experiment type for the video recordings.
    :param thresholds_config:
        A dictionary of threshold values for the experiment.
    :param save_path:
        Path to the folder where to save the Excel files and the graphs.
    :param use_gpu:
        Boolean value denoting whether to use GPU for pose tracking and border estimation or not.
    :param num_frames:
        Number of frames for performing the pose estimation on the video recording (for debugging purposes).
    :param clean_cache:
        Boolean value denoting whether to delete the converted video files or not.
    """
    labels = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set up containers for showing inference progress
    inference_container = st.container()
    col1, col2 = inference_container.columns(2)

    # Set up queue to save in-progress videos and finished videos
    progress_queue = video_paths.copy()
    finished_queue = []
    col1.markdown("**In Progress**")
    col2.markdown("**Finished**")

    # placeholders to display the in-progress and finished videos
    progress_view = col1.empty()
    finished_view = col2.empty()

    # placeholder to show the progress status message
    status_message = inference_container.empty()
    inference_container.markdown("""---""")

    # Container to show error message
    error_message = st.container()

    # CSS for color of the spinner
    inference_container.markdown(
        body="""<style>.stSpinner > div > div {border-top-color: #775db7;}</style>""",
        unsafe_allow_html=True
    )

    # Set up container to display results
    result_container = st.container()

    # Set flag to see if the loop is running for the first time // used to show the "Result" title just once
    is_running_first = True

    # Holds the path for the manually labeled corners file for OF/SI experiments
    box_keypoints_path = None

    # Run loop for each video // experiment type is assumed to be same
    for video_path in video_paths:
        # Show spinner:
        with st.spinner("Running Pipeline..."):
            # Show in prgoress videos
            with progress_view.container():
                for path in progress_queue:
                    st.info(os.path.basename(path))

            # Show finished videos
            with finished_view.container():
                for path in finished_queue:
                    st.info(os.path.basename(path))

            # Initialize error flag
            error = False

            # Run pipeline
            try:
                with status_message.container():
                    st.markdown(f"**Current Video:** {video_path}")

                pose_estimation = PoseEstimation(
                    video_path,
                    experiment_type,
                    thresholds_config,
                    use_gpu,
                    num_frames,
                    clean_cache,
                    box_keypoints_path
                )
                pose_keypoints_path, box_keypoints_path, video_config = pose_estimation.run_pose_estimation(status_message)

                with status_message.container():
                    st.markdown(f"**Current Video:**: {video_path}")
                    st.warning("Status: Running feature engineering...")

                feature_engineering = FeatureEngineering(
                    pose_keypoints_path,
                    box_keypoints_path,
                    experiment_type,
                    thresholds_config,
                    video_config,
                    save_path
                )
                features_df, graphs_path = feature_engineering.run_feature_engineering()

                with status_message.container():
                    st.markdown(f"**Current Video:**: {video_path}")
                    st.warning("Status: Running classification...")

                classification = Classification(features_df, experiment_type)
                anxiety_label = classification.run_classification()
                labels.append(anxiety_label)
            except Exception as e:
                # Catch error
                error = True
                error_message.error(f"{e}\n\nVideo: {video_path}. ")

            # Show error message
            if error:
                progress_queue.pop(0)
            # Add to finished queue if no error
            else:
                finished_queue.append(progress_queue.pop(0))

                # Show results:
                if is_running_first:
                    result_container.markdown('#### Results')  # runs only once
                    
                is_running_first = False

                # Display expander
                result_expander = result_container.expander(video_path)

                # Display anxiety label
                if experiment_type in ["OF", "EPM"]:
                    result_expander.markdown(f'**Anxiety: {anxiety_label}**')

                # Show figures from the result folder
                images_path = []

                for graph_file_name in os.listdir(graphs_path):
                    if graph_file_name.endswith(".png"):
                        images_path.append(str(Path(graphs_path).joinpath(graph_file_name)))
                
                result_expander.markdown(f'**Figures**')
                
                for image_path in images_path:
                    image = Image.open(image_path)
                    result_expander.image(image)

            # Show finished message if no video in progress
            if not progress_queue:
                progress_view.info("Finished")
            else:
                # Clear the progress and finished video queue to show new updated queue
                progress_view.empty()
                finished_view.empty()

    # Display message if the finished queue is empty
    if not finished_queue:
        finished_view.info("Couldn't finish inference on the videos.")
        status_message.empty()
        return False
    else:
        status_message.success(f'Status: Finished running pipeline. Results saved at: {save_path}')
        return True


if __name__ == "__main__":
    video_paths = [
        # str(data_path.joinpath("temp/videos/OF 4w/Control/Trial     2_OF_4w_Control.mpg")),
        # str(data_path.joinpath("temp/videos/OF 4w/Treated/Trial     1_OF_4w_Treated.mpg")),
        # str(data_path.joinpath("temp/videos/OF 6w/Control/Trial     2_OF_6w_Control.mpg")),
        # str(data_path.joinpath("temp/videos/OF 6w/Treated/Trial     1_OF_6w_Treated.mpg")),
        str(data_path.joinpath("videos/Trial     1_YM_6w_Treated.mpg")),
        # str(data_path.joinpath("temp/videos/EPM 6w/Treated/Trial     1_EPM_6w_Treated.mpg")),
    ]
    experiment_type = "YM"
    save_path = str(data_path.joinpath("data/derived_features"))
    clean_cache = True
    use_gpu = False
    num_frames = -1

    run_pipeline(
        video_paths=video_paths,
        experiment_type=experiment_type,
        thresholds_config={},
        save_path=save_path,
        use_gpu=use_gpu,
        num_frames=num_frames,
        clean_cache=clean_cache
    )
