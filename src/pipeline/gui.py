from tkinter import filedialog

import streamlit as st
import tkinter as tk

from src.pipeline.main import run_pipeline


def show_config():
    """
    Showing the main configurations on the GUI.
    """
    # Set Title
    st.title("Welcome to BRATZ app!")
    st.markdown("##### An app to analyse mice experiment videos.")
    st.markdown("""---""")

    # Dictionary to map experiment types with abbreviations
    experiment_dict = {
        "Open Field Experiment": "OF",
        "Social Interaction Experiment": "SI",
        "Elevated Plus Maze Experiment": "EPM",
        "Y-Maze Experiment": "YM"
    }

    # Selectbox to choose experiment type
    experiment_type = st.selectbox(
        "Choose the Experiment Type",
        ("Open Field Experiment", "Social Interaction Experiment", "Elevated Plus Maze Experiment", "Y-Maze Experiment")
    )

    # Map experiment type with abbreviation
    experiment_type = experiment_dict[experiment_type]

    # Choose experiment configs
    experiment_configs = st.radio("Choose Configurations for the Experiment Videos", ("Default", "Custom"))

    # Initialize configurations/thresholds for experiments
    experiment_configs_dict = {}

    # Let user select configurations when set to custom
    if experiment_configs == "Custom":
        if experiment_type in ["OF", "SI"]:
            arena_length = st.number_input("Arena Length in cm (default is 45)", 0.00, 100.00)
            experiment_configs_dict['arena_length'] = arena_length
            
        if experiment_type == "OF":
            velocity_threshold = st.number_input("Velocity Threshold in cm/s (default is 53)", 0.00, 100.00)
            experiment_configs_dict['velocity_threshold'] = velocity_threshold

        if experiment_type == "SI":
            proximity = st.number_input("Proximity in cm (default is 5)", 0.00, 100.00)
            experiment_configs_dict['proximity'] = proximity
            
        if experiment_type == "EPM":
            arena_length = st.number_input("Arm Length in cm (default is 35)", 0.00, 100.00)
            experiment_configs_dict['arena_length'] = arena_length
            
        if experiment_type == "YM":
            arena_length = st.number_input("Arm Length in cm (default is 30)", 0.00, 100.00)
            experiment_configs_dict['arena_length'] = arena_length

        bin_length = st.number_input("Bin length in seconds (default is 60)", 0.00, 100.00)
        experiment_configs_dict['bin_length'] = bin_length
            
    # Setup 2 column layout
    col1, col2 = st.columns(2)

    # Let user choose whether to use CPU or GPU for pose inference
    use_gpu = True if col1.radio(
        "Run pose inference on",
        ("CPU", "GPU"),
        help="Choose either to run pose inference on CPU or GPU"
    ) == "GPU" else False

    # Ask user whether to clean cache files after running the pipeline
    clean_cache = col2.checkbox(
        "Clean cache", help="Clean temporary files after the inference")

    # Accepted video types
    VIDEO_TYPES = ['mp4', 'mpg', 'mpeg', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'ogv']

    # Choose experiment videos
    root = tk.Tk()
    root.withdraw()
    video_paths = False
    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)
    # Folder picker button
    st.markdown('**Choose video Files**')
    clicked = st.button('Browse', key='video_upload')
    
    if clicked:
        video_paths = filedialog.askopenfilenames(
            master=root,
            filetypes=[("Video files", " ".join("." + x for x in VIDEO_TYPES))]
        )
        st.session_state['video_paths'] = video_paths

    # Retrieve uploaded files from session state if available
    if 'video_paths' in st.session_state:
        video_paths = st.session_state['video_paths']

    # Show video queue
    if video_paths:
        st.markdown("**Video Queue** (Note: Only video files with MP4 format can be shown.)")
        # Show expander that shows video file and video
        for video_path in video_paths:
            video_expander = st.expander(video_path)
            video_expander.video(video_path)

    # Button to choose output directory
    # Set up tkinter
    root = tk.Tk()
    root.withdraw()

    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)

    save_path = ""

    # Folder picker button
    st.markdown('**Select a folder to save the output**')
    clicked = st.button('Browse', key="output_folder")
    
    if clicked:
        save_path = st.text_input('Output folder', filedialog.askdirectory(master=root), key=1)
        st.session_state['save_path'] = save_path
    # Retrieve output directory from session state if available
    elif 'save_path' in st.session_state:
        if st.session_state['save_path'] != "":
            save_path = st.text_input('Output folder', value=st.session_state['save_path'], key=1)
            
    st.markdown("""---""")
    
    # Submit button
    submit = st.button("Start")
    st.markdown("""---""")
    
    if submit:
        # check if video files are selected
        if not video_paths:
            st.error('Please choose at least one experiment video.')
        # check if output path is selected
        elif not save_path or save_path.strip() == "":
            st.error('Please choose a directory to save output files.')
        else:
            # Run the pipeline
            run_pipeline(
                video_paths=list(video_paths),
                experiment_type=experiment_type,
                thresholds_config=experiment_configs_dict,
                save_path=save_path,
                use_gpu=use_gpu,
                num_frames=-1,
                clean_cache=clean_cache
            )


if __name__ == '__main__':
    # Set web app information
    st.set_page_config(
        layout="centered",
        page_icon="",
        page_title="Brats - Brainy Rats"
    )
    show_config()
