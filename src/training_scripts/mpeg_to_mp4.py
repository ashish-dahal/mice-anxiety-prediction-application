import moviepy.editor as moviepy
import os

from src.config import CONVERTED_DATA_FOLDER, RAW_DATA_FOLDER


def convert_video_files():
	new_ext = '.mp4'
	
	for file_name in os.listdir(RAW_DATA_FOLDER):
		name, old_ext = os.path.splitext(file_name)
		
		if old_ext == '.mpg':
			old_file_path = os.path.join(RAW_DATA_FOLDER, file_name)
			clip = moviepy.VideoFileClip(old_file_path)
			new_file_path = os.path.join(CONVERTED_DATA_FOLDER, name + new_ext)
			clip.write_videofile(new_file_path)


if __name__ == "__main__":
	convert_video_files()
