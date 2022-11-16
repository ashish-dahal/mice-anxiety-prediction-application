import os

from src.config import CORNER_DATA_FOLDER, CORNER_TABLES_FOLDER


def convert_slp_to_h5(slp_folder, h5_folder):
	slp_files = os.listdir(slp_folder)
	count = 0
	total = len(slp_files)
	
	for slp_file in slp_files:
		name, ext = os.path.splitext(slp_file)
		slp_file_path = os.path.join(slp_folder, slp_file)
		h5_file_path = os.path.join(h5_folder, f'{name}.h5')
		os.system(f"sleap-convert {slp_file_path} -o {h5_file_path} --format analysis")
		count += 1
		
		print(f"\nConverted {count} out of {total}.\n")

	print("\nFINISHED!")


if __name__ == "__main__":
	convert_slp_to_h5(CORNER_DATA_FOLDER, CORNER_TABLES_FOLDER)
