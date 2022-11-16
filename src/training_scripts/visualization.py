import cv2
import os

from src.config import CONVERTED_DATA_FOLDER, KEYPOINTS


def show_video(video_file_name, df, corners):
	def _draw_pose_keypoints(frame, df, idx):
		for keypoint in KEYPOINTS:
			try:
				frame = cv2.circle(frame, (int(df.loc[idx, f'{keypoint}_x']), int(df.loc[idx, f'{keypoint}_y'])), 3,
				                   (255, 0, 255), 3)
			except ValueError:
				pass
		
		return frame
	
	def _draw_corner_keypoints(frame, corners):
		for corner in corners:
			try:
				frame = cv2.circle(frame, (int(corner[0]), int(corner[1])), 3, (255, 0, 255), 3)
			except ValueError:
				pass
		
		return frame
	
	def _draw_pose_lines(frame, df, idx):
		# Draw target mouse's body line
		
		for k in range(5):
			try:
				frame = cv2.line(
					frame,
					(int(df.loc[idx, f'{KEYPOINTS[k]}_x']), int(df.loc[idx, f'{KEYPOINTS[k]}_y'])),
					(int(df.loc[idx, f'{KEYPOINTS[k + 1]}_x']), int(df.loc[idx, f'{KEYPOINTS[k + 1]}_y'])),
					(255, 0, 255),
					3
				)
			except ValueError:
				pass
		
		# Draw target mouse's lines with limbs
		for k, l in zip([1, 1, 2, 2], [6, 7, 8, 9]):
			try:
				frame = cv2.line(
					frame,
					(int(df.loc[idx, f'{KEYPOINTS[k]}_x']), int(df.loc[idx, f'{KEYPOINTS[k]}_y'])),
					(int(df.loc[idx, f'{KEYPOINTS[l]}_x']), int(df.loc[idx, f'{KEYPOINTS[l]}_y'])),
					(255, 0, 255),
					3
				)
			except ValueError:
				pass
		
		# Draw non-target mouse's lines
		for k in range(10, 14):
			try:
				frame = cv2.line(
					frame,
					(int(df.loc[idx, f'{KEYPOINTS[k]}_x']), int(df.loc[idx, f'{KEYPOINTS[k]}_y'])),
					(int(df.loc[idx, f'{KEYPOINTS[k + 1]}_x']), int(df.loc[idx, f'{KEYPOINTS[k + 1]}_y'])),
					(255, 0, 255),
					3
				)
			except ValueError:
				pass
		
		return frame
	
	cap = cv2.VideoCapture(os.path.join(CONVERTED_DATA_FOLDER, video_file_name))
	idx = 0
	
	while cap.isOpened():
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		gray = _draw_pose_keypoints(gray, df, idx)
		gray = _draw_corner_keypoints(gray, corners)
		gray = _draw_pose_lines(gray, df, idx)
		
		cv2.imshow('frame', gray)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
		idx += 1
	
	cap.release()
	cv2.destroyAllWindows()
