import pandas as pd

from src.pipeline.epm_feature_engineering import ElevatedPlusMazeFeatureEngineering
from src.pipeline.of_feature_engineering import OpenFieldExperimentFeatureEngineering
from src.pipeline.si_feature_engineering import SocialInteractionFeatureEngineering
from src.pipeline.ym_feature_engineering import YMazeExperimentFeatureEngineering


class FeatureEngineering:
	"""
	A class that derives time-series and aggregate features for all of the video recordings with specified experiment
	types.
	"""
	def __init__(self, pose_keypoints_path: str, box_keypoints_path: str, experiment_type: str, thresholds_config: dict,
	             video_config: dict, save_path: str):
		"""
		Initialize the variables.
		:param pose_keypoints_path:
			A path to the pose keypoints file for a video recording.
		:param box_keypoints_path:
			A path to the box keypoints file for a video recording.
		:param experiment_type:
			A string abbreviation denoting the experiment type of a video recording.
		:param thresholds_config:
            A dictionary of threshold values for the experiment.
		:param video_config:
			A dictionary containing video recording specifications such as FPS, duration, etc.
		:param save_path:
			A path to the folder where to save graphs and features.
		"""
		self.pose_keypoints_path = pose_keypoints_path
		self.box_keypoints_path = box_keypoints_path
		self.experiment_type = experiment_type
		self.thresholds_config = thresholds_config
		self.video_config = video_config
		self.save_path = save_path
		
		self.classes = {
			"EPM": ElevatedPlusMazeFeatureEngineering,
			"OF": OpenFieldExperimentFeatureEngineering,
			"SI": SocialInteractionFeatureEngineering,
			"YM": YMazeExperimentFeatureEngineering
		}
		
	def run_feature_engineering(self) -> tuple:
		"""
		Derives features for a single video recording.
		:return:
			A Pandas DataFrame with aggregate features for a single video recording and a path for graphs and charts.
		"""
		if self.experiment_type in self.classes.keys():
			df, path = self.classes[self.experiment_type](
				self.pose_keypoints_path,
				self.box_keypoints_path,
				self.thresholds_config,
				self.video_config,
				self.save_path
			).run_feature_engineering()
		else:
			df, path = pd.DataFrame(), None
			
		return df, path
