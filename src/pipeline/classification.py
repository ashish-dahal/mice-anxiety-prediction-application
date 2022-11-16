import numpy as np
import pandas as pd
import pickle

from src.config import config, data_path


class Classification:
	"""
	The class performs the classification of the video recording to one of the anxiety levels (Low, Middle, High) based
	on aggregated features from the feature engineering part.
	"""
	def __init__(self, feature_df: pd.DataFrame, experiment_type: str):
		"""
		Initialize the variables.
		:param feature_df:
			A Pandas DataFrame containing the aggregate features from video recordings.
		:param experiment_type:
			A string abbreviation denoting the experiment type for each video recording.
		"""
		self.feature_df = feature_df
		self.experiment_type = experiment_type
		
		self.allowed_experiments = config["pipeline"]["classification"]["allowed_experiments"]
		self.models_folder = data_path.joinpath(config["pipeline"]["classification"]["models_folder"])

	def run_classification(self) -> str:
		"""
		Perform the classification on a video recording.
		:return:
			A string denoting the anxiety level (Low, Middle, High) for a video recording.
		"""
		anxiety_label = self.__classify(self.feature_df, self.experiment_type)
		return anxiety_label
		
	def __classify(self, df: pd.DataFrame, experiment_type: str) -> str:
		"""
		Classify the video recording of the mouse into an anxiety level (low, middle, high).
		:param df:
			Pandas DataFrame with the derived summary features for a given experiment type.
		:param experiment_type:
			Short abbreviation string denoting the experiment type.
		:return:
			String denoting the anxiety level of the mouse.
		"""
		if experiment_type not in self.allowed_experiments:
			print("[Classification] Warning! The experiment type is not %s. No classification will be performed." %
			      (', '.join(self.allowed_experiments)))
			return ""
		
		temp_df = df.dropna(axis=1)
		data, status = self.__preprocess_data(temp_df, experiment_type)
		
		if status != 0:
			return ""
		
		data = np.reshape(data, (1, -1))
		
		model_path = str(self.models_folder.joinpath(f"{experiment_type}_classification_model.sav"))
		model = pickle.load(open(model_path, 'rb'))
		result = model.predict(data)[0]
		
		mapping = {
			0: "Low",
			1: "Middle",
			2: "High"
		}
		
		result = mapping[result]
		
		return result
	
	@staticmethod
	def __preprocess_data(df: pd.DataFrame, experiment_type: str) -> tuple:
		"""
		Perform legacy_data preprocessing before doing the classification.
		:param df:
			A Pandas DataFrame with derived summary features.
		:param experiment_type:
			String denoting the experiment type.
		:return:
			Preprocessed Pandas DataFrame ready for classification.
			Status code that denotes whether the error was raised in the function.
		"""
		OF_COLUMNS = [
			"DistanceTraveled_Total",
			"TimeCenter_Total",
			"TimeOuter_Total",
			"DistanceInner_Total",
			"DistanceOuter_Total",
			"Velocity_Total",
			"VelocityOuter_Total",
			"FrequencyToEnterZones_Total",
			"Immobility_Total",
			"LatencyToFirstEnter_Total",
			"TurnRights",
			"TurnLeft",
			"RunTurn"
		]
		EPM_COLUMNS = [
			"time_in_open",
			"time_in_closed",
			"latency_to_enter_open",
			"time_head_dipping",
			"frequency_of_entry_to_open",
			"frequency_of_entry_to_closed",
			"distance_traveled_closed",
			"distance_traveled_open"
		]
		
		try:
			if experiment_type == "OF":
				df = df[OF_COLUMNS]
			if experiment_type == "EPM":
				df = df[EPM_COLUMNS]
		except (KeyError, ValueError):
			print(
				"[Classification] Warning! The number of features of the sample is not equal to the number of features "
				"of the training legacy_data.")
			return df, 1
		
		return df, 0
