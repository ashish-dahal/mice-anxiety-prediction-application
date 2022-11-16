from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.training_scripts.data_preparation import read_dataset

sns.set(rc={'figure.figsize': (15, 15)}, style="whitegrid")


def get_features_for_clustering(dataset):
	X = []
	columns = list(dataset[0]['Features'].keys())[:4]
	columns.extend(['has_treatment', 'is_week_6'])
	
	for i in range(len(dataset)):
		values = list(dataset[i]['Features'].values())
		del values[4]  # Deleting time stamped active interaction
		
		if '_treatment_' in dataset[i]['Video']:
			values.append(1)
		else:
			values.append(0)
		
		if '_w6_' in dataset[i]['Video']:
			values.append(1)
		else:
			values.append(0)
		
		X.append(values)
	
	X = np.array(X)
	X = pd.DataFrame(X, columns=columns)
	X = X.drop(columns='interaction_frequency')
	X[['has_treatment', 'is_week_6']] = X[['has_treatment', 'is_week_6']].astype(np.uint8)
	
	return X


def show_information(df, experiment_no=1):
	print(f"Shape of df: {df.shape}")
	print(f"Columns of df: {list(df.columns)}")
	print(f"Description of df: ")
	print(df.describe().T[['mean', 'std', 'max', 'min']])
	
	g = sns.pairplot(df.drop(columns="is_week_6"), hue="has_treatment")
	g.fig.suptitle(f"Treatment and Control groups (Experiment #{experiment_no})", fontsize=20)
	plt.show()
	
	
def show_predictions(df, y_pred, n_clusters, experiment_no):
	for i in range(n_clusters):
		print(df[y_pred == i].describe())
	
	color_palette = sns.color_palette("husl", n_clusters).as_hex()
	
	temp_X = df.copy().drop(columns=["has_treatment", "is_week_6"])
	temp_X['y_pred'] = y_pred
	
	g = sns.pairplot(temp_X, hue='y_pred', palette=color_palette)
	g.fig.suptitle(f"Resultant clusters (Experiment #{experiment_no})", fontsize=20)
	plt.show()
	
	
def point_features_clustering(dataset, n_clusters=3, experiment_no=1, use_hdbscan=False, metric="euclidean",
                              show_graphs=True):
	X = get_features_for_clustering(dataset)
	
	train_X = X[[
		'distance_moved',
		'first_interaction_latency',
		'time_of_active_interaction'
	]]
	
	scaler = StandardScaler()
	train_X_transformed = scaler.fit_transform(train_X)
	train_X = pd.DataFrame(train_X_transformed, columns=train_X.columns)
	
	if use_hdbscan:
		clusterer = hdbscan.HDBSCAN(metric=metric)
	else:
		clusterer = TimeSeriesKMeans(
			metric=metric,
			n_clusters=n_clusters,
			n_jobs=-1,
			random_state=0
		)
		
	clusterer.fit(train_X)
	
	y_pred = clusterer.labels_
	
	if use_hdbscan:
		n_clusters = len(pd.unique(y_pred))
	
	if show_graphs:
		show_information(X, experiment_no)
		show_predictions(X, y_pred, n_clusters, experiment_no)


def main():
	first_dataset, second_dataset = read_dataset()
	
	experiment_no = 1
	
	if experiment_no == 1:
		dataset = first_dataset
		n_clusters = 3
	else:
		dataset = second_dataset
		n_clusters = 2
	
	point_features_clustering(dataset, n_clusters=n_clusters, experiment_no=experiment_no, show_graphs=False)


if __name__ == "__main__":
	main()
