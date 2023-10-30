import numpy as np
import random
import os
import csv
from app_learning import ApprenticeshipLearning
import pandas as pd
import timeit
from regression_network import Net
import torch as T


def load_expert_trajectories(path, centroid, train_ratio, cluster, save_path):
	'''
	Load expert trajectories
	data: choose which variables you want to use
	cluster_data: pre-defined clusters of trajectories
	'''
	data = pd.read_csv(path)
	data = data[
		['trajectory_ID', 'longitude', 'latitude', 'altitude', 'timestamp', 'UGRD', 'HGT', 'TMP', 'VGRD', 'Cluster']]
	cols = list(data.columns)
	
	cl = data['Cluster'] == cluster
	cluster_data = data[cl]

	keys = list(cluster_data.groupby('trajectory_ID').groups.keys())
	random.shuffle(keys)
	num_experts = np.size(keys)
	train_set_size = int(train_ratio * num_experts)
	train_keys = keys[:train_set_size]
	test_keys = keys[train_set_size:]
	if centroid not in train_keys:
		train_keys.append(centroid)

	grouped_data = cluster_data.groupby('trajectory_ID')
	random.shuffle(test_keys)
	
	all_df = pd.concat([grouped_data.get_group(x) for x in list(keys)])
	train_df = pd.concat([grouped_data.get_group(x) for x in list(train_keys)])
	test_df = pd.concat([grouped_data.get_group(x) for x in list(test_keys)])
	
	filename = save_path + "test_keys.csv"
	saved_test_df = test_df[['trajectory_ID', 'longitude', 'latitude', 'altitude']]
	saved_test_df.to_csv(filename, index=False)

	return all_df, train_df, test_df


def preprocess_trajectories(all_data, train_data, test_data):
	'''
	Min-max normalization
	'''
	tr_data = train_data.drop(['trajectory_ID', 'Cluster', 'timestamp'], axis=1).copy()

	train_norm_data = train_data.drop(['trajectory_ID', 'Cluster', 'timestamp'], axis=1)
	train_norm_data = (train_norm_data - tr_data.min()) / (tr_data.max() - tr_data.min())
	train_norm_data.insert(0, 'trajectory_ID', train_data['trajectory_ID'])
	train_norm_data.insert(len(list(train_data.columns)) - 2, 'Cluster', train_data['Cluster'])
	train_norm_data.insert(4, 'timestamp', train_data['timestamp'])

	test_norm_data = test_data.drop(['trajectory_ID', 'Cluster', 'timestamp'], axis=1)
	test_norm_data = (test_norm_data - tr_data.min()) / (tr_data.max() - tr_data.min())
	test_norm_data.insert(0, 'trajectory_ID', test_data['trajectory_ID'])
	test_norm_data.insert(len(list(test_data.columns)) - 2, 'Cluster', test_data['Cluster'])
	test_norm_data.insert(4, 'timestamp', test_data['timestamp'])

	return train_norm_data, test_norm_data


def find_transition_steps(data, centroid):
	'''
	Find the deltas (x,y,z) of the movement of the aircraft
	'''
	centroid_data = data['trajectory_ID'] == centroid
	temp_data = data[centroid_data]
	temp_data = temp_data.drop(['trajectory_ID'], axis=1)

	temp_data = temp_data.diff().abs()
	delta_x, delta_y, delta_z = temp_data.mean(axis=0)[0:3]

	return delta_x, delta_y, delta_z


def main():
	'''
	Parameters should change according to each scenario
	'''
	# Parameters
	experiment_number = 1
	test_train_ratio = 0.9
	e_decay = 140
	alpha = 0.001
	gamma = 0.99
	epsilon = 0.9
	num_episodes = 15000
	mem_size = int(1e5)
	batch_size = 500
	is_trained = True
	scenario_name = "HEL"
	scenario_day = "01"
	cluster = 1

	# Paths
	save_path = str(scenario_name) + "/" + str(scenario_name) + "_" + str(scenario_day) + "_experiment_" + str(experiment_number) + "/"
	expert_path = str(scenario_name) + "/" + str(scenario_name) + "_" + str(scenario_day) + "_cluster_final.csv"
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	'''
	Centroid trajectories that are pre-defined according to the scenario
	'''
	#1
	centroid = '20190107_FIN3PA_HEL_LIS'
	#2
	#centroid = '20190127_TAP747_HEL_LIS'
	#3
	#centroid = '20190125_TAP747_HEL_LIS'

	all_data, train_data, test_data = load_expert_trajectories(expert_path, centroid, test_train_ratio, cluster, save_path)
	train_norm_data, test_norm_data = preprocess_trajectories(all_data, train_data, test_data)
	train_flights = train_norm_data.groupby('trajectory_ID')
	test_flights = test_norm_data.groupby('trajectory_ID')

	print("Number of expert trajectories (training) : " + str(len(train_flights)))
	print("Number of expert trajectories (testing) : " + str(len(test_flights)))

	dx, dy, dz = find_transition_steps(train_data, centroid)
	print()
	print("DX step: " + str(dx))
	print("DY step: " + str(dy))
	print("DZ step: " + str(dz))
	print("===============================")

	# Angles
	'''
	Angles that are found during pre-processing - should change according to the scenario heading angles
	'''
	actions = [190, 210, 230, 250, 260]

	# Regression
	'''
	Parameters for the regression network that is used to predict the z-dimension
	'''
	n_episodes = 100000
	num_features = 2
	num_hidden = 75
	num_output = 1
	regressor = Net(num_features, num_hidden, num_output)
	regressor = regressor.float()

	if is_trained:
		path = str(scenario_name) + "_" + str(scenario_day) + "_" + str(cluster) + "_regression"
		regressor.load_state_dict(T.load(path))
	else:
		X = train_norm_data[['longitude', 'latitude']].to_numpy()
		Y = train_norm_data['altitude'].to_numpy()
		regressor.train(n_episodes, X, Y)
		path = str(scenario_name) + "_" + str(scenario_day) + "_" + str(cluster) + "_regression"
		T.save(regressor.state_dict(), path)

	# TRAINING
	app_learning = ApprenticeshipLearning(alpha, gamma, epsilon, mem_size, batch_size, actions, all_data, train_data,
										  test_data, train_norm_data, test_norm_data, dx, dy, regressor,
										  centroid, save_path, e_decay, scenario_name, scenario_day, cluster)
	start = timeit.default_timer()
	app_learning.train(num_episodes)
	stop = timeit.default_timer()
	elapsed = stop - start
	hours, rem = divmod(elapsed, 3600)
	minutes, seconds = divmod(rem, 60)
	print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
	main()
