import sys
import re
import pandas as pd
import numpy as np
import math
from geo.sphere import distance,bearing
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import copy
from tslearn.metrics import dtw, dtw_path
from sklearn.metrics import mean_squared_error

mean_std = {}

matplotlib.rcParams.update({'font.size': 36})


def normalize(expert_df, generated_df):
	df = expert_df
	mean_std['longitude_avg'] = df['longitude'].mean()
	mean_std['longitude_std'] = df['longitude'].std()
	mean_std['latitude_avg'] = df['latitude'].mean()
	mean_std['latitude_std'] = df['latitude'].std()
	mean_std['altitude_avg'] = df['altitude'].mean()
	mean_std['altitude_std'] = df['altitude'].std()

	expert_df['longitude'] = (expert_df['longitude']-mean_std['longitude_avg'])/mean_std['longitude_std']
	expert_df['latitude'] = (expert_df['latitude']-mean_std['latitude_avg'])/mean_std['latitude_std']
	expert_df['altitude'] = (expert_df['altitude']-mean_std['altitude_avg'])/mean_std['altitude_std']

	generated_df['longitude'] = (generated_df['longitude']-mean_std['longitude_avg'])/mean_std['longitude_std']
	generated_df['latitude'] = (generated_df['latitude']-mean_std['latitude_avg'])/mean_std['latitude_std']
	generated_df['altitude'] = (generated_df['altitude']-mean_std['altitude_avg'])/mean_std['altitude_std']

def unnormalize(df):

	unnormalized = copy.deepcopy(df)
	unnormalized['longitude'] = (unnormalized['longitude']*mean_std['longitude_std']+mean_std['longitude_avg'])
	unnormalized['latitude'] = (unnormalized['latitude']*mean_std['latitude_std']+mean_std['latitude_avg'])
	unnormalized['altitude'] = (unnormalized['altitude']*mean_std['altitude_std']+mean_std['altitude_avg'])

	return unnormalized


def plot_rmse(rmse_dfs,method):
	dfs = []
	dimension = []
	df_method = []
	plt.figure(figsize=(20,10))
	for (m,rmse_df) in enumerate(rmse_dfs):
		dfs.append(rmse_df['lon'])
		dimension.extend(['Longitude']*len(rmse_df.index))
		dfs.append(rmse_df['lat'])
		dimension.extend(['Latitude']*len(rmse_df.index))
		dfs.append(rmse_df['alt'])
		dimension.extend(['Altitude']*len(rmse_df.index))
		dfs.append(rmse_df['all'])
		dimension.extend(['3D']*len(rmse_df.index))
		df_method.extend([method[m]]*4*len(rmse_df.index))


	df = pd.DataFrame({'rmse':pd.concat(dfs,ignore_index=True)})

	df['dimension']=dimension
	df['Method']=df_method

	ax = sns.boxplot(x='dimension',y='rmse',hue='Method',data=df,palette=['whitesmoke','darkgray'])
	dim_meth = [['Longitude','OnePolicy'],['Longitude','MultPolicies'],
				['Latitude','OnePolicy'],['Latitude','MultPolicies'],
				['Altitude','OnePolicy'],['Altitude','MultPolicies'],
				['3D','OnePolicy'],['3D','MultPolicies']]
	medians = [df[(df['dimension']==x[0]) & (df['Method']==x[1])].median().values for x in dim_meth]

	median_labels = [str(int(np.round(s))) for s in medians]
	pos = range(len(medians) // 2)
	ax.set_xlabel('Dimension')
	ax.set_ylabel('RMSE (meters)')
	ax.set(ylim=(-5000,140000))
	for tick,label in zip(pos,ax.get_xticklabels()):
		ax.text(pos[tick]-0.04, medians[2*tick] + 0.5, median_labels[2*tick], horizontalalignment='right', size='x-small', color='black', weight='semibold')
		ax.text(pos[tick]+0.04, medians[2*tick+1] + 0.5, median_labels[2*tick+1], horizontalalignment='left', size='x-small', color='black', weight='semibold')
	plt.show()


def plot_track_errors(track_dfs,method):
	dfs = []
	error = []
	df_method = []
	plt.figure(figsize=(20,9))
	for (m,track_df) in enumerate(track_dfs):
		dfs.append(track_df['along_track'])
		error.extend(['Along Track']*len(track_df.index))
		dfs.append(track_df['cross_track'])
		error.extend(['Cross Track']*len(track_df.index))
		dfs.append(track_df['vertical'])
		error.extend(['Vertical']*len(track_df.index))
		df_method.extend([method[m]]*3*len(track_df.index))

	df = pd.DataFrame({'error':pd.concat(dfs,ignore_index=True)})

	df['error_type']=error
	df['Method']=df_method
	ax = sns.boxplot(x='error_type',y='error',hue='Method',data=df,palette=['whitesmoke','darkgray'])
	error_type_meth = [['Along Track','OnePolicy'],['Along Track','MultPolicies'],
				['Cross Track','OnePolicy'],['Cross Track','MultPolicies'],
				['Vertical','OnePolicy'],['Vertical','MultPolicies']]

	medians = [df[(df['error_type']==x[0]) & (df['Method']==x[1])].median().values for x in error_type_meth]

	median_labels = [str(int(np.round(s))) for s in medians]
	pos = range(len(medians) // 2)
	ax.set_xlabel('Type of Error')
	ax.set_ylabel('Error (meters)')
	ax.set(ylim=(-101000,60000))
	for tick,label in zip(pos,ax.get_xticklabels()):
		ax.text(pos[tick]-0.04, medians[2*tick] + 0.5, median_labels[2*tick], horizontalalignment='right', size='x-small', color='black', weight='semibold')
		ax.text(pos[tick]+0.04, medians[2*tick+1] + 0.5, median_labels[2*tick+1], horizontalalignment='left', size='x-small', color='black', weight='semibold')

	plt.show()


def compute_RMSE(path,expert,generated):
	i = 0
	dists = []
	for pair in path:
		expert_point = expert[pair[0]]
		generated_point = generated[pair[1]]

		lon_dist = generated_point[1] * (111412.84 * math.cos(math.radians(generated_point[2])) - 93.5 * math.cos(math.radians(3*generated_point[2]))+0.118 * math.cos(math.radians(5*generated_point[2])))\
		- expert_point[1] * (111412.84 * math.cos(math.radians(expert_point[2])) - 93.5 * math.cos(math.radians(3*expert_point[2]))+0.118 * math.cos(math.radians(5*expert_point[2])))
		lat_dist = generated_point[2] * (111132.92 - 559.82 * math.cos(math.radians(2*generated_point[2])) + 1.175 * math.cos(math.radians(4*generated_point[2])) - 0.023* math.cos(math.radians(6*generated_point[2])))\
		- expert_point[2] * (111132.92 - 559.82 * math.cos(math.radians(2*expert_point[2])) + 1.175 * math.cos(math.radians(4*expert_point[2])) - 0.023* math.cos(math.radians(6*expert_point[2])))

		alt_dist = 0.3048*(abs(generated_point[3]-expert_point[3]))

		dists.append([lon_dist,lat_dist,alt_dist])


	return np.sqrt(np.mean(np.square(dists),axis=0)), np.sqrt(np.mean(np.sum(np.square(dists),axis=1)))

def calculate_bearing(pred_traj):
	bearing_np = np.zeros(len(pred_traj))
	last_bearing = 0
	for i in range(len(pred_traj)):
		if i == len(pred_traj)-1:
			bearing_np[i] = last_bearing
		else:
			bearing_np[i] = math.radians(bearing([pred_traj[i][1], pred_traj[i][2]], [pred_traj[i+1][1], pred_traj[i+1][2]]))
			last_bearing = bearing_np[i]
	return bearing_np

def calculate_track_errors(path, real_traj, pred_traj):
	along_track_error = 0
	cross_track_error = 0
	altitude_error = 0
	pred_bearing = calculate_bearing(pred_traj)
	sign = lambda a: 1 if a>0 else -1 if a<0 else 0
	for p1, p2 in path:
		delta_x = pred_traj[p2][1] * (111412.84 * math.cos(math.radians(pred_traj[p2][2])) - 93.5 * math.cos(math.radians(3*pred_traj[p2][2]))+0.118 * math.cos(math.radians(5*pred_traj[p2][2])))\
		- real_traj[p1][1] * (111412.84 * math.cos(math.radians(real_traj[p1][2])) - 93.5 * math.cos(math.radians(3*real_traj[p1][2]))+0.118 * math.cos(math.radians(5*real_traj[p1][2])))
		delta_y = pred_traj[p2][2] * (111132.92 - 559.82 * math.cos(math.radians(2*pred_traj[p2][2])) + 1.175 * math.cos(math.radians(4*pred_traj[p2][2])) - 0.023* math.cos(math.radians(6*pred_traj[p2][2])))\
		- real_traj[p1][2] * (111132.92 - 559.82 * math.cos(math.radians(2*real_traj[p1][2])) + 1.175 * math.cos(math.radians(4*real_traj[p1][2])) - 0.023* math.cos(math.radians(6*real_traj[p1][2])))

		along_track_error += abs(delta_x * math.sin(pred_bearing[p2]) + delta_y * math.cos(pred_bearing[p2]))
		cross_track_error += abs(delta_x * math.cos(pred_bearing[p2]) - delta_y * math.sin(pred_bearing[p2]))
		altitude_error += abs((pred_traj[p2][3] - real_traj[p1][3]) / 3.2808)

	return along_track_error/len(path), cross_track_error/len(path), altitude_error/len(path)


if __name__ == "__main__":
	# ------Input Files------
	'''
	Provide paths for expert and generated trajectories
	'''
	exp_num = 5
	prcnt = 0
	phase = 'cruising'
	expert_file_path = f'datasets/{phase}_test.csv'

	expert_df = pd.read_csv(expert_file_path)

	generated_files = [
		f'experiments/exp{i}_{phase}/exp{i}.csv'
		for i in range(exp_num)
	]

	vector_rmses = []
	rmses = []
	rmse_dfs = []
	track_errors = []
	track_error_dfs = []
	etas = []

	rmses_dict = {}

	for generated_file_path in generated_files:
		generated_df = pd.read_csv(generated_file_path)
		normalize(expert_df,generated_df)
		expert_grouped = expert_df[['trajectory_ID','longitude','latitude','altitude']].groupby(['trajectory_ID'])
		generated_grouped = generated_df[['traj_id','longitude','latitude','altitude']].groupby(['traj_id'])
		paths = {}
		
		for ((expert_name, expert_group),(generated_name,generated_group)) in zip(expert_grouped, generated_grouped):
			etas.append((len(generated_group.index)-len(expert_group.index))*5)
			
			if expert_name != generated_name:
				print('error')
				exit(0)

			paths[expert_name] = dtw_path(
				expert_group[['longitude','latitude','altitude']].values,
				generated_group[['longitude','latitude','altitude']].values
			)[0]

			unnormalized_expert = unnormalize(expert_group).values
			unnormalized_generated = unnormalize(generated_group).values

			vector_rmse, rmse = compute_RMSE(paths[expert_name],
											unnormalized_expert,
											unnormalized_generated)

			vector_rmses.append(vector_rmse)
			rmses.append(rmse)

			track_error = calculate_track_errors(paths[expert_name], 
												unnormalized_expert, 
												unnormalized_generated)
			track_errors.append(track_error)

	rmse_dfs = pd.DataFrame(vector_rmses,columns=['lon','lat','alt'])
	rmse_dfs['all'] = rmses
	track_error_dfs = pd.DataFrame(track_errors,columns=['along_track','cross_track','vertical'])

	print(' Cross Track')
	print('---Average---')
	print(np.mean(track_error_dfs,axis=0))
	print('---Median---')
	print(np.median(track_error_dfs,axis=0))
	print('---Std---')
	print(np.std(track_error_dfs,axis=0))
	print()
	print(' RMSE')
	print('---Average---')
	print(np.mean(rmses))
	print(np.mean(vector_rmses,axis=0))
	print('---Median---')
	print(np.median(rmses))
	print(np.median(vector_rmses,axis=0))
	print('---Std---')
	print(np.std(rmses))
	print(np.std(vector_rmses,axis=0))
	print()
	print(' ETA')
	print('---Average---')
	print(np.mean(etas))
	print('---Median---')
	print(np.median(etas))
	print('---Std---')
	print(np.std(etas))
	print()
	exit()

	plot_rmse(rmse_dfs,method)
	plot_track_errors(track_error_dfs,method)
	exit(0)
