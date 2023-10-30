import os
import argparse
import torch
#import wandb
import TripleGAIL.environment as environment
import pandas as pd
import numpy as np
from distutils.util import strtobool
from utils.torch import *
from utils.math import *
import torch.nn.functional as F
import math
import time
from torch import nn
from sklearn.model_selection import KFold
from statistics import mean
from TripleGAIL.ppo import ppo_step
from TripleGAIL.models import Policy, Value, Discriminator, Selector
from TripleGAIL.agent import Agent

dtype = torch.float32
# CUDA GPU
#device = torch.device("cuda")
# CPU
device = torch.device("cpu")
# MACOS
#device = torch.device("cpu")

t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)

config ={
	"learning_rate": 1e-4,
	"epochs": 1000,
	"batch_size": 500,
	"path_size": 50,
	"samples_per_episode": 15000,
	"critic_epochs": 50,
	"selector_epochs": 50,
	"discriminator_epochs": 25,
	"bcloning_epochs": 100,
	"bcloning_batch": 64,
	"bcloning_learning_rate": 1e-3,
	"logstd": -1.0,
	"phase": "cruising",
	"activation": "leaky_relu"
}	

#wandb.init(project="triple-gail-" + config["phase"], config=config)

def get_args():
	'''
	We can select what features we want to use
	E.g. if we want to drop weather features -> -wf False
	'''
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-wf', '--weather_features', default=True)
	parser.add_argument('-tf', '--time_features', default=True)
	parser.add_argument('-mf', '--model_features', default=True)
	parser.add_argument('-df', '--delay_features', default=True)
	parser.add_argument('-e', '--experiment', default="1")
	parser.add_argument('-ne', '--num_episodes', default=1500, type=int)
	parser.add_argument('-nm', '--num_modes', default=9, type=int)
	parser.add_argument('-ps', '--path_size', default=150, type=int)
	parser.add_argument('-spe', '--samples_per_episode', default=10000, type=int)
	parser.add_argument('-pt', '--pre_train', default=True)
	parser.add_argument('-bce', '--bcloning_epochs', default=100, type=int)
	parser.add_argument('-ce', '--critic_epochs', default=50, type=int)
	parser.add_argument('-pe', '--post_epochs', default=50, type=int)
	parser.add_argument('-de', '--disc_epochs', default=15, type=int)
	parser.add_argument('-bcb', '--bcloning_batch', default=64, type=int)
	parser.add_argument('-bs', '--batch_size', default=2500, type=int)
	parser.add_argument('-clr', '--critic_lr', default=1e-3, type=float)
	parser.add_argument('-plr', '--post_lr', default=3e-4, type=float)
	parser.add_argument('-dlr', '--disc_lr', default=3e-4, type=float)
	parser.add_argument('-erl', '--env_reward_lambda', default=0.000005, type=float)
	parser.add_argument('-lstd', '--logstd', default=-1.0, type=float)
	parser.add_argument('-ld', '--l_discriminator', default=0.7, type=float)
	parser.add_argument('-l', '--log_dir', default= "TripleGAIL/logs/")
	parser.add_argument('-i', '--input_dir', default="datasets/")
	parser.add_argument('-o', '--output_dir', default="TripleGAIL/experiments/")
	parser.add_argument('-s', '--save_dir', default="TripleGAIL/model_weights/")
	args = parser.parse_args()

	return args

def behavioral_cloning(x_train, y_train, c_train, policy, num_modes):
	'''
	Behavioral cloning to initialize the weights of the policy network
	'''
	epochs = config["bcloning_epochs"]
	batch_size = config["bcloning_batch"]
	criterion = nn.MSELoss()
	optim = torch.optim.Adam(policy.parameters(), lr=config["bcloning_learning_rate"])
	kf = KFold(n_splits=10, shuffle=True)
	f = 1

	#wandb.watch(policy, criterion, log="all", log_freq=1)
	
	c_train_onehot = np.zeros([c_train.shape[0], num_modes])
	for i in range(x_train.shape[0]):
		code = c_train[i][-1]
		c_train_onehot[i][code] = 1

	for train_index, validation_index in kf.split(x_train):
		policy.train()
		x_t = x_train[train_index]
		x_val = x_train[validation_index]
		y_t = y_train[train_index]
		y_val = y_train[validation_index]
		c_t = c_train_onehot[train_index]
		c_val = c_train_onehot[validation_index]

		x_val = tensor(x_val, dtype=torch.float32).unsqueeze(0).to(device)
		y_val = tensor(y_val, dtype=torch.float32).unsqueeze(0).to(device)
		c_val = tensor(c_val, dtype=torch.float32).unsqueeze(0).to(device)
		print()
		print(f"Fold: {f}")
		print()

		f += 1
		for i in range(epochs):
			train_loss = []
			lon_loss = []
			lat_loss = []
			alt_loss = []

			for j in range(0, len(x_t), batch_size):
				if len(x_t[j:j+batch_size]) < batch_size:
					continue

				x_batch = tensor(x_t[j:j+batch_size], dtype=torch.float32).unsqueeze(0).to(device)
				y_batch = tensor(y_t[j:j+batch_size], dtype=torch.float32).unsqueeze(0).to(device)
				c_batch = tensor(c_t[j:j+batch_size], dtype=torch.float32).unsqueeze(0).to(device)

				optim.zero_grad()

				outputs = policy.select_bc_action(x_batch, c_batch)
				
				loss = criterion(outputs, y_batch)

				loss.backward()
				optim.step()

				train_loss.append(loss.item())


			val_outputs = policy.select_bc_action(x_val, c_val)
			val_loss = criterion(val_outputs, y_val)

			print(f"Train loss: {round(mean(train_loss), 3)} | Val loss: {round(val_loss.item(), 3)}")
		
			#wandb.log({"Train loss":loss, "Validation loss":val_loss}, step=f)

	torch.save(policy.state_dict(), "TripleGAIL/model_weights/bc_" + str(config["phase"]) + ".pth")
	exit()

def expert_reward(state, action, code, discrim_net):
	state_action_code = tensor(np.hstack([state, action, code]), dtype=dtype)
	with torch.no_grad():
		return -math.log(discrim_net(state_action_code)[0].item())

def estimate_advantages(rewards, masks, values, gamma, tau, device):
	'''
	GAE estimation
	'''
	rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
	tensor_type = type(rewards)
	deltas = tensor_type(rewards.size(0), 1)
	advantages = tensor_type(rewards.size(0), 1)

	prev_value = 0
	prev_advantage = 0
	for i in reversed(range(rewards.size(0))):
		deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
		advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

		prev_value = values[i, 0]
		prev_advantage = advantages[i, 0]

	returns = values + advantages
	advantages = (advantages - advantages.mean()) / advantages.std()

	advantages, returns = to_device(device, advantages, returns)
	return advantages, returns

def normalize_observations(obs, features, conf):
	'''
	Min-max and zero-mean normalization
	'''
	norm_obs = obs.copy()

	for f in features:
		f_avg = f + '_avg'
		f_std = f + '_std'
		f_max = f + '_max'
		f_min = f + '_min'

		if f_avg in conf:
			norm_obs[f] = (obs[f] - conf[f_avg]) / conf[f_std]
		else:
			norm_obs[f] = (obs[f] - conf[f_min]) / (conf[f_max] - conf[f_min])

	return norm_obs


def normalize_actions(actions, conf):
	'''
	Action normalization with zero-mean
	'''
	norm_actions = actions.copy()

	norm_actions['dlon'] = (actions['dlon'].copy() - conf['dlon_avg']) / conf['dlon_std']
	norm_actions['dlat'] = (actions['dlat'].copy() - conf['dlat_avg']) / conf['dlat_std']
	norm_actions['dalt'] = (actions['dalt'].copy() - conf['dalt_avg']) / conf['dalt_std']

	return norm_actions

def grad_penalty(sa_real, sa_fake):
	'''
	Calculation of the gradient penalty term
	'''
	real_data = sa_real.data
	fake_data = sa_fake.data
				   
	if real_data.size(0) < fake_data.size(0):
		idx = np.random.permutation(fake_data.size(0))[0: real_data.size(0)]
		fake_data = fake_data[idx, :]
	else: 
		idx = np.random.permutation(real_data.size(0))[0: fake_data.size(0)]
		real_data = real_data[idx, :]

	alpha = torch.rand(real_data.size(0), 1).expand(real_data.size()).to(device)
	x_hat = alpha * real_data + (1 - alpha) * fake_data

	x_hat_out = discrim_net(x_hat.to(device).requires_grad_())
	gradients = torch.autograd.grad(outputs=x_hat_out, inputs=x_hat, \
					grad_outputs=torch.ones(x_hat_out.size()).to(device), \
					create_graph=True, retain_graph=True, only_inputs=True)[0]

	# gp_lp = 0
	# if gp_lp:
	# 	return ( torch.max(0, gradients.norm(2, dim=1) - self.gp_center) ** 2).mean() * self.gp_lambda
	# else:
	return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 1.0

def update_params(batch, batch_selector, i_iter, generated_flight_ids):
	'''
	Update all the networks: policy, discriminator, selector
	For policy, we could choose among PPO and TRPO
	Training steps could change according to the loss function of each network
	'''
	disc_steps = 25
	sel_steps = 50
	#########################################################################
	# Optional: Select the expert trajectories that were used in the rollouts
	exp_obs = []
	exp_actions = []
	exp_codes = []
	exp_modes = []
	for exp in generated_flight_ids:
		exp_obs.extend(obs_train_np[exp].tolist())
		exp_actions.extend(actions_train_np[exp].tolist())
		exp_codes.extend(codes_train_np[exp].tolist())
		exp_modes.extend(clusters_train_np[exp].tolist())

	exp_obs = np.array(exp_obs)
	exp_actions = np.array(exp_actions)
	exp_codes = np.array(exp_codes)
	exp_modes = np.array(exp_modes).T[0]
	exp_modes  = torch.from_numpy(np.stack(exp_modes)).to(torch.long).to(device)

	expert_traj = np.concatenate([exp_obs, exp_actions, exp_codes], axis=1)
	##########################################################################

	selector_states = torch.from_numpy(np.stack(batch_selector.state)).to(dtype).to(device)
	selector_actions = torch.from_numpy(np.stack(batch_selector.action)).to(dtype).to(device)
	selector_codes = torch.from_numpy(np.stack(batch_selector.code)).to(dtype).to(device)

	states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
	actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
	codes = torch.from_numpy(np.stack(batch.code)).to(dtype).to(device)
	modes = torch.from_numpy(np.stack(batch.mode)).to(torch.long).to(device)
	rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
	masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
	with torch.no_grad():
		values = value_net(states)
		fixed_log_probs = policy_net.get_log_prob(states, codes, actions)

	# Get advantage estimation from the trajectories
	advantages, returns = estimate_advantages(rewards, masks, values, 0.99, 0.95, device)
	
	# Update Discriminator
	expert_state_action_codes = torch.from_numpy(expert_traj).to(dtype).to(device)
	for _ in range(disc_steps):
		omega = 0.5
		g_o = discrim_net(torch.cat([states, actions, codes], 1))
		s_o = discrim_net(torch.cat([selector_states, selector_actions, selector_codes], 1))
		e_o = discrim_net(expert_state_action_codes)

		optimizer_discrim.zero_grad()
		generator_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device))
		selector_loss = discrim_criterion(s_o, ones((selector_states.shape[0], 1), device=device)) 
		real_loss = discrim_criterion(e_o, zeros((expert_traj.shape[0], 1), device=device))
		discrim_loss = omega * generator_loss + \
					   (1 - omega) * selector_loss + \
					   real_loss

		discrim_loss += grad_penalty(expert_state_action_codes, torch.cat([states, actions, codes], 1))
		
		discrim_loss.backward()
		optimizer_discrim.step()

	print(f"Discriminator Loss: {discrim_loss.item()}")
	

	# Update Selector
	expert_state_action = torch.from_numpy(np.concatenate([exp_obs, exp_actions], axis=1)).to(dtype).to(device)
	for _ in range(sel_steps):
		l_g = 0.5
		l_e = 0.5
		g_o = selector_net(torch.cat([states, actions], 1))
		e_o = selector_net(expert_state_action)
		s_o = discrim_net(torch.cat([selector_states, selector_actions, selector_codes], 1))
		disc_loss = (1 - omega) * discrim_criterion(s_o, ones((selector_states.shape[0], 1), device=device))
		
		optimizer_selector.zero_grad()
		sel_loss = l_g * sel_criterion(g_o, modes) + l_e * sel_criterion(e_o, exp_modes) - 0.5*disc_loss
		sel_loss.backward()
		optimizer_selector.step()

	print(f"Selector Loss: {sel_loss.item()}")

	# wandb.log({
	# 	"Discriminator loss":discrim_loss,
	# 	"Real loss (D)": real_loss,
	# 	"Generator loss (D)": generator_loss,
	# 	"Selector loss (D)": selector_loss,
	# 	"Selector loss":sel_loss
	# })
	
	# Perform mini-batch PPO update
	optim_batch_size = 2048
	optim_epochs = 3
	optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
	for _ in range(optim_epochs):
		perm = np.arange(states.shape[0])
		np.random.shuffle(perm)
		perm = LongTensor(perm).to(device)

		states, actions, codes, returns, advantages, fixed_log_probs = \
			states[perm].clone(), actions[perm].clone(), codes[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

		for i in range(optim_iter_num):
			ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
			states_b, actions_b, codes_b, advantages_b, returns_b, fixed_log_probs_b = \
				states[ind], actions[ind], codes[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

			ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, codes_b, returns_b,
					 advantages_b, fixed_log_probs_b, 0.2, 1e-3)

			#trpo_step(policy_net, value_net, states, actions, codes, returns, advantages, max_kl, damping, 1e-3, use_fim=True)

args = get_args()

####### MAIN ########
train_bcloning = False
train_method = True

'''
Dataset feature selection
'''
features = ['longitude', 'latitude', 'altitude']
exp_name = ''
if bool(args.time_features) is True:
	features.append('timestamp')
	exp_name += 'tf_'
if bool(args.weather_features) is True:
	features.append('temp_iso')
	features.append('v_wind_component')
	features.append('u_wind_component')
	exp_name += 'wf_'
if bool(args.model_features) is True:
	features.append('model_id')
	exp_name += 'mf_'
if bool(args.delay_features) is True:
	features.append('delay')
	exp_name += 'df_'
exp_name += args.experiment
args.experiment = exp_name

#features = ['longitude', 'latitude', 'altitude', 'timestamp', 'temp_iso', 'v_wind_component', 'u_wind_component', 'model_id']
actions = ['dlon', 'dlat', 'dalt']

# Read the scenarios
print("Reading the scenario...")
phase = config["phase"]
train_set_df = pd.read_csv(args.input_dir + f"paris_{phase}_train.csv")
test_set_df = pd.read_csv(args.input_dir + f"paris_{phase}_test.csv")

# DataFrames
original_data_df = train_set_df[['trajectory_ID', 'longitude', 'latitude', 'altitude', 'timestamp',
								'temp_iso', 'v_wind_component', 'u_wind_component',
								'model_id', 'delay', 'Clusters', 'dlon', 'dlat', 'dalt']]

obs_train_df = original_data_df[['trajectory_ID', 'longitude', 'latitude', 'altitude', 'timestamp',
								'temp_iso', 'v_wind_component', 'u_wind_component',
								'model_id', 'delay', 'Clusters']]
actions_train_df = original_data_df[['trajectory_ID', 'dlon', 'dlat', 'dalt', 'Clusters']]
cluster_train_df = original_data_df[['trajectory_ID', 'Clusters']]

obs_test_df = test_set_df[['trajectory_ID', 'longitude', 'latitude', 'altitude', 'timestamp',
						'temp_iso', 'v_wind_component', 'u_wind_component',
						'model_id', 'delay', 'Clusters']]
actions_test_df = test_set_df[['trajectory_ID', 'dlon', 'dlat', 'dalt', 'Clusters']]
cluster_test_df = test_set_df[['trajectory_ID', 'Clusters']]

# Number of flights
train_flight_num = train_set_df['trajectory_ID'].nunique()
test_flight_num = test_set_df['trajectory_ID'].nunique()
print(f"Number of train flights: {train_flight_num} | Number of test flights: {test_flight_num}")

obs_test = obs_test_df.drop(['Clusters'], axis=1)
print(f"Number of features: {len(features)}")
print(f"Number of modes: {len(train_set_df['Clusters'].unique())}")
assert len(obs_train_df) == len(actions_train_df)

# DataFrame Statistics
conf = {
		'dlon_avg': original_data_df.loc[:, "dlon"].mean(),
		'dlon_std': original_data_df.loc[:, "dlon"].std(),
		'dlat_avg': original_data_df.loc[:, "dlat"].mean(),
		'dlat_std': original_data_df.loc[:, "dlat"].std(),
		'dalt_avg': original_data_df.loc[:, "dalt"].mean(),
		'dalt_std': original_data_df.loc[:, "dalt"].std(),
		'longitude_avg': original_data_df.loc[:, "longitude"].mean(),
		'longitude_std': original_data_df.loc[:, "longitude"].std(),
		'latitude_avg': original_data_df.loc[:, "latitude"].mean(),
		'latitude_std': original_data_df.loc[:, "latitude"].std(),
		'altitude_avg': original_data_df.loc[:, "altitude"].mean(),
		'altitude_std': original_data_df.loc[:, "altitude"].std(),
		'timestamp_avg': original_data_df.loc[:, "timestamp"].mean(),
		'timestamp_std': original_data_df.loc[:, "timestamp"].std(),
		# 'temp_iso_avg': original_data_df.loc[:, "temp_iso"].mean(),
		# 'temp_iso_std': original_data_df.loc[:, "temp_iso"].std(),
		# 'v_wind_avg': original_data_df.loc[:, "v_wind_component"].mean(),
		# 'v_wind_std': original_data_df.loc[:, "v_wind_component"].std(),
		# 'u_wind_avg': original_data_df.loc[:, "u_wind_component"].mean(),
		# 'u_wind_std': original_data_df.loc[:, "u_wind_component"].std(),
		'temp_iso_min': original_data_df.loc[:, "temp_iso"].min(),
		'temp_iso_max': original_data_df.loc[:, "temp_iso"].max(),
		'v_wind_component_min': original_data_df.loc[:, "v_wind_component"].min(),
		'v_wind_component_max': original_data_df.loc[:, "v_wind_component"].max(),
		'u_wind_component_min': original_data_df.loc[:, "u_wind_component"].min(),
		'u_wind_component_max': original_data_df.loc[:, "u_wind_component"].max(),
		# 'model_id_avg': original_data_df.loc[:, "model_id"].mean(),
		# 'model_id_std': original_data_df.loc[:, "model_id"].std(),
		'model_id_min': original_data_df.loc[:, "model_id"].min(),
		'model_id_max': original_data_df.loc[:, "model_id"].max(),
		'delay_min': original_data_df.loc[:, "delay"].min(),
		'delay_max': original_data_df.loc[:, "delay"].max()
	}

# Create / Load latent vectors
latent_dimensions = len(train_set_df['Clusters'].unique())
observation_dimensions = len(features)
action_dimensions = len(actions)


# Normalization
obs_train = normalize_observations(obs_train_df, features, conf)
actions_train = normalize_actions(actions_train_df, conf)
clusters_train = cluster_train_df.values

# NDArrays
obs_train_grouped = obs_train.groupby('trajectory_ID')
actions_train_grouped = actions_train.groupby('trajectory_ID')
obs_train_list = []
actions_train_list = []
clusters_train_list = []
codes_train_list = []
traj_modes = {}
traj_cnt = 0
for (obs_train_values, act_train_values) in zip(obs_train_grouped, actions_train_grouped):
	traj_id = obs_train_values[0]
	obs = obs_train_values[1].drop(['trajectory_ID', 'Clusters'], axis=1).values
	act = act_train_values[1].drop(['trajectory_ID', 'Clusters'], axis=1).values
	clusters = obs_train_values[1].loc[:,['Clusters']].values

	codes = []
	for cluster in clusters:
		m = cluster[0]
		code = [0 for _ in range(latent_dimensions)]
		code[m] = 1
		codes.append(code)
	codes = np.array(codes)

	if m in traj_modes:
		traj_modes[m].append(traj_cnt)
	else:
		traj_modes[m] = [traj_cnt]
	
	obs_train_list.append(obs)
	actions_train_list.append(act)
	clusters_train_list.append(clusters)
	
	codes_train_list.append(codes)
	traj_cnt += 1

obs_train_np = np.array(obs_train_list, dtype=object)
actions_train_np = np.array(actions_train_list, dtype=object)
clusters_train_np = np.array(clusters_train_list, dtype=object)
codes_train_np = np.array(codes_train_list, dtype=object)

exp_obs = np.concatenate(obs_train_np)
exp_actions = np.concatenate(actions_train_np)
exp_codes = np.concatenate(codes_train_list)
exp_modes = np.concatenate(clusters_train_np)
expert_traj = np.concatenate([exp_obs, exp_actions], axis=1)
###################

######## TORCH ##############
# Create nets
policy_net = Policy(observation_dimensions, action_dimensions, latent_dimensions, activation=config["activation"], logstd=config["logstd"])
value_net = Value(observation_dimensions, activation=config["activation"])
discrim_net = Discriminator(observation_dimensions, action_dimensions, latent_dimensions, activation=config["activation"])
selector_net = Selector(observation_dimensions, action_dimensions, latent_dimensions, activation=config["activation"])
discrim_criterion = nn.BCELoss()
sel_criterion = nn.CrossEntropyLoss()

# Load BC policy
#saved_timestamp = "Dec-09-2022_2038"	#cruising
#saved_timestamp = "Dec-09-2022_1311"	#ascending
#saved_timestamp = "Dec-07-2022_1820"	#descending
#policy_net.load_state_dict(torch.load("model_weights/bc_" + str(config["phase"]) + "_" + str(saved_timestamp) + ".pth"))

if not train_bcloning:
	'''
	Testing
	Load the weights of the last iteration of train (change path) into the policy and begin testing
	Create the necessary folders to save/load the weights
	'''
	policy_net.load_state_dict(torch.load(f"TripleGAIL/model_weights/bc_{phase}.pth"))


# CUDA
policy_net.to(device)
value_net.to(device)
discrim_net.to(device)
selector_net.to(device)

# Optimizers
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=3e-4)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=3e-4)
optimizer_selector = torch.optim.Adam(selector_net.parameters(), lr=3e-4)


# Environment
env = environment.Environment(args, features, conf, config)

agent = Agent(env, traj_modes, test_flight_num, policy_net, discrim_net, selector_net, device, custom_reward=expert_reward, path_size=config["path_size"], phase=phase)

if train_bcloning:
	behavioral_cloning(exp_obs, exp_actions, exp_codes, policy_net, latent_dimensions)

# Train
if train_method:
	for i_iter in range(config["epochs"]):
		print(f"Episode: {i_iter}")
		discrim_net.to(torch.device('cpu'))
		selector_net.to(torch.device('cpu'))
		batch_generator, batch_selector, log, generated_flight_ids = agent.collect_samples(config["samples_per_episode"], i_iter, latent_dimensions)
		discrim_net.to(device)
		selector_net.to(device)

		if i_iter == 0 or (i_iter > 1000 and i_iter % 50 == 0):
			torch.save(policy_net.state_dict(), f"TripleGAIL/model_weights/policy_{i_iter}" + ".pth")
			torch.save(selector_net.state_dict(), f"TripleGAIL/model_weights/selector_{i_iter}" + ".pth")
		
		t0 = time.time()
		update_params(batch_generator, batch_selector, i_iter, generated_flight_ids)
		t1 = time.time()
else:
	agent.evaluate(policy_net)




