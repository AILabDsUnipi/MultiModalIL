import numpy as np
import cvxpy as cp
import random
import torch as T
import csv
from q_network import Net
import pandas as pd
from geo.sphere import distance
from tslearn.metrics import dtw, dtw_path
from replay_memory import ReplayMemory

'''
Coordination and weather conditions of the scenario - should change according to each scenario
'''
conf = {
		'mbr_top_left': (-10.5,61.5),
		'mbr_bot_right': (26.0,37.5),
		'td': 10,                                       #!!!!!!!!!
		'state': [24.95286, 60.321042, 940.0],
		'origin': (24.95286, 60.321042),                # Helsinki
		'destination': (-9.133832798, 38.771163582),    # Lisbon
		'destination_distance': 8000,
		'done': False,
		'max_alt': 40000,
		'grib': "HEL_01_grib",                          #!!!!!!!!!
		'iso_index': "HEL_01_isobaric_index",           #!!!!!!!!!
		'fir': 'HEL_01_fir'}                            #!!!!!!!!!


class ApprenticeshipLearning:
	def __init__(self, alpha, gamma, epsilon, mem_size, batch_size, actions, all_data, train_data, test_data, train_norm_data, test_norm_data, dx, dy, regressor, centroid, save_path, e_decay, scenario_name, scenario_day, cluster):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.actions = actions
		self.all_data = all_data
		self.train_data = train_data
		self.test_data = test_data
		self.train_norm_data = train_norm_data
		self.test_norm_data = test_norm_data
		self.regressor = regressor
		self.centroid = centroid
		self.e_decay = e_decay
		self.scenario_day = scenario_day
		self.scenario_name = scenario_name
		self.cluster = cluster
		self.num_expert = len(train_norm_data.groupby('trajectory_ID'))
		self.first_states = np.array(self.train_data.drop(['timestamp', 'Cluster'],axis=1).groupby('trajectory_ID').first())
		self.first_norm_states = np.array(self.train_norm_data.drop(['timestamp', 'Cluster'],axis=1).groupby('trajectory_ID').first())
		self.timestamps = np.array(self.train_data[['trajectory_ID', 'timestamp']].groupby('trajectory_ID').first())
		self.dx = dx
		self.dy = dy
		self.x_max = train_data['longitude'].max()
		self.x_min = train_data['longitude'].min()
		self.y_max = train_data['latitude'].max()
		self.y_min = train_data['latitude'].min()
		self.z_max =train_data['altitude'].max()
		self.z_min = train_data['altitude'].min()
		self.ugrd_min = train_data['UGRD'].min()
		self.ugrd_max = train_data['UGRD'].max()
		self.hgt_min = train_data['HGT'].min()
		self.hgt_max = train_data['HGT'].max()
		self.tmp_min = train_data['TMP'].min()
		self.tmp_max = train_data['TMP'].max()
		self.vgrd_min = train_data['VGRD'].min()
		self.vgrd_max = train_data['VGRD'].max()
		self.save_path = save_path

		self.coord_features = {}
		self.traj_features = {}
		self.expert_name = {}
		self.metar = {}
		exp = 0
		for key, values in self.train_norm_data.groupby('trajectory_ID'):
			features = np.array(values.drop(['trajectory_ID', 'timestamp', 'Cluster'],axis=1))
			self.expert_name[exp] = key
			self.traj_features[key] = []
			self.coord_features[key] = []
			self.metar[key] = []
			exp +=1
			for feature in features:
				self.traj_features[key].append(list(feature))
				self.coord_features[key].append(list(feature.T[0:3]))
				self.metar[key].append(list(feature.T[3:]))

		# ----------------METEO POINTS-----------------------
		'''
		Used later to compute features of meteorological variables
		More functions could be used here
		'''
		#meteo_features = 20
		# self.mu_meteo = np.array(met)
		# self.sigma_met = 1.0
		self.mu_meteo  = np.array(self.metar[centroid])
		self.step = 40
		self.mu_meteo = self.mu_meteo[::self.step]
		self.sigma_met =  1.0#0.1*np.sqrt(np.var(self.mu_meteo))

		# ----------------SPATIAL POINTS---------------------
		'''
		Used later to compute features of spatial variables
		More functions could be used here
		'''
		self.max_steps = 1500
		self.num_actions = len(actions)
		self.mu_points = np.array(self.coord_features[centroid])
		self.step = 15
		self.mu_points = self.mu_points[::self.step] 
		self.num_features = len(self.mu_points) + len(self.mu_meteo) 
		self.sigma = 1.0#0.1*np.sqrt(np.var(self.mu_points))
		print("Total Features (spatial + meteo): " + str(self.num_features))

		# ------------------WEATHER--------------------------
		'''
		Weather conditions according to scenario
		'''
		self.mbr_top_left = conf['mbr_top_left']
		self.mbr_bot_right = conf['mbr_bot_right']
		self.td = conf['td']
		self.state = conf['state']
		self.sub_trajectory = []
		self.norm_state = []
		self.destination = conf['destination']
		self.done = conf['done']
		self.max_alt = conf['max_alt']
		weather_df = pd.read_csv(str(scenario_name) + '/' + conf['grib']+'.csv')
		self.shape = []
		for idx, col in enumerate(['timestamp','longitude','latitude','altitude']):
			self.shape.append(weather_df[col].nunique())
		self.shape.append(4)
		print(self.shape)
		self.min_timestamp = weather_df['timestamp'].min()
		print(self.min_timestamp)
		self.min_lon = weather_df['longitude'].min()
		self.min_lat = weather_df['latitude'].min()
		self.weather_np = np.reshape(weather_df.drop(
			['longitude', 'latitude', 'altitude', 'timestamp'], axis=1
		).to_numpy(), self.shape)
		self.iso_index = pd.read_csv(str(scenario_name) + '/' + conf['iso_index'] + '.csv').to_numpy()
		fir_df = pd.read_csv(str(scenario_name) + '/' + conf['fir'] + '.csv')
		self.shape_fir = []
		for idx, col in enumerate(['longitude','latitude']):
			self.shape_fir.append(fir_df[col].nunique())
		self.shape_fir.append(2)
		print(self.shape_fir)
		print()
		self.count = 0
		self.fir_np = np.reshape(fir_df.drop(['longitude', 'latitude'], axis=1).to_numpy(), self.shape_fir)
		# ------------------------------------------------------

		self.input_dims = (1, self.num_features)
		self.memory = ReplayMemory(self.mem_size, self.input_dims, self.num_actions)
		self.q_network = Net(self.num_actions, self.mem_size, self.batch_size, self.alpha, self.gamma, self.num_features)
		self.q_target_network = Net(self.num_actions, self.mem_size, self.batch_size, self.alpha, self.gamma, self.num_features)

	def get_features(self, state):
		'''
		Features for spatial and meteorological variables
		'''
		spatial = state[0:3]
		meteo = state[3:]

		pos_features = []
		for mu in self.mu_points:
			feat = np.dot((spatial - mu), (spatial - mu))
			pos_features.append(feat)

		pos_features = np.array(pos_features)
		pos_features = pos_features / (2 * self.sigma ** 2)
		pos_features = np.exp(-pos_features)

		met_features = []
		for mu in self.mu_meteo:
			feat = np.dot((meteo - mu), (meteo - mu))
			met_features.append(feat)

		met_features = np.array(met_features)
		met_features = met_features / (2 * self.sigma_met ** 2)
		met_features = np.exp(-met_features)

		features = np.concatenate((pos_features, met_features), axis=0)

		return features

	def train(self, num_episodes):
		is_done = False
		
		# ----------------------IRL step-----------------------------------
		policy, is_done = self.calculate_features_expectation_dtw()
		policy = np.matrix([policy])
				
		expert_policy = self.calculate_expert_features_expectation()
		expert_policy = np.matrix([expert_policy])
		
		if is_done:
			W, status = self.calculate_weights(policy, expert_policy)
		else:
			W = np.random.rand(self.num_features) * -1.0
		#-------------------------------------------------------------

		success = 0
		succ_per = 0
		change_lr = False
		for episode in range(num_episodes):
			if episode % self.e_decay == 0 and episode != 0:
				self.epsilon = self.epsilon - 0.01

			if self.epsilon < 0.001:
				self.epsilon = 0.0

			final_state = []
			final_norm_state = []

			# Expert Trajectory
			expert_number = np.random.randint(0, self.num_expert)
			name = self.expert_name[expert_number]
			state = np.array(self.first_states[expert_number])
			norm_state = np.array(self.first_norm_states[expert_number])
			timestamp = int(self.timestamps[expert_number])
			final_state.append([state[0], state[1], state[2]])
			final_norm_state.append([norm_state[0], norm_state[1], norm_state[2]])
			
			done = False
			t = 0
			while not done:
				timestamp += 5

				features_vector = self.get_features(norm_state)
				action = self.choose_action(features_vector)
				next_state, next_norm_state = self.get_next_state(state, norm_state, action)
				
				# Regression
				input_longlat = T.tensor(np.array([next_norm_state[0:2]], dtype=np.float32))
				pred_alt = self.regressor.forward(input_longlat)
				next_norm_state[2] = pred_alt
				next_state[2] = pred_alt * (self.z_max - self.z_min) + self.z_min

				# Weather Condition
				weather = self.get_weather(next_state[0], next_state[1], next_state[2], timestamp)
				next_state, next_norm_state = self.apply_weather(next_state, next_norm_state, weather)

				next_features_vector = self.get_features(next_norm_state)

				reward = np.dot(W, features_vector)
			
				find, status = self.check_destination(next_state)

				if t == self.max_steps or status or find:			
					if status:
						reward = -10
					
					if find:
						success = success + 1
						succ_per = succ_per + 1

					done = True

				self.store_transition(features_vector, action, reward, next_features_vector, done)

				if self.epsilon == 0.0 and t % 10 == 0:
					self.q_network.adjust_lr()

				state = next_state
				norm_state = next_norm_state
				
				final_state.append([state[0], state[1], state[2]])
				final_norm_state.append([norm_state[0], norm_state[1], norm_state[2]])
				t += 1
				
			
			print("Episode: " + str(episode))
			

			if episode % 2 == 0:
				self.replace_target_network()

			if episode % 50 == 0:
				if self.epsilon == 0:
					path = self.save_path
					path_nn = path + str(episode)
					T.save(self.q_network.state_dict(), path_nn)
				
				status = "infeasible"
				temp_policy, is_done = self.calculate_features_expectation_dtw()
				if is_done:
					policy = self.add_feature_expectation(policy, temp_policy)

					while status=="infeasible":
						W, status = self.calculate_weights(policy, expert_policy)
						if status=="infeasible":
							policy = self.subtract_feature_expectation(policy)

					print("++++++++++++++++++++++")
					print("LATEST POLICY'S FEATURE EXPECTATION")
					print(policy[-1])
					print("EXPERT POLICY'S FEATURE EXPECTATION")
					print(expert_policy)
					print("WEIGHTS")
					print(W)
					print("++++++++++++++++++++++")

	def calculate_features_expectation_dtw(self):
		'''
		Calculation of the policy's features expectation
		Tip: include actions to the calculation of the features expectation
		'''
		features_expectation = np.zeros(self.num_features)
		global_features_expectation = np.zeros(self.num_features)
		N = self.num_expert - 1
		correct_traj = 0
		random_experts = random.sample(range(self.num_expert), N)
		print("Calculating Feature's Expectation...")
		for i in random_experts:	
			# Expert Trajectory
			expert_number = i
			name = self.expert_name[expert_number]
			state = np.array(self.first_states[expert_number])
			norm_state = np.array(self.first_norm_states[expert_number])
			timestamp = int(self.timestamps[expert_number])

			found_states = [list(norm_state)]
			generated_pos = [list(norm_state[0:3])]
			t = 0
			done = False
			success = False
			while not done:
				timestamp += 5
				t += 1
				
				features_vector = self.get_features(norm_state)

				action = self.choose_action(features_vector)
				
				next_state, next_norm_state = self.get_next_state(state, norm_state, action)
				# Regression
				input_xy = np.array([next_norm_state[0:2]], dtype=np.float32)
				input_XY = T.tensor(input_xy)
				pred_Z = self.regressor.forward(input_XY)
				next_norm_state[2] = pred_Z
				next_state[2] = pred_Z * (self.z_max - self.z_min) + self.z_min

				weather = self.get_weather(next_state[0], next_state[1], next_state[2], timestamp)
				next_state, next_norm_state = self.apply_weather(next_state, next_norm_state, weather)

				found_states.append(list(next_norm_state))
				generated_pos.append(list(next_norm_state[0:3]))

				find, status = self.check_destination(next_state)

				if t == self.max_steps or status or find:
					done = True
					if find:
						correct_traj += 1
						success = True
					

				state = next_state
				norm_state = next_norm_state

			if success == True:
				path = dtw_path(self.coord_features[name], generated_pos)
				expert_points = []
				ts = 0
				for p1, p2 in path[0]:
					if p1 not in expert_points:
						ts += 1
						expert_points.append(p1)
						features_vector = self.get_features(found_states[p2])
						features_expectation += (self.gamma ** ts) * np.array(features_vector)
		
		if correct_traj == 0:
			print("Found 0 correct trajectories!")
			global_features_expectation = global_features_expectation / N

			return global_features_expectation, False
		else:

			print("Found " + str(correct_traj) + " correct trajectories!")
			features_expectation = features_expectation / correct_traj

			return features_expectation, True

	def calculate_expert_features_expectation(self):
		'''
		Calculation of the expert's features expectation
		'''
		features_expectation = np.zeros(self.num_features)
		for key, values in self.train_norm_data.groupby('trajectory_ID'):
			t = 0
			features = np.array(values.drop(['trajectory_ID', 'timestamp', 'Cluster'],axis=1))
			for feature in features:
				t += 1
				features_vector = self.get_features(feature)

				features_vector = np.array(list(features_vector))
				features_expectation += (self.gamma ** t) * features_vector
		
		features_expectation = features_expectation / self.num_expert
		return features_expectation

	def add_feature_expectation(self, policy, temp_policy):
		policy = np.vstack([policy, temp_policy])
		return policy

	def subtract_feature_expectation(self, policy):
		policy = policy[:-1][:]
		return policy


	def get_weather(self, lon, lat, alt, t):
		'''
		Function that returns the weather variables for the specific (lon, lat, alt, t)
		Constants should change according to each scenario's specifications
		'''
		t_idx = int(((t + 10799) - self.min_timestamp) / 21600)
		lon_idx = int(((lon + 0.25) - self.min_lon) / 0.5)
		lat_idx = int(((lat + 0.25) - self.min_lat) / 0.5)
		idx = (np.abs(self.iso_index[:, 2] - alt)).argmin()
		alt_idx = 19 - int(self.iso_index[idx][0])

		if lon_idx >= self.shape[1] or lat_idx >= self.shape[2] or idx >= self.shape[3]\
			or t_idx >= self.shape[0]:
			print(t_idx, lon_idx, lat_idx, idx, alt_idx)
			# if t_idx < self.shape[3]:
			#     print('other than t')
			self.done = True
			# print('test')
			with open(myconfig['output_dir']+'/exp'+myconfig['exp']+'_env_log.csv', "a") as env_log:
				env_log.write(str(lon)+","+str(lat)+","+str(alt)+","+str(t)+"\n")
			return [-11]*4
		# alt_idx = int(self.iso_index[idx][0])
		return self.weather_np[t_idx, lon_idx, lat_idx, idx]

	def apply_weather(self, next_state, next_norm_state, weather):
		next_state[3:7] = weather

		next_norm_state[3] = (weather[0] - self.ugrd_min) / (self.ugrd_max - self.ugrd_min)
		next_norm_state[4] = (weather[1] - self.hgt_min) / (self.hgt_max - self.hgt_min)
		next_norm_state[5] = (weather[2] - self.tmp_min) / (self.tmp_max - self.tmp_min)
		next_norm_state[6] = (weather[3] - self.vgrd_min) / (self.vgrd_max - self.vgrd_min)

		return next_state, next_norm_state


	def get_next_state(self, state, norm_state, idx):
		'''
		Perform a transition step
		Constants should change according to the scenario
		'''
		theta = self.actions[idx]
		num_feat = 7
		next_state = np.zeros(num_feat)
		next_norm_state = np.zeros(num_feat)

		next_state[0] = state[0] + 1.25*self.dx * np.cos(theta*np.pi/180)
		next_state[1] = state[1] + 1.05*self.dy * np.sin(theta*np.pi/180)

		next_norm_state[0] = (next_state[0] - self.x_min) / (self.x_max - self.x_min)
		next_norm_state[1] = (next_state[1] - self.y_min) / (self.y_max - self.y_min)

		return next_state, next_norm_state


	def choose_action(self, state):
		if np.random.random() > self.epsilon:
			state_nn = T.tensor(state, dtype=T.float).to(self.q_network.device)
			actions_nn = self.q_network.forward(state_nn)
			return T.argmax(actions_nn).item()
		else:
			return random.randint(0, self.num_actions - 1)


	def calculate_weights(self, policy, expert_policy):
		'''
		Convex optimization to get the weights measured by the distance between
		policy's and experts' feature expectations
		'''
		w = cp.Variable(self.num_features)

		thres = 1e-02

		obj_func = cp.Minimize(cp.norm(w,2)**2)

		constraints = [(expert_policy-policy)*w >= thres]#, cp.norm(w,1)<=1]

		prob = cp.Problem(obj_func, constraints)
		prob.solve()


		if prob.status == "optimal":
			print("status:", prob.status)
			print("optimal value", prob.value)

			weights = np.squeeze(np.asarray(w.value))
			return weights, prob.status
		else:
			print("status:", prob.status)
			
			weights = np.zeros(self.num_features)
			return weights, prob.status


	def check_destination(self, state):
		alt = state[2]
		point = (state[0], state[1])
		dist = distance(point,  self.destination)

		status = False
		if point[0] < self.mbr_top_left[0] \
				or point[1] > self.mbr_top_left[1]\
				or point[0] > self.mbr_bot_right[0]\
				or point[1] < self.mbr_bot_right[1]\
				or alt > self.max_alt:
				#or alt < 0:

			status = True

		if dist < conf['destination_distance']:
			find = True
		else:
			find = False

		return find, status

	def sample_memory(self):
		state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

		states = T.tensor(state).to(self.q_network.device)
		rewards = T.tensor(reward).to(self.q_network.device)
		dones = T.tensor(done).to(self.q_network.device)
		actions = T.tensor(action).to(self.q_network.device)
		states_ = T.tensor(new_state).to(self.q_network.device)

		return states, actions, rewards, states_, dones

	def replace_target_network(self):
		self.q_target_network.load_state_dict(self.q_network.state_dict())

	def store_transition(self, state, action, reward, next_state, done):
		self.memory.store_transition(state, action, reward, next_state, done)

	def update(self):
		if self.memory.mem_cntr < self.batch_size:
			return
		epochs = 10

		for i in range(epochs):
			self.q_network.optimizer.zero_grad()

			states, actions, rewards, states_, dones = self.sample_memory()
			indices = np.arange(self.batch_size)

			q_pred = self.q_network.forward(states)[indices, actions]
			q_next = self.q_target_network.forward(states_)
			q_eval = self.q_network.forward(states_)

			max_actions = T.argmax(q_eval, dim=1)
			q_next[dones] = 0.0

			q_target = rewards + self.gamma*q_next[indices, max_actions]
			loss = self.q_network.loss(q_target, q_pred).to(self.q_network.device)
			loss.backward()

			self.q_network.optimizer.step()