from utils.replay_memory import Memory
from utils.torch import *
import math
import time
import os
import multiprocessing
import random

def collect_samples(pid, queue, env, generated_flight_ids, policy, discriminator, posterior, custom_reward,
					mean_action, path_size, min_batch_size, episode, num_modes, phase, test_flight_num):
	'''
	Collect policy samples using multiprocessing
	'''
	if pid > 0:
		torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
		if hasattr(env, 'np_random'):
			env.np_random.seed(env.np_random.randint(5000) * pid)
		if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
			env.env.np_random.seed(env.env.np_random.randint(5000) * pid)

	if episode % 25 == 0:
		evaluate(env, policy, episode, num_modes, path_size)
	
	log = dict()
	memory = Memory()
	not_enough_samples = True
	num_steps = 0
	total_reward = 0
	total_c_reward = 0
	num_episodes = 0

	random.shuffle(generated_flight_ids)
	while not_enough_samples:
		'''
		Main loop for generating trajectories
		'''
		random_traj = generated_flight_ids[num_episodes]

		raw_state, state, cluster, traj_id = env.reset(random_traj)
		
		onehot_code = np.zeros([1, num_modes])
		onehot_code[:, cluster] = 1
		code = onehot_code.tolist()[0]
		code_var = tensor(onehot_code, dtype=torch.float32)
		for t in range(path_size):
			state_var = tensor(state, dtype=torch.float32).unsqueeze(0)
			
			with torch.no_grad():
				if mean_action:
					action = policy(state_var, code_var)[0][0].numpy()
				else:
					action = policy.select_action(state_var, code_var, episode)[0].numpy()

			action = action.astype(np.float32)
			next_raw_state, next_state, _, done, flag = env.step(action, t)
			
			if custom_reward is not None:
				reward = custom_reward(state, action, cluster, discriminator, posterior)
				total_c_reward += reward

			mask = 0 if done else 1

			memory.push(state, action, code, cluster, mask, next_state, reward)

			if done:
				break

			state = next_state

		# log stats
		num_steps += (t + 1)
		num_episodes += 1

		if num_episodes == len(generated_flight_ids):
			num_episodes = 0
			random.shuffle(generated_flight_ids)

		if num_steps >= min_batch_size:
			not_enough_samples = False

	
	log['num_steps'] = num_steps
	log['num_episodes'] = num_episodes
	if custom_reward is not None:
		log['total_c_reward'] = total_c_reward
		log['avg_c_reward'] = total_c_reward / num_steps
	
	if queue is not None:
		queue.put([pid, memory, log])
	else:
		return memory, log

def evaluate(env, policy, episode, num_modes, path_size, phase, test_flight_num):
	'''
	Generate trajectories from the testing set
	'''
	save_path = f'InfoGAIL/experiments/evaluation_{phase}_{episode}.csv'
	with open(save_path, 'w') as results:
		results.write('traj_id,longitude,latitude,altitude\n')

		for test_traj in range(test_flight_num):

			raw_state, state, cluster, traj_id = env.reset_test(test_traj)
			code = cluster
			onehot_code = np.zeros([1, num_modes])
			onehot_code[:, code] = 1
			code_var = tensor(onehot_code, dtype=torch.float32)

			line = ""
			raw_state = raw_state[0:3]
			for ob in raw_state:
				line += "," + str(ob)
			results.write(str(traj_id) + line + "\n")

			for t in range(path_size):
				state_var = tensor(state, dtype=torch.float32).unsqueeze(0)
				with torch.no_grad():
					action = policy(state_var, code_var)[0][0].numpy()

				action = action.astype(np.float32)
				next_raw_state, next_state, _, done, flag = env.step(action, t)

				next_raw_state = next_raw_state[0:3]
				if not done :
					line = ""
					
					for ob in next_raw_state:
						line += "," + str(ob)
					results.write(str(traj_id) + line + "\n")
				
				if done:
					break

				state = next_state

				
class Agent:
	def __init__(self, env, traj_modes, policy, discriminator, posterior, device, custom_reward=None, num_threads=1, path_size=0, phase="cruising"):
		self.env = env
		self.policy = policy
		self.discriminator = discriminator
		self.posterior = posterior
		self.device = device
		self.custom_reward = custom_reward
		self.traj_modes = traj_modes
		self.num_threads = 1
		self.path_size = path_size
		self.phase = phase

	def collect_samples(self, min_batch_size, episode, mean_action=False):
		t_start = time.time()
		to_device(torch.device('cpu'), self.policy)
		thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
		queue = multiprocessing.Queue()
		workers = []


		###### DETERMINE WHICH TRAJECTORIES TO USE
		generated_flight_ids = []
		for k in self.traj_modes:
			if k == 5 and self.phase == "cruising":
				random_flights = np.random.choice(len(self.traj_modes[k]), 10, replace=False)
				generated_flight_ids.extend(random_flights)
			else:
				generated_flight_ids.extend(self.traj_modes[k])
		###########################################

		for i in range(self.num_threads-1):
			worker_args = (i+1, queue, self.env, generated_flight_ids, self.policy, self.discriminator, self.posterior, self.custom_reward, mean_action,
						   self.path_size, thread_batch_size, episode)
			workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
		for worker in workers:
			worker.start()

		memory, log = collect_samples(0, None, self.env, generated_flight_ids, self.policy, self.discriminator, self.posterior, self.custom_reward, mean_action,
									  self.path_size, thread_batch_size, episode)

		worker_logs = [None] * len(workers)
		worker_memories = [None] * len(workers)
		for _ in workers:
			pid, worker_memory, worker_log = queue.get()
			worker_memories[pid - 1] = worker_memory
			worker_logs[pid - 1] = worker_log
		for worker_memory in worker_memories:
			memory.append(worker_memory)
		batch = memory.sample()
		if self.num_threads > 1:
			log_list = [log] + worker_logs
			log = merge_log(log_list)
		to_device(self.device, self.policy)
		t_end = time.time()
		log['sample_time'] = t_end - t_start
		log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
		log['action_min'] = np.min(np.vstack(batch.action), axis=0)
		log['action_max'] = np.max(np.vstack(batch.action), axis=0)

		return batch, log, generated_flight_ids