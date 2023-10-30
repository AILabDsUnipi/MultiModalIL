from geo.sphere import distance
import numpy as np
import copy
import pandas as pd

class Environment(object):
	def __init__(self, args, features, conf, config):
		# Scenario configuration
		#self.mbr_top_left = (2.5,50.28)							# top-left borders
		#self.mbr_bot_right = (26.8,40.35)							# bottom-right borders
		# self.mbr_top_left = (1.9,50.0)							# full
		# self.mbr_bot_right = (29.35,40.7)							# full
		#self.mbr_top_left = (1.9,49.17)								# ascending
		#self.mbr_bot_right = (3.58,48.26)							# ascending
		self.mbr_top_left = (26.28,42.0)							# descending
		self.mbr_bot_right = (29.3,40.79)							# descending
		self.td = 20 												# delta timestep
		self.state = [2.54692811347671, 49.009754952015, 400.0]		# initial state
		self.norm_state = []
		self.destination = (28.81, 40.97)							# destination airport
		self.done = False
		self.destination_distance = 5000							# Termination point before reaching the airport
		self.max_alt = 41000										# experts' max altitude
		self.sub_trajectory_summary = []
		self.conf = conf
		self.args = args
		self.features = features
		self.path_size = config["path_size"]

		# Paths
		grib_path = "paris_weather_grib"
		weather_path = "weather_gmt.npy"
		phase = config["phase"]
		train_starting_points_path = f"{phase}_test_sp.csv"
		test_starting_points_path = f"{phase}_test_sp.csv"
		train_ending_points_path = f"{phase}_test_ep.csv"
		test_ending_points_path = f"{phase}_test_ep.csv"

		df_columns = features + ['trajectory_ID', 'Clusters']

		df = pd.read_csv(args.input_dir +
						 train_starting_points_path)[df_columns]

		test_df = pd.read_csv(args.input_dir +
						 test_starting_points_path)[df_columns]

		self.train_starting_points = df.values.tolist()
		self.test_starting_points = test_df.values.tolist()
		
		self.train_end_df = pd.read_csv(args.input_dir +
						 train_ending_points_path)[['trajectory_ID', 'longitude', 'latitude']]

		self.test_end_df = pd.read_csv(args.input_dir +
						 test_ending_points_path)[['trajectory_ID', 'longitude', 'latitude']]
		
		
		print()
		print("Loading the weather...")
		# weather_df = pd.read_csv(args.input_dir + grib_path +'.csv')
		# self.shape = []
		# for idx, col in enumerate(['timestamp','longitude','latitude','altitude']):
		# 	self.shape.append(weather_df[col].nunique())
		# self.shape.append(3)
		self.shape = [521, 117, 46, 24, 3]

		self.min_timestamp = 1515981600
		self.min_lon = 1.25
		self.min_lat = 39.750008
		self.min_alt = 363.6446

		self.ts_list = [1515981600, 1515985200, 1515992400, 1515996000, 1515999600, 1516003200, 1516006800, 1516010400, 1516014000, 1516017600, 1516021200, 1516024800, 1516028400, 1516032000, 1516035600, 1516039200, 1516042800, 1516046400, 1516050000, 1516082400, 1516086000, 1516089600, 1516093200, 1516096800, 1516100400, 1516104000, 1516107600, 1516111200, 1516114800, 1516118400, 1516122000, 1516125600, 1516129200, 1516132800, 1516136400, 1516168800, 1516172400, 1516176000, 1516179600, 1516183200, 1516186800, 1516190400, 1516194000, 1516197600, 1516201200, 1516204800, 1516208400, 1516212000, 1516215600, 1516219200, 1516222800, 1516226400, 1516230000, 1516255200, 1516258800, 1516262400, 1516266000, 1516269600, 1516273200, 1516276800, 1516280400, 1516284000, 1516287600, 1516291200, 1516294800, 1516298400, 1516302000, 1516305600, 1516309200, 1516345200, 1516348800, 1516352400, 1516356000, 1516359600, 1516363200, 1516366800, 1516370400, 1516374000, 1516377600, 1516381200, 1516384800, 1516388400, 1516392000, 1516395600, 1516399200, 1516428000, 1516431600, 1516435200, 1516438800, 1516442400, 1516446000, 1516449600, 1516453200, 1516456800, 1516460400, 1516464000, 1516467600, 1516471200, 1516474800, 1516478400, 1516482000, 1516485600, 1516489200, 1516518000, 1516521600, 1516525200, 1516528800, 1516532400, 1516536000, 1516539600, 1516543200, 1516546800, 1516550400, 1516554000, 1516557600, 1516561200, 1516564800, 1523836800, 1523840400, 1523858400, 1523862000, 1523865600, 1523869200, 1523872800, 1523876400, 1523880000, 1523883600, 1523887200, 1523890800, 1523894400, 1523898000, 1523901600, 1523905200, 1523908800, 1523912400, 1523916000, 1523919600, 1523923200, 1523944800, 1523948400, 1523952000, 1523955600, 1523959200, 1523962800, 1523966400, 1523970000, 1523973600, 1523977200, 1523980800, 1523984400, 1523988000, 1523991600, 1523995200, 1523998800, 1524002400, 1524006000, 1524009600, 1524031200, 1524034800, 1524038400, 1524042000, 1524045600, 1524049200, 1524052800, 1524056400, 1524060000, 1524063600, 1524067200, 1524070800, 1524074400, 1524078000, 1524081600, 1524085200, 1524088800, 1524117600, 1524121200, 1524124800, 1524128400, 1524132000, 1524135600, 1524139200, 1524142800, 1524146400, 1524150000, 1524153600, 1524157200, 1524160800, 1524164400, 1524168000, 1524171600, 1524175200, 1524178800, 1524182400, 1524204000, 1524207600, 1524211200, 1524214800, 1524218400, 1524222000, 1524225600, 1524229200, 1524232800, 1524236400, 1524240000, 1524243600, 1524247200, 1524250800, 1524254400, 1524258000, 1524261600, 1524265200, 1524268800, 1524272400, 1524276000, 1524279600, 1524283200, 1524290400, 1524294000, 1524297600, 1524301200, 1524304800, 1524308400, 1524312000, 1524315600, 1524319200, 1524322800, 1524326400, 1524330000, 1524333600, 1524337200, 1524340800, 1524344400, 1524348000, 1524351600, 1524355200, 1524376800, 1524380400, 1524384000, 1524387600, 1524391200, 1524394800, 1524398400, 1524402000, 1524405600, 1524409200, 1524412800, 1524416400, 1524420000, 1524423600, 1524427200, 1524430800, 1524434400, 1524438000, 1524441600, 1534107600, 1534111200, 1534114800, 1534118400, 1534140000, 1534143600, 1534147200, 1534150800, 1534154400, 1534158000, 1534161600, 1534165200, 1534168800, 1534172400, 1534176000, 1534179600, 1534183200, 1534186800, 1534190400, 1534194000, 1534197600, 1534201200, 1534204800, 1534226400, 1534230000, 1534233600, 1534237200, 1534240800, 1534244400, 1534248000, 1534251600, 1534255200, 1534258800, 1534262400, 1534266000, 1534269600, 1534273200, 1534276800, 1534280400, 1534284000, 1534287600, 1534291200, 1534312800, 1534316400, 1534320000, 1534323600, 1534327200, 1534330800, 1534334400, 1534338000, 1534341600, 1534345200, 1534348800, 1534352400, 1534356000, 1534359600, 1534363200, 1534366800, 1534370400, 1534374000, 1534377600, 1534399200, 1534402800, 1534406400, 1534410000, 1534413600, 1534417200, 1534420800, 1534424400, 1534428000, 1534431600, 1534435200, 1534438800, 1534442400, 1534446000, 1534449600, 1534453200, 1534456800, 1534460400, 1534464000, 1534485600, 1534489200, 1534492800, 1534496400, 1534500000, 1534503600, 1534507200, 1534510800, 1534514400, 1534518000, 1534521600, 1534525200, 1534528800, 1534532400, 1534536000, 1534539600, 1534543200, 1534546800, 1534550400, 1534572000, 1534575600, 1534579200, 1534582800, 1534586400, 1534590000, 1534593600, 1534597200, 1534600800, 1534604400, 1534608000, 1534611600, 1534615200, 1534618800, 1534622400, 1534626000, 1534629600, 1534633200, 1534636800, 1534658400, 1534662000, 1534665600, 1534669200, 1534672800, 1534676400, 1534680000, 1534683600, 1534687200, 1534690800, 1534694400, 1534698000, 1534701600, 1534705200, 1534708800, 1534712400, 1534716000, 1534719600, 1534723200, 1544400000, 1544425200, 1544428800, 1544432400, 1544436000, 1544439600, 1544443200, 1544446800, 1544450400, 1544454000, 1544457600, 1544461200, 1544464800, 1544468400, 1544472000, 1544475600, 1544479200, 1544482800, 1544486400, 1544508000, 1544511600, 1544515200, 1544518800, 1544522400, 1544526000, 1544529600, 1544533200, 1544536800, 1544540400, 1544544000, 1544547600, 1544551200, 1544554800, 1544558400, 1544562000, 1544565600, 1544569200, 1544572800, 1544594400, 1544598000, 1544601600, 1544605200, 1544608800, 1544612400, 1544616000, 1544619600, 1544623200, 1544626800, 1544630400, 1544634000, 1544637600, 1544641200, 1544644800, 1544648400, 1544652000, 1544655600, 1544659200, 1544684400, 1544688000, 1544691600, 1544695200, 1544698800, 1544702400, 1544706000, 1544709600, 1544713200, 1544716800, 1544720400, 1544724000, 1544727600, 1544731200, 1544734800, 1544738400, 1544742000, 1544745600, 1544770800, 1544774400, 1544778000, 1544781600, 1544785200, 1544788800, 1544792400, 1544796000, 1544799600, 1544803200, 1544806800, 1544810400, 1544814000, 1544817600, 1544821200, 1544824800, 1544828400, 1544832000, 1544835600, 1544853600, 1544857200, 1544860800, 1544864400, 1544868000, 1544871600, 1544875200, 1544878800, 1544882400, 1544886000, 1544889600, 1544893200, 1544896800, 1544900400, 1544904000, 1544907600, 1544911200, 1544914800, 1544918400, 1544943600, 1544947200, 1544950800, 1544954400, 1544958000, 1544961600, 1544965200, 1544968800, 1544972400, 1544976000, 1544979600, 1544983200, 1544986800, 1544990400, 1544994000, 1544997600, 1545001200]
		self.lon_list = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5, 21.75, 22.0, 22.25, 22.5, 22.75, 23.0, 23.25, 23.5, 23.75, 24.0, 24.25, 24.5, 24.75, 25.0, 25.25, 25.5, 25.75, 26.0, 26.25, 26.5, 26.75, 27.0, 27.25, 27.5, 27.75, 28.0, 28.25, 28.5, 28.75, 29.0, 29.25, 29.5, 29.75, 30.0, 30.25]
		self.lat_list = [39.750008, 40.000008, 40.250008, 40.500008, 40.750008, 41.000008, 41.250008, 41.500008, 41.750008, 42.000008, 42.250008, 42.500008, 42.750008, 43.000008, 43.250008, 43.500008, 43.750008, 44.000008, 44.250008, 44.500008, 44.750008, 45.000008, 45.250008, 45.500008, 45.750008, 46.000008, 46.250008, 46.500008, 46.750008, 47.000008, 47.250008, 47.500008, 47.750008, 48.000008, 48.250008, 48.500008, 48.750008, 49.000008, 49.250008, 49.500008, 49.750008, 50.000008, 50.250008, 50.500008, 50.750008, 51.000008]
		self.alt_list = [363.6446, 1060.5262, 1772.0302, 2498.86, 3241.7744, 4001.5908, 4779.1934, 5575.5415, 6391.675, 7228.728, 8087.9385, 9878.385, 11775.57, 13794.855, 15955.335, 18281.182, 20803.67, 23564.346, 26620.215, 30052.74, 33984.703, 36195.496, 100923.22, 106414.93]

		with open(args.input_dir + weather_path, 'rb') as f:
			self.weather_np = np.load(f)

		print("Weather loaded!")
		print()

	def get_weather(self, lon, lat, alt, t):		
		ts_idx = np.abs(np.array(self.ts_list) - t).argmin()
		lon_idx = round((lon - self.min_lon) / 0.25)
		lat_idx = round((lat - self.min_lat) / 0.25)
		alt_idx = np.abs(np.array(self.alt_list) - alt).argmin()

		if lon_idx >= self.shape[1] or lat_idx >= self.shape[2] or \
					alt_idx >= self.shape[3] or ts_idx >= self.shape[0]:
			
			self.done = True
			return [-11]*4
		
		return self.weather_np[ts_idx, lon_idx, lat_idx, alt_idx]

	def next_starting_point(self, p):
		s_point = self.train_starting_points[p]

		return s_point

	def next_testing_point(self, p):
		s_point = self.test_starting_points[p]

		return s_point

	def reset(self, p):
		p=0
		self.state = copy.deepcopy(self.next_starting_point(p))
		code = self.state[-1]
		self.state = self.state[:-1]
		traj_id = self.state[-1]
		self.state = self.state[:-1]
		self.target_lon = self.train_end_df[self.train_end_df.trajectory_ID == traj_id]['longitude'].iloc[0]
		self.target_lat = self.train_end_df[self.train_end_df.trajectory_ID == traj_id]['latitude'].iloc[0]
		self.destination = (self.target_lon, self.target_lat)
		
		self.norm_state = []
		for i, f in enumerate(self.features):
			f_avg = f + '_avg'
			f_std = f + '_std'
			f_max = f + '_max'
			f_min = f + '_min'

			if f_avg in self.conf:
				self.norm_state.append((self.state[i] - self.conf[f_avg]) / self.conf[f_std])
			else:
				self.norm_state.append((self.state[i] - self.conf[f_min]) / (self.conf[f_max] - self.conf[f_min]))

			if bool(self.args.model_features) is True and f == 'model_id':
				self.model_id = self.state[i]

			if bool(self.args.delay_features) is True and f == 'delay':
				self.delay = self.state[i]

		self.done = False
		
		return self.state, self.norm_state, code, traj_id

	def reset_test(self, p):
		p=0
		self.state = copy.deepcopy(self.next_testing_point(p))
		code = self.state[-1]
		self.state = self.state[:-1]
		traj_id = self.state[-1]
		self.state = self.state[:-1]

		self.target_lon = self.test_end_df[self.test_end_df.trajectory_ID == traj_id]['longitude'].iloc[0]
		self.target_lat = self.test_end_df[self.test_end_df.trajectory_ID == traj_id]['latitude'].iloc[0]
		self.destination = (self.target_lon, self.target_lat)
		
		self.norm_state = []
		for i, f in enumerate(self.features):
			f_avg = f + '_avg'
			f_std = f + '_std'
			f_max = f + '_max'
			f_min = f + '_min'

			if f_avg in self.conf:
				self.norm_state.append((self.state[i] - self.conf[f_avg]) / self.conf[f_std])
			else:
				self.norm_state.append((self.state[i] - self.conf[f_min]) / (self.conf[f_max] - self.conf[f_min]))

			if bool(self.args.model_features) is True and f == 'model_id':
				self.model_id = self.state[i]

			if bool(self.args.delay_features) is True and f == 'delay':
				self.delay = self.state[i]


		self.done = False
		
		return self.state, self.norm_state, code, traj_id

	def revert_action(self, action):
		dlon_real = action[0] * self.conf['dlon_std'] + self.conf['dlon_avg']
		dlat_real = action[1] * self.conf['dlat_std'] + self.conf['dlat_avg']
		dalt_real = action[2] * self.conf['dalt_std'] + self.conf['dalt_avg']
		
		return [dlon_real, dlat_real, dalt_real]

	def step(self, action, tstep):
		# Revert action in real coordinates
		dlon, dlat, dalt = self.revert_action(action)

		if bool(self.args.time_features) is True:
			timestamp = self.state[3] + self.td

		# Move to the new coordinates
		point = [self.state[0] + dlon, self.state[1] + dlat]
		alt = self.state[2] + dalt
		alt = min(alt, self.max_alt)
		alt = max(alt, self.min_alt)
		self.state = [point[0], point[1], alt]

		if bool(self.args.time_features) is True:
			self.state.append(timestamp)

		# New spatio-temporal state | Find the weather at that state
		if bool(self.args.weather_features) is True:
			weather_vars = self.get_weather(point[0], point[1], alt, timestamp)
			self.state.extend(weather_vars)

		if bool(self.args.model_features) is True:
			self.state.append(self.model_id)

		if bool(self.args.delay_features) is True:
			self.state.append(self.delay)


		self.norm_state = []
		for i, f in enumerate(self.features):
			f_avg = f + '_avg'
			f_std = f + '_std'
			f_max = f + '_max'
			f_min = f + '_min'

			if f_avg in self.conf:
				self.norm_state.append((self.state[i] - self.conf[f_avg]) / self.conf[f_std])
			else:
				self.norm_state.append((self.state[i] - self.conf[f_min]) / (self.conf[f_max] - self.conf[f_min]))

		# Check if the flight trajectory has ended 
		flag = 0
		reward = 0
		if distance(point, self.destination) < self.destination_distance:
			#print("Reached destination")
			self.done = True
			flag = 1
		# elif point[0] > self.target_lon + 0.1:
		# 	self.done = True
		#	reward = -distance(point, self.destination)*self.args.env_reward_lambda
		elif point[0] < self.mbr_top_left[0] \
			or point[1] > self.mbr_top_left[1] \
			or point[0] > self.mbr_bot_right[0] \
			or point[1] < self.mbr_bot_right[1]:
			#print("Out of bounds")
			self.done = True
			#reward = -distance(point, self.destination)*self.args.env_reward_lambda
		elif alt > self.max_alt:
			#print("Max altitude error")
			self.done = True
			#reward = -distance(point, self.destination)*self.args.env_reward_lambda
		elif alt < 0:
			#print("Min altitude error")
			self.done = True
			#reward = -distance(point, self.destination)*self.args.env_reward_lambda
		elif tstep == self.path_size - 1:
			#print("End of steps")
			self.done = True
			#reward = -distance(point, self.destination)*self.args.env_reward_lambda
		
		# if bool(strtobool(self.args.weather_features)) is True and weather_vars[0] == 0 and weather_vars[1] == 0 and\
		# 	weather_vars[2] == 0:
		# 	#print("Weather Error")
		# 	self.done = True
		# 	reward = -distance(point, self.destination)*self.args.env_reward_lambda

		return self.state, self.norm_state, reward, self.done, flag