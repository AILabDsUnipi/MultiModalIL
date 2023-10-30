import pandas as pd
import numpy as np

asc_df = pd.read_csv("datasets/paris_ascending_test.csv")
cru_df = pd.read_csv("datasets/paris_cruising_test.csv")
des_df = pd.read_csv("datasets/paris_descending_test.csv")

train_list = ['20180118_398434', '20180118_393120', '20180813_497346', '20180813_515259',
 '20180814_550324', '20180417_425547', '20180819_713857', '20180815_586026',
 '20180118_389618', '20180118_381023', '20180814_537144', '20180416_398334',
 '20180420_536158', '20180118_372059', '20180418_446232', '20180816_623840',
 '20180118_390347', '20180420_539517', '20180815_577715', '20180815_590261',
 '20180121_463530', '20180817_654901', '20180814_542043', '20180422_580023',
 '20180115_304386']

asc_df = asc_df[~asc_df['trajectory_ID'].isin(train_list)]
cru_df = cru_df[~cru_df['trajectory_ID'].isin(train_list)]
des_df = des_df[~des_df['trajectory_ID'].isin(train_list)]

asc_flights = asc_df['trajectory_ID'].nunique()
cru_flights = cru_df['trajectory_ID'].nunique()
des_flights = des_df['trajectory_ID'].nunique()

asc_df.to_csv('datasets/ascending_test.csv', index=False)
cru_df.to_csv('datasets/crusing_test.csv', index=False)
des_df.to_csv('datasets/descending_test.csv', index=False)

asc_df.groupby('trajectory_ID').head(1).to_csv('datasets/ascending_test_sp.csv', index=False)
cru_df.groupby('trajectory_ID').head(1).to_csv('datasets/cruising_test_sp.csv', index=False)
des_df.groupby('trajectory_ID').head(1).to_csv('datasets/descending_test_sp.csv', index=False)

asc_df.groupby('trajectory_ID').tail(1).to_csv('datasets/ascending_test_ep.csv', index=False)
cru_df.groupby('trajectory_ID').tail(1).to_csv('datasets/cruising_test_ep.csv', index=False)
des_df.groupby('trajectory_ID').tail(1).to_csv('datasets/descending_test_ep.csv', index=False)



# Train Test Split
'''
asc_df = pd.read_csv("datasets/paris_ascending_test_sp.csv")[['trajectory_ID', 'Clusters']]
cru_df = pd.read_csv("datasets/paris_cruising_test_sp.csv")[['trajectory_ID', 'Clusters']]
des_df = pd.read_csv("datasets/paris_descending_test_sp.csv")[['trajectory_ID', 'Clusters']]
asc_flights = asc_df['trajectory_ID'].nunique()
cru_flights = cru_df['trajectory_ID'].nunique()
des_flights = des_df['trajectory_ID'].nunique()

print(f"Ascending: {asc_flights} | Cruising: {cru_flights} | Descending: {des_flights}")


asc_grouped = asc_df.groupby('trajectory_ID')
cru_grouped = cru_df.groupby('trajectory_ID')
des_grouped = des_df.groupby('trajectory_ID')


flight_modes = {}
asc_modes = {}
traj_cnt = 0
for (asc, cru, des) in zip(asc_grouped, cru_grouped, des_grouped):
	asc_cluster = asc[1].loc[:,['Clusters']].values[0][0]
	cru_cluster = cru[1].loc[:,['Clusters']].values[0][0]
	des_cluster = des[1].loc[:,['Clusters']].values[0][0]
	if cru_cluster == 5 and asc_cluster == 6:
		#flight_modes[cru[0]] = [des_cluster]

		if des_cluster in asc_modes:
			asc_modes[des_cluster].append(des[0])
		else:
			asc_modes[des_cluster] = [des[0]]

		traj_cnt += 1

train_list = np.random.choice(asc_modes[0], 25, replace=False)

test_list = []
for tr in asc_modes[0]:
	if tr not in train_list:
		test_list.append(tr)

print(train_list)
print(test_list)
print(len(train_list))
print(len(test_list))
'''