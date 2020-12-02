import pandas as pd
from EV_data_analysis import EV
from TOU_analysis_and_prediction import TOU
from charging_recommendation import charging_recommendation
from pandas.tseries.offsets import DateOffset
from simulation import Simulation
pd.options.mode.chained_assignment = None
##########################################################################################

# file3 = 'full_data.csv'
# subdir_TOU = 'data/TOU_Data'
# TOU_obj = TOU(file3, subdir=subdir_TOU)

# # # uncomment the line below if you're running for the first time
# results = TOU_obj.create_and_fit_model()

# start_time1 = pd.to_datetime('2019-01-31 00:00:00')
# end_time1 = pd.to_datetime('2019-01-31 23:30:00')
# start_time = pd.to_datetime('2019-02-01 00:00:00')
# end_time = pd.to_datetime('2019-02-01 23:30:00')


# pred = TOU_obj.predict_and_compare(start_time1, end_time)
# # # hi = TOU_obj.predict_and_compare(start_time, end_time)
# print(pred)

# ##########################################################################################

# file1 = 'Device12_26_9_17.csv'
# file2 = 'Device12_25_9_17.csv'
# subdir = 'data/yun_solution_drive_cycle'

# EV_obj = EV(file1, subdir)
# EV_obj2 = EV(file2, subdir)

# EV_obj.soc_over_time()
# EV_obj2.soc_over_time()

# y = ['P_electric_motor', 'speed_mps', 'P_regen', 'n_rb', 'soc', 'P_total']
# file_name = ['energy_consumption.png', 'speed_profile.png', 'energy_consumption_with_regen.png',
#              'n_rb.png', 'soc.png', 'total_energy_conumption.png']
# EV_obj.graph_plotter(y=y, file_name=file_name, subdir=subdir, date=file1.strip('.csv'))
# EV_obj2.graph_plotter(y=y, file_name=file_name, subdir=subdir, date=file2.strip('.csv'))

# # cols_to_drop = ['cycle_sec', 'timestep', 'speed_mph', 'accel_meters_ps', 'speed_mps',
# #                  'accel_mps2', 'P_wheels', 'P_electric_motor', 'n_rb', 'P_regen']

# cols_to_drop = ['speed_mps', 'accel_mps2', 'P_wheels', 'P_electric_motor', 'n_rb', 'P_regen']

# P_total = EV_obj.data.copy()
# P_total = P_total.drop(columns=cols_to_drop)
# P_total = P_total.set_index('timestamp')
# P_total = P_total.set_index(P_total.index
#                             + DateOffset(days=(start_time.floor(freq='D')
#                                                - P_total.iloc[0].name.floor(freq='D')).days))
# # print('PTOTAL:\n{}'.format(P_total))

# P_total2 = EV_obj2.data.copy()
# P_total2 = P_total2.drop(columns=cols_to_drop)
# P_total2 = P_total2.set_index('timestamp')
# P_total2 = P_total2.set_index(P_total2.index
#                             + DateOffset(days=(start_time1.floor(freq='D')
#                                                - P_total2.iloc[0].name.floor(freq='D')).days))
# # print('PTOTAL2:\n{}'.format(P_total2))
# ##########################################################################################

# #just runs the recommendation system that hasn't considered EV SOC

# charging_recom_obj = charging_recommendation(P_total, pred, P_total2)
# print(charging_recom_obj.TOU_data.sort_values(by=['TOU']))

# json_path = '/Users/koeboonshyang/Documents/GitHub/MEng-V2I/utils/user_config.json'
# charging_recom_obj.update_user_config(json_path)
# print(charging_recom_obj.recommend())

drive_cycle_file = 'Device12_formatted.csv'
drive_cycle_subdir = 'data/yun_solution_drive_cycle'
tou_file = 'full_data.csv'
tou_subdir = 'data/TOU_Data'
json_path = "./utils/user_config.json"

simulation_obj = Simulation(drive_cycle_file=drive_cycle_file, drive_cycle_subdir=drive_cycle_subdir, config_path=json_path, tou_file=tou_file, tou_subdir=tou_subdir, train_tou=False)
simulation_obj.plugged_in()
# simulation_obj.trigger_discharge()
# simulation_obj.plugged_in()

# previous_ev_data = simulation_obj.get_ev_data(start_time=pd.to_datetime('2019-09-25 00:00:00'), end_time=pd.to_datetime('2019-09-25 23:59:59'))
# predicted_tou_data = simulation_obj.get_tou_data(start_time=pd.to_datetime('2019-09-25 00:00:00'), end_time=pd.to_datetime('2019-09-26 23:30:00'))
# ev_consumption_data = simulation_obj.get_ev_data(start_time=pd.to_datetime('2019-09-26 00:00:00'), end_time=pd.to_datetime('2019-09-26 23:59:59'))
# # recommendation_obj = charging_recommendation(ev_consumption_data, predicted_tou_data, previous_ev_data)
# print(previous_ev_data.iloc[-1, :].name)
# simulation_obj.tou_obj.time_idx_TOU_price.columns = ['TOU']
# print(simulation_obj.tou_obj.time_idx_TOU_price.iloc[0,:])
# print(hi)
# TOU_data = predicted_tou_data.loc[previous_ev_data.iloc[-1, :].name:, :]

