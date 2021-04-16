import pandas as pd
from EV_data_analysis import EV
from TOU_analysis_and_prediction import TOU
from charging_recommendation import charging_recommendation
from pandas.tseries.offsets import DateOffset
from simulation import Simulation
import json
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

drive_cycle_file = 'Device12_formatted_filtered.csv'
tou_file = 'full_data.csv'

# for Windows
# drive_cycle_subdir = 'data\yun_solution_drive_cycle'
# tou_subdir = 'data\TOU_Data'
# json_path = ".\\utils\\user_config.json"

# for Mac
drive_cycle_subdir = 'data/yun_solution_drive_cycle'
tou_subdir = 'data/TOU_Data'
json_path = "./utils/user_config.json"

saving_TOU_df = pd.DataFrame([])
average_cost = []
TOU_threshold = []
Max_threshold = 1
increments = 0.25 #p/kWh
temp = int(1/increments)

for TOU in range(0,(Max_threshold*temp)+1):
    with open(json_path) as f:
        config_dict = json.load(f)
    print(config_dict)
    config_dict['Charge_level'] = 3760
    config_dict['TOU_threshold'] = TOU * increments
    TOU_threshold.append(TOU * increments)
    print(config_dict)
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    simulation_obj = Simulation(drive_cycle_file=drive_cycle_file, drive_cycle_subdir=drive_cycle_subdir, config_path=json_path, tou_file=tou_file, tou_subdir=tou_subdir, train_tou=False)
    for x in range(0, 7):
        simulation_obj.plugged_in()
        simulation_obj.trigger_discharge()
    total_energy_bought = sum(simulation_obj.energy_bought)
    energy_bought_df = []#simulation_obj.energy_bought.copy()
    energy_bought_df.append(total_energy_bought)
    total_energy_cost = sum(simulation_obj.energy_cost)
    energy_cost_df = []#simulation_obj.energy_cost
    energy_cost_df.append(total_energy_cost)
    total_charge_time_df = sum(simulation_obj.charge_time)
    print('Total charge time : {}'.format(total_charge_time_df))
    print('energy bought : {}'.format(energy_bought_df))
    print('energy cost : {}'.format(energy_cost_df))
    print('Total deficit (error): ',simulation_obj.ev_obj.deficit)
    # simulation_obj.charging_schedule.to_csv('charging_schedule.csv')
    average_energy_cost = total_energy_cost/total_energy_bought
    average_cost.append(average_energy_cost)


saving_TOU_df['TOU_threshold'] = TOU_threshold
saving_TOU_df['Average_cost'] = average_cost
saving_TOU_df.to_csv('TOU_vs_savings.csv')