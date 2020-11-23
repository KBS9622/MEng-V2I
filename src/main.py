import pandas as pd
from src.EV_data_analysis import EV
from src.TOU_analysis_and_prediction import TOU
from src.charging_recommendation import charging_recommendation
from pandas.tseries.offsets import DateOffset

##########################################################################################

file = 'agile_rates_2019.xlsx'
TOU_obj = TOU(file)

# uncomment the line below if you're running for the first time
# results = TOU_obj.create_and_fit_model()

start_time = pd.to_datetime('2019-01-31 00:00:00')
end_time = pd.to_datetime('2019-01-31 23:30:00')


pred = TOU_obj.predict_and_compare(start_time, end_time)
print(pred)

##########################################################################################

file = '2012-03-01.csv'
file2 = '2012-03-02.csv'
subdir = '1119055_1'

EV_obj = EV(file, subdir)
EV_obj2 = EV(file2, subdir)

EV_obj.soc_over_time()
EV_obj2.soc_over_time()

y = ['P_electric_motor', 'speed_mps', 'P_regen', 'n_rb', 'soc', 'P_total']
file_name = ['energy_consumption.png', 'speed_profile.png', 'energy_consumption_with_regen.png',
             'n_rb.png', 'soc.png', 'total_energy_conumption.png']
EV_obj.graph_plotter(y=y, file_name=file_name, subdir=subdir, date=file.strip('.csv'))
EV_obj2.graph_plotter(y=y, file_name=file_name, subdir=subdir, date=file2.strip('.csv'))

cols_to_drop = ['cycle_sec', 'timestep', 'speed_mph', 'accel_meters_ps', 'speed_mps',
                 'accel_mps2', 'P_wheels', 'P_electric_motor', 'n_rb', 'P_regen']

P_total = EV_obj.data.copy()
P_total = P_total.drop(columns=cols_to_drop)
P_total = P_total.set_index('timestamp')
P_total = P_total.set_index(P_total.index
                            + DateOffset(days=(start_time.floor(freq='D')
                                               - P_total.iloc[0].name.floor(freq='D')).days))
print('PTOTAL:\n{}'.format(P_total))

P_total2 = EV_obj2.data.copy()
P_total2 = P_total2.drop(columns=cols_to_drop)
P_total2 = P_total2.set_index('timestamp')
P_total2 = P_total2.set_index(P_total2.index
                            + DateOffset(days=(start_time.floor(freq='D')
                                               - P_total2.iloc[0].name.floor(freq='D')).days))
print('PTOTAL2:\n{}'.format(P_total2))
##########################################################################################

#just runs the recommendation system that hasn't considered EV SOC

# charging_recom_obj = charging_recommendation(P_total, pred)

# json_path = 'user_config.json'
# print(charging_recom_obj.recommend(json_path))
