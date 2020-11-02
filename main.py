import pandas as pd
from EV_data_analysis import EV
from TOU_analysis_and_prediction import TOU
from charging_recommendation import charging_recommendation
from pandas.tseries.offsets import DateOffset

file = '2012-03-01.csv'
subdir = '1119055_1'

# file = '2012-05-22.csv'
# subdir = '1035198_1'

EV_obj = EV(file, subdir)

EV_obj.soc_over_time()

y = ['P_electric_motor', 'speed_mps', 'P_regen', 'n_rb', 'soc', 'P_total']
file_name = ['energy_consumption.png', 'speed_profile.png', 'energy_consumption_with_regen.png',
             'n_rb.png', 'soc.png', 'total_energy_conumption.png']
EV_obj.graph_plotter(y=y, file_name=file_name, subdir=subdir, date=file.strip('.csv'))

cols_to_drop = ['cycle_sec', 'timestep', 'speed_mph', 'accel_meters_ps', 'speed_mps',
                 'accel_mps2', 'P_wheels', 'P_electric_motor', 'n_rb', 'P_regen']

P_total = EV_obj.data.copy()
P_total = P_total.drop(columns=cols_to_drop)
P_total = P_total.set_index('timestamp')

##########################################################################################

file = 'agile_rates_2019.xlsx'
TOU_obj = TOU(file)

# uncomment the line below if you're running for the first time
# results = TOU_obj.create_and_fit_model()

start_time = pd.to_datetime('2019-01-31 00:00:00')
end_time = pd.to_datetime('2019-01-31 23:30:00')

P_total = P_total.set_index(P_total.index
                            + DateOffset(days=(start_time.floor(freq='D')
                                               - P_total.iloc[0].name.floor(freq='D')).days))
pred = TOU_obj.predict_and_compare(start_time, end_time)

##########################################################################################

#just runs the recommendation system that hasn't considered EV SOC

charging_recom_obj = charging_recommendation(P_total, pred)

threshold = 0
charger_power = 3e3
print(charging_recom_obj.recommend(threshold=threshold, charger_power=charger_power))
