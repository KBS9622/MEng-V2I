import pandas as pd
from EV_data_analysis import EV
from TOU_analysis_and_prediction import TOU
from charging_recommendation import charging_recommendation
from pandas.tseries.offsets import DateOffset
from datetime import datetime, timedelta

start_time = pd.to_datetime('2019-01-31 00:00:00')
end_time = pd.to_datetime('2019-01-31 23:30:00')

file = '2012-03-01.csv'
subdir = '1119055_1'

EV_obj = EV(file, subdir)


cols_to_drop = ['cycle_sec', 'timestep', 'speed_mph', 'accel_meters_ps', 'speed_mps',
                 'accel_mps2', 'P_wheels', 'P_electric_motor', 'n_rb', 'P_regen']

P_total = EV_obj.data.copy()
P_total = P_total.drop(columns=cols_to_drop)
P_total = P_total.set_index('timestamp')
print('PTOTAL:\n{}'.format(P_total))
P_total = P_total.set_index(P_total.index
                            + DateOffset(days=(start_time.floor(freq='D')
                                               - P_total.iloc[0].name.floor(freq='D')).days))
print('PTOTAL:\n{}'.format(P_total))
start = P_total.iloc[3, :].name -timedelta(hours=10)
end = P_total.iloc[-2, :].name #+timedelta(seconds=3)
print(start)
P_total.loc[:start,:] = 0
hi = P_total.loc[:start,:]
# P_total2 = P_total.copy()
# P_total2 = P_total2.iloc[len(hi):,:].append(hi)
# print(P_total2)
print(hi)
# create dataframe based on number of rows needed, then index to timestamp
# df = pd.DataFrame()