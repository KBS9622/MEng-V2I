import pandas as pd
import matplotlib.pyplot as plt
from asammdf import MDF4

# MDF4 Handling
# data = MDF4('extracted_OBD2.mf4')
# EngineRPM = data.get('S1_PID_0C_EngineRPM')
# EngineRPM.plot()

data = pd.read_csv('channel1.csv')

for i in range(2,9):

    filename = 'channel'+str(i)+'.csv'
    df = pd.read_csv(filename)

    data = data.merge(df,how='left', left_on='timestamps', right_on='timestamps')

data['timestamps'] = data['timestamps'].str.rstrip('+01:00')
data['timestamps'] = pd.to_datetime(data['timestamps'])

data.to_csv('OBD2_combined_data.csv')

vehicle_speed_data = data
vehicle_speed_data.dropna(subset=['S1_PID_0D_VehicleSpeed'], inplace=True)

vehicle_speed_data.plot(x='timestamps', y='S1_PID_0D_VehicleSpeed')
plt.show()