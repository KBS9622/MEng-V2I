import pandas as pd
import numpy as np
from EV_data_analysis import *
from TOU_analysis_and_prediction import TOU
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace import sarimax
from pandas.tseries.offsets import DateOffset

file = 'EV_characteristics.csv'
EV_selection = load_csv_data(file)
choice = 0 #Nissan Leaf
EV_chosen = EV_selection.iloc[choice]

file = '2012-05-22.csv'
subdir = '1035198_1'
data = load_csv_data(file, subdir)

data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')

sliced_data = calculate_energy_consumption(data, EV_chosen)

regen_sliced_data = regen_braking(sliced_data)

cols_to_drop = ['vehicle_model', 'cycle_sec', 'timestep', 'speed_mph', 'accel_meters_ps',
                'speed_mps', 'accel_mps2', 'P_wheels', 'P_electric_motor', 'n_rb']

P_regen = regen_sliced_data.copy()

P_regen = P_regen.drop(columns=cols_to_drop)

P_regen = P_regen.set_index('timestamp')

# non_zero_P = P_regen[P_regen['P_regen']!=0]
non_zero_P = P_regen.copy()
non_zero_P = non_zero_P.set_index(non_zero_P.index + DateOffset(days=2445))

min_gap = 30*60 #30 mins in seconds
min_gap = pd.Timedelta(min_gap, unit='sec')

journey_start = [non_zero_P.iloc[0,:].name]
journey_end = [non_zero_P.iloc[-1,:].name]

for i in range(0, len(non_zero_P)-1):
    time_diff = non_zero_P.iloc[i+1,:].name - non_zero_P.iloc[i,:].name
    if time_diff >= min_gap:
        journey_start.append(non_zero_P.iloc[i+1,:].name)
        journey_end.insert(0, non_zero_P.iloc[i,:].name)

print(journey_start)
print(journey_end)

file = 'agile_rates_2019.xlsx'
TOU_obj = TOU(file)

future_dates = [TOU_obj.time_idx_TOU_price.index[-1] + DateOffset(minutes=30*i) for i in range(1, 49)]
future_dates_df = pd.DataFrame(index=future_dates, columns=TOU_obj.time_idx_TOU_price.columns)

future_df = pd.concat([TOU_obj.time_idx_TOU_price, future_dates_df])

# results = TOU_obj.create_and_fit_model()
results = sarimax.SARIMAXResultsWrapper.load('fitted_model.pickle')

start_time = pd.to_datetime('2019-01-31 00:00:00')
end_time = pd.to_datetime('2019-01-31 23:30:00')

pred = results.predict(start=start_time + DateOffset(minutes=30), end=end_time + DateOffset(minutes=30), dynamic=False)
pred = pred.to_frame(name='TOU')
pred = pred.set_index(pred.index - DateOffset(minutes=30))
pred['charging'] = 0
pred['journey'] = 0

level_2_charger = 6.6e3 #Power in W

for start, end in zip(journey_start, journey_end):

    start_time_slot = start.floor(freq='30min')
    end_time_slot = end.floor(freq='30min')

    pred.loc[np.logical_and(pred.index > start_time_slot, pred.index < end_time_slot), 'charging'] = 30

    if start_time_slot == end_time_slot:
        pred.loc[start_time_slot, ['charging', 'journey']] += (end - start).seconds/60
    else:
        pred.loc[start_time_slot, ['charging', 'journey']] += (start.ceil(freq='30min') - start).seconds/60
        pred.loc[end_time_slot, ['charging', 'journey']] += (end - end_time_slot).seconds/60

    journey_energy_consumption = sum(non_zero_P.loc[start:end]['P_regen'])
    charge_time = journey_energy_consumption/(level_2_charger*60)

    quotient = int(charge_time // 30)
    remainder = charge_time % 30

    free_time_slots = pred.loc[np.logical_and(pred.index < start, pred['charging'] < 30)].copy()
    free_time_slots = free_time_slots.sort_values(by=['TOU'])

    if charge_time + sum(free_time_slots['charging']) > 30*len(free_time_slots):
        print('Not enough time slots to charge')
        break

    remainder += sum(free_time_slots.iloc[list(range(0,quotient))]['charging'])
    idx_offset = 0
    while remainder != 0:
        remainder += free_time_slots.iloc[quotient+idx_offset]['charging']
        if remainder >= 30:
            pred.loc[free_time_slots.iloc[quotient+idx_offset].name, 'charging'] = 30
            remainder -= 30
        else:
            pred.loc[free_time_slots.iloc[quotient+idx_offset].name, 'charging'] = remainder
            remainder = 0

        idx_offset += 1

    if quotient > 0: pred.loc[free_time_slots.iloc[list(range(0,quotient))].index, 'charging'] = 30

pred['charging'] -= pred['journey']
print(pred.loc[pred['charging'] > 0, 'charging'])

# pred = results.get_prediction(start=start_time, end=end_time, dynamic=False)
# pred = results.get_forecast(steps=end_time)

# pred.plot(figsize=(10, 5))
# plt.show()

# y = ['P_electric_motor', 'speed_mps', 'P_regen', 'n_rb']
# file_name = ['energy_consumption.png', 'speed_profile.png', 'energy_consumption_with_regen.png', 'n_rb.png']
# graph_plotter(regen_sliced_data, y=y, file_name=file_name, subdir=subdir, date=file.strip('.csv'))

# print(sum(regen_sliced_data['P_regen'])) #calculate the final energy consumption, accounting for RB efficiency
# print(sum(regen_sliced_data['P_electric_motor'])) #calculate the final energy consumption, NOT accounting for RB efficiency (therefore should be smaller)
