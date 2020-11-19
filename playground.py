import pandas as pd
from EV_data_analysis import EV
from TOU_analysis_and_prediction import TOU
from charging_recommendation import charging_recommendation
from pandas.tseries.offsets import DateOffset

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

pred['charging'] = 0
pred['journey'] = 0

free_time_slots = pred.loc[np.logical_and(pred.index < start, pred['charging'] < 30)].copy()
free_time_slots = free_time_slots.sort_values(by=['TOU'])
# if the cheapest TOU slot is below threshold, charge for that slot
if free_time_slots['TOU'].iloc[[0]] <= threshold:
    pred.loc[free_time_slots.iloc[[0]].index, 'charging'] = 30