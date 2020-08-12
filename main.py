from TOU_analysis_and_prediction import TOU
import pandas as pd
from statsmodels.tsa.statespace import sarimax
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

file = 'agile_rates_2019.xlsx'
TOU_obj = TOU(file)

# results = TOU_obj.create_and_fit_model()
results = sarimax.SARIMAXResultsWrapper.load('fitted_model.pickle')

# start_time = pd.to_datetime('2019-07-01 00:00:00')
# end_time = pd.to_datetime('2019-07-01 23:30:00')

start_time = pd.to_datetime('2019-01-31 00:00:00')
end_time = pd.to_datetime('2019-01-31 23:30:00')

pred = results.predict(start_time, end_time, dynamic=False)
pred = pred.to_frame(name='TOU')
pred = pred.set_index(pred.index - DateOffset(minutes=30))

ax = TOU_obj.time_idx_TOU_price[start_time:end_time].plot(label='actual')
pred.iloc[1:].plot(ax=ax, label='predicted', figsize=(10, 5))
plt.legend()
plt.show()
# pred = TOU_obj.predict_and_compare(start=start_time, end=end_time)
#
# selected_date = '2019-01-31'
# TOU_obj.plot_daily_TOU(selected_date)
#
# date_list = ['2019-01-31', '2019-02-01', '2019-07-01']
# TOU_obj.plot_multiple_TOU(date_list)
#
# TOU_obj.plot_yearly_avg_TOU()
