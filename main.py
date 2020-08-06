from TOU_analysis_and_prediction import TOU
import pandas as pd

file = 'agile_rates_2019.xlsx'
TOU_obj = TOU(file)

results = TOU_obj.create_and_fit_model()

start_time = pd.to_datetime('2019-07-01 00:00:00')
end_time = pd.to_datetime('2019-07-01 23:30:00')

pred = TOU_obj.predict_and_compare(start=start_time, end=end_time)

selected_date = '2019-01-31'
TOU_obj.plot_daily_TOU(selected_date)

date_list = ['2019-01-31', '2019-02-01', '2019-07-01']
TOU_obj.plot_multiple_TOU(date_list)

TOU_obj.plot_yearly_avg_TOU()
