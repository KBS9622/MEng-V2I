from TOU_analysis_and_prediction import TOU
import pandas as pd

tou_file = 'full_data.csv'
tou_subdir = 'data/TOU_Data'

tou_obj = TOU(tou_file, tou_subdir)
# uncomment line below when training
# tou_obj.create_and_fit_model(seasonality = 12, fitted_model_filename='cluster_TOU.pickle')

# start_time1 = pd.to_datetime('2019-01-31 00:00:00')
# end_time1 = pd.to_datetime('2019-01-31 23:30:00')
start_time = pd.to_datetime('2019-01-31 00:00:00')
end_time = pd.to_datetime('2019-02-01 23:30:00')


pred = tou_obj.predict_and_compare(start_time, end_time, fitted_model_filename= 'fitted_model.pickle')

# with open('test_file.txt', 'w') as f:
#     f.write("This is a sample file")
