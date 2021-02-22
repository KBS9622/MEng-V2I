from TOU_analysis_and_prediction import TOU
import pandas as pd

tou_training_file = 'full_data.csv'
tou_test_file = "data/test_data.csv"
tou_subdir = "/data/TOU_Data"

tou_obj = TOU(tou_training_file, tou_subdir)
# uncomment line below when training
tou_obj.create_and_fit_model(seasonality=48, fitted_model_filename='cluster_TOU_48.pickle')

# start_time1 = pd.to_datetime('2019-01-31 00:00:00')
# end_time1 = pd.to_datetime('2019-01-31 23:30:00')
start_time = pd.to_datetime('2021-01-16 00:00:00')
end_time = pd.to_datetime('2021-01-16 23:30:00')

pred = tou_obj.predict_and_compare(start_time, end_time, fitted_model_filename='cluster_TOU_48.pickle')

# with open('test_file.txt', 'w') as f:
#     f.write("This is a sample file")
