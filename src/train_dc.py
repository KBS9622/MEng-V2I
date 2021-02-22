from dc_analysis_and_prediction import DriveCycle
import pandas as pd

dc_file = '12_sep_oct_nov_nov_dec.csv'

dc_obj = DriveCycle(dc_file,preprocess_resample=True)
# uncomment line below when training
dc_obj.create_and_fit_model(seasonality = 2, fitted_model_filename='fitted_model_dc.pickle')

# start_time1 = pd.to_datetime('2019-01-31 00:00:00')
# end_time1 = pd.to_datetime('2019-01-31 23:30:00')
start_time = pd.to_datetime('2017-12-23 00:00:00')
end_time = pd.to_datetime('2017-12-23 23:59:59')


pred = dc_obj.predict_and_compare(start_time, end_time, fitted_model_filename= 'fitted_model_dc.pickle')

# with open('test_file.txt', 'w') as f:
#     f.write("This is a sample file")
