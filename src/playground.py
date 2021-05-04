import pandas as pd
from TOU_analysis_and_prediction import TOU

# df = pd.read_csv('cleaned_house_3_agg.csv')
# df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
# df.set_index('Time', inplace=True)
# print(df)
# resampled_df = df.resample('5T').mean()
# print(resampled_df)
# resampled_df.to_csv('resampled_5min_cleaned_house_3_agg.csv')

# df = pd.read_csv('IntelliCharga_EV_profile_for_2019-10-13.csv')
# df['time_stamp'] = pd.to_datetime(df['time_stamp'], dayfirst=True)
# df.set_index('time_stamp', inplace=True)
# # print(df)
# resampled_df = df.resample('5T').mean()
# # print(resampled_df)
# resampled_df.to_csv('resampled_5min_IntelliCharga_EV_profile_for_2019-10-13.csv')
tou_file = 'full_data.csv'
tou_subdir = 'data/TOU_Data'
tou_obj = TOU(tou_file, tou_subdir)
start_time = pd.to_datetime('2021-01-16 00:00:00')
end_time = pd.to_datetime('2021-01-16 23:30:00')

pred = tou_obj.predict_and_compare(start_time, end_time, fitted_model_filename='cluster_TOU_48.pickle')
