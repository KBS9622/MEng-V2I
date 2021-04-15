import pandas as pd

df = pd.read_csv('uncontrolled_EV_profile_for_2019-10-13.csv')
df['time_stamp'] = pd.to_datetime(df['time_stamp'], dayfirst=True)
df.set_index('time_stamp', inplace=True)
print(df)
resampled_df = df.resample('T').mean()
print(resampled_df)
resampled_df.to_csv('resampled_1min_uncontrolled_EV_profile_for_2019-10-13.csv')