import pandas as pd

df = pd.read_csv('cleaned_house_1_agg.csv')
df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
df.set_index('Time', inplace=True)
print(df)
resampled_df = df.resample('5T').mean()
print(resampled_df)
resampled_df.to_csv('resampled_5min_cleaned_house_1_agg.csv')