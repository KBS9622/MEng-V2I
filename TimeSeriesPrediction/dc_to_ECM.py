import os
import numpy as np
import pandas as pd

from EV_data_analysis import EV_mod


def load_data(path, which_data, preprocess=True, resample=False):
    """Load, Preprocess and Resample the CSV file"""
    data_dir = os.path.join(path, which_data)

    data = pd.read_csv(data_dir, parse_dates=['timeStamp'])

    # remove duplicate timestamps
    data = data.drop_duplicates(subset = 'timeStamp')
    data.reset_index(inplace = True)

    data['timeStamp'] = pd.to_datetime(data['timeStamp'])

    cols_to_drop = ['Unnamed: 0',
                    'tripID',
                    'deviceID',
                    'accData',
                    'battery',
                    'cTemp',
                    'dtc',
                    'eLoad',
                    'iat',
                    'imap',
                    'kpl',
                    'maf',
                    'rpm',
                    'tAdv',
                    'tPos',
                    'fuel',
                    'gps_speed'
                    ]

    
    if preprocess:
        outlier_index = data.loc[data['speed'] > 200, :].index

        for x in outlier_index:
            data.loc[x, 'speed'] = data.loc[x - 1, 'speed']

    data.drop(cols_to_drop, axis=1, inplace=True)
    data = data.set_index('timeStamp')

    #temp code to have a smaller dataframe
    # data = data.loc['2017-09-25 13:41:41':'2017-09-26 10:17:14']

    if resample:
        data = data.resample("1S").first().ffill()

    # data['date'] = data.index.copy()

    # data['year'] = data['date'].dt.year
    # data['month'] = data['date'].dt.month
    # data['day'] = data['date'].dt.day
    # data['dayofweek'] = data['date'].dt.dayofweek
    # data['hour'] = data['date'].dt.hour
    # data['minute'] = data['date'].dt.minute
    # data['second'] = data['date'].dt.second

    # data.drop(['date'], axis=1, inplace=True)

    data = calculate_acceleration(data)

    return data

def calculate_acceleration(df):
    df['time_stamp'] = pd.to_datetime(df.index.copy())

    # makes new column called timestep to help calculate acceleration
    df['timestep'] = (df['time_stamp']-df['time_stamp'].shift(1)).astype('timedelta64[s]')
    df['acceleration'] = (df['speed']-df['speed'].shift(1))/df['timestep'] #the unit of acceleration is km/h/s, so be cautious when converting

    #makes the first row observation of acceleration to 0, as there is no previous speed value to calculate acceleration
    # df.loc[0,'acceleration'] = 0

    # the speed column is in km/h whereas acceleration column is in km/h/s, need to convert both to m/s and m/s^2 respectively
    kmph_to_mps = 3.6
    df['speed_mps'] = df['speed']/kmph_to_mps
    df['accel_mps2'] = df['acceleration']/kmph_to_mps

    df = df.set_index('time_stamp')
    # print(df)
    cols_to_drop = ['index', 'timestep', 'speed', 'acceleration']
    df = df.drop(columns=cols_to_drop)
    
    return df

def find_journey_start_and_end_points(data, min_gap=30):
    """
    Method to indetify sub-journey start/end times (Classifier Module)

    :param data: min_gap (minimum time gap to identify as two seperate sub-journey)
    :return: journey_start varible (with sub-journey start times) and journey_end variable (with corresponding sub-journey end times)
    """

    df = data.copy()
    min_gap = pd.Timedelta(min_gap * 60, unit='sec')

    journey_start = [df.iloc[0, :].name]
    journey_end = [df.iloc[-1, :].name]

    for i in range(0, len(df) - 1):
        time_diff = df.iloc[i + 1, :].name - df.iloc[i, :].name
        if time_diff >= min_gap:
            journey_start.append(df.iloc[i + 1, :].name)
            journey_end.insert(len(journey_end) - 1, df.iloc[i, :].name)

    return journey_start, journey_end

def remove_wrong_accel(data, journey_start):
    for start in journey_start:
        #changes the first sub-journey observation of the 'accel_mps2' column in the data to 0, 
        # as the first observation fo acceleration in every sub-journey should be 0
        data.loc[start,'accel_mps2'] = 0

    return data

def sum_energy(data, journey_start, journey_end):

    sum_df = []

    for start, end in zip(journey_start, journey_end):

        #get the sum of P_total for each sub-journey
        sum_df.append(sum(data.loc[start:end]['P_total']))
    
    #concatenate the list into a df
    final_df = pd.DataFrame({'start': journey_start, 'end': journey_end, 'energy(J)': sum_df})

    return final_df

if __name__ == "__main__":
    # path = 'TimeSeriesPrediction/combined/'
    # which_data = '12_sep_oct_nov_nov_dec.csv'
    # df = load_data(path, which_data)
    # df['timeStamp'] = pd.to_datetime(df['timeStamp'], format='%d/%m/%Y %H:%M:%S')
    # print(df.head())
    # journey_start, journey_end = find_journey_start_and_end_points(df)
    # print(journey_start, journey_end)
    # df = remove_wrong_accel(df, journey_start)
    # # print(df)
    # json_path = "./utils/user_config.json"
    # ev_obj = EV_mod(df, json_path)
    # df_ecm = ev_obj.data
    # print(df_ecm)
    # final_df = sum_energy(df_ecm, journey_start, journey_end)
    # # final_df.to_csv(r'Device_12_ecm.csv')
    # print(final_df)
    # final_df['start'] = pd.to_datetime(final_df['start'])
    # final_df = final_df.set_index('start')
    # final_df.drop(['end'], axis=1, inplace=True)
    # print(final_df)
    # resampled_df = final_df.resample("1D").sum()
    # print(resampled_df)
    # # resampled_df.to_csv(r'Device_12_ecm_resampled_1D.csv')

    path = ''
    which_data = 'Device_12_ecm.csv'
    data_dir = os.path.join(path, which_data)
    df = pd.read_csv(data_dir, parse_dates=['start', 'end'])
    df['index'] = df['start']
    df = df.loc[:,['start','end','energy(J)','index']].set_index('index')
    print(df)
    df = df.groupby(pd.Grouper(freq='1D')).agg({'start':'first','end':'last','energy(J)':'sum'})
    df.to_csv(r'Device_12_ecm_resampled_1D_with_start_end.csv')
    # print(df)