#script to process yun solutions data
import os
import pandas as pd
from datetime import datetime

class process():

    def __init__(self, filepath):
        self.filepath = filepath
    
    def load_csv(self,filename):
        path = self.filepath + '/' + filename
        df = pd.read_csv(path)
        cols_wanted = ['timeStamp','speed']
        df = df.loc[:,cols_wanted]

        return df

if __name__ == "__main__":
    filepath = '/Users/koeboonshyang/Desktop/OpenData'
    process_obj = process(filepath)
    filename_dec = 'Dec-17/13.0-0.csv'
    df_dec = process_obj.load_csv(filename_dec)
    filename_nov = 'Nov-17/13.0-0.csv'
    df_nov = process_obj.load_csv(filename_nov)
    filename_oct = 'Oct-17/13.0.csv'
    df_oct = process_obj.load_csv(filename_oct)
    filename_sept = 'Sep-17/13.0.csv'
    df_sept = process_obj.load_csv(filename_sept)

    df = [df_sept, df_oct, df_nov, df_dec]
    df = pd.concat(df, ignore_index = True)
    
    # @Heejoon, u can start here since u have already combined the csv
    #removes duplicate rows based on 'timeStamp'
    df = df.drop_duplicates(subset = 'timeStamp')
    print(df)
    df.reset_index(inplace = True)
    print(df)
    # df = df[df.index.duplicated()]

    #identifies the index of the outliers
    outlier_index = df.loc[df['speed']>200,:].index
    #loops thrrough the list of outlier index and changes the df so that any outlier is replaced by the previous observation
    for x in outlier_index:
        df.loc[x,'speed'] = df.loc[x-1,'speed'] 
        
    
    df['timestamp'] = pd.to_datetime(df['timeStamp'])

    # makes new column called timestep to help calculate acceleration
    df['timestep'] = (df['timestamp']-df['timestamp'].shift(1)).astype('timedelta64[s]')
    df['acceleration'] = (df['speed']-df['speed'].shift(1))/df['timestep'] #the unit of acceleration is km/h/s, so be cautious when converting

    #makes the first row observation of acceleration to 0, as there is no previous speed value to calculate acceleration
    df.loc[0,'acceleration'] = 0

    # the speed column is in km/h whereas acceleration column is in km/h/s, need to convert both to m/s and m/s^2 respectively
    kmph_to_mps = 3.6
    df['speed_mps'] = df['speed']/kmph_to_mps
    df['accel_mps2'] = df['acceleration']/kmph_to_mps

    df = df.set_index('timestamp')
    print(df)
    cols_to_drop = ['timeStamp', 'timestep', 'index', 'speed', 'acceleration']
    df = df.drop(columns=cols_to_drop)
    
    

    print(df)

df.to_csv(r'/Users/koeboonshyang/Desktop/Device13_formatted.csv')

