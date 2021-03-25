from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

def load_csv_data(file_name, subdir=''):
    """
    Loads data from .csv file in to DataFrame

    :param file_name: .csv file name in string
    :param subdir: optional parameter to specify the subdirectory of the file
    :return: extracted data in DataFrame
    """

    file_dir = os.path.realpath('../')
    print(file_dir)
    for root, dirs, files in os.walk(file_dir):
        if root.endswith(subdir):
            for name in files:
                if name == file_name:
                    file_path = os.path.join(root, name)

    df = pd.read_csv(file_path, chunksize=2000000)

    return df


file_name = 'generated_data.csv'
subdir = ''
df_chunk = load_csv_data(file_name, subdir)
print(df_chunk)


chunk_list = []  # append each chunk df here 

# Each chunk is in df format
for chunk in df_chunk:  
    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk)
    
# concat the list into dataframe 
drive_cycle_df = pd.concat(chunk_list)


date_today = datetime.now()
ms_index = pd.date_range(date_today, date_today + timedelta(milliseconds=len(drive_cycle_df)-1), freq='L')
drive_cycle_df['Datetime'] = ms_index
drive_cycle_df = drive_cycle_df.set_index('Datetime')
print(drive_cycle_df)
drive_cycle_df = drive_cycle_df.resample('S').mean()
print(drive_cycle_df)