import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def load_csv_data(self, file_name, subdir=''):
    """
    Loads data from .csv file in to DataFrame
    :param file_name: .csv file name in string
    :param subdir: optional parameter to specify the subdirectory of the file
    :return: extracted data in DataFrame
    """

    file_dir = os.path.realpath('./')
    for root, dirs, files in os.walk(file_dir):
        if root.endswith(subdir):
            for name in files:
                if name == file_name:
                    file_path = os.path.join(root, name)

    df = pd.read_csv(file_path)

    return df

def format_EV_data(self):
    """
    Formats date and time in the 'timestamp' column
    :return: formatted DataFrame
    """

    df = self.load_csv_data(self.file_name, self.subdir)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    return df

if __name__ == "__main__":
    miles_per_km = 0.621371

    file_path = 'data/yun_solution_drive_cycle/Device12.csv'
    df = pd.read_csv(file_path, index_col=2)
    # df['timeStamp'] = pd.to_datetime(df['timeStamp'], format='%d/%m/%Y %H:%M:%S')
    print(df.head())