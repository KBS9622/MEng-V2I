from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as ec
# from selenium.webdriver.common.by import By
import pandas as pd
import requests
import csv
import os

# driver = webdriver.Chrome('../utils/chromedriver.exe')


# start_time = pd.to_datetime('2019-01-31 01:23:00')
# end_time = pd.to_datetime('2019-01-31 23:13:00')

# if start_time:
#     print(start_time.date())

def load_csv_data(file_name, subdir=''):
    """
    Loads data from .csv file in to DataFrame
    :param file_name: .csv file name in string
    :param subdir: optional parameter to specify the subdirectory of the file
    :return: extracted data in DataFrame
    """

    file_dir = os.path.realpath('./'+subdir)
    print(file_dir)
    for root, dirs, files in os.walk(file_dir):
        print(root)
        if root.endswith(subdir):
            for name in files:
                if name == file_name:
                    file_path = os.path.join(root, name)

    df = pd.read_csv(file_path)

    return df

file1 = 'Device12_26_9_17.csv'
file2 = 'Device12_25_9_17.csv'
subdir = 'data/yun_solution_drive_cycle'

hi = load_csv_data(file1,subdir=subdir)
print(hi)


