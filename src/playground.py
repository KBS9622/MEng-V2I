from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as ec
# from selenium.webdriver.common.by import By
import pandas as pd
import requests
import csv

# driver = webdriver.Chrome('../utils/chromedriver.exe')


start_time = pd.to_datetime('2019-01-31 01:23:00')
end_time = pd.to_datetime('2019-01-31 23:13:00')

if start_time:
    print(start_time.date())



