from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
import pandas as pd
import requests
import csv


class GrabTOUData:
    def __init__(self):
        self.options = webdriver.ChromeOptions()
        prefs = {'download.default_directory': ".\\sample-data\\TOU_figures"}
        self.options.add_experimental_option('prefs', prefs)
        self.options.headless = True

    def grab_today_tomorrow_data(self):
        self.driver = webdriver.Chrome('chromedriver.exe', options=self.options)
        todays_table_url = "https://energy-db.energy-stats.uk/d-solo/Y9Oh1wmZk/octopus-agile-tariff-dashboard?org" \
                           "Id=1&from=now%2Fd&to=now%2Fd&var-area_name=London&panelId=6"
        tomorrows_table_url = "https://energy-db.energy-stats.uk/d-solo/Y9Oh1wmZk/octopus-agile-tariff-dashboard?org" \
                              "Id=1&from=now%2B1d%2Fd&to=now%2B1d%2Fd&var-area_name=London&panelId=6"
        todays_prices_df = self._parse_to_dataframe(todays_table_url)
        tomorrows_prices_df = self._parse_to_dataframe(tomorrows_table_url)

        return todays_prices_df, tomorrows_prices_df

    def _parse_to_dataframe(self, url: str) -> pd.DataFrame:
        self.driver.get(url)
        xpath = '/html/body/grafana-app/div/div/div/react-container/div/div/div/div/' \
                'plugin-component/panel-plugin-table/grafana-panel/div/div[2]/ng-transclude' \
                '/div[1]/div[2]/table/tbody/tr[1]/td[2]/div'
        WebDriverWait(self.driver, 2).until(ec.presence_of_element_located((By.XPATH, xpath)))
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        table = soup.find('table')
        rows = table.find('tbody').findAll('tr')

        data_set = []

        for row in rows:
            row_elements = row.findAll('td')
            time_of_use = row_elements[0].text
            price_per_unit = row_elements[1].text
            data_entry = (time_of_use, price_per_unit)
            data_set.append(data_entry)

        data_set[0] = (data_set[0][0].lstrip('Time'), data_set[0][1].lstrip('Price per unit (p)'))

        prices_df = pd.DataFrame(data_set, columns=['Time', 'Price per unit (p)'])

        return prices_df

    def download_today_tomorrow_csv(self):
        response = requests.get('https://www.energy-stats.uk/wp-content/historic-data/csv_agileoutgoing_C_London.csv')
        with open('./TOU_Data/full_data.csv', 'w') as file:
            writer = csv.writer(file)
            for line in response.iter_lines():
                writer.writerow(line.decode('utf-8').split(','))

