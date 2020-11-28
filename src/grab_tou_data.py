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
        self.driver = None

    def grab_today_tomorrow_data(self):
        """
        Uses selenium to load driver and webpage containing the TOU tables
        :return: Returns a tuple of todays and tomorrows TOU data in a pandas dataframe
        """
        self.driver = webdriver.Chrome('../utils/chromedriver.exe', options=self.options)
        todays_table_url = "https://energy-db.energy-stats.uk/d-solo/Y9Oh1wmZk/octopus-agile-tariff-dashboard?org" \
                           "Id=1&from=now%2Fd&to=now%2Fd&var-area_name=London&panelId=6"
        tomorrows_table_url = "https://energy-db.energy-stats.uk/d-solo/Y9Oh1wmZk/octopus-agile-tariff-dashboard?org" \
                              "Id=1&from=now%2B1d%2Fd&to=now%2B1d%2Fd&var-area_name=London&panelId=6"
        todays_prices_df = self._parse_to_dataframe(todays_table_url)
        tomorrows_prices_df = self._parse_to_dataframe(tomorrows_table_url)

        return todays_prices_df, tomorrows_prices_df

    def _parse_to_dataframe(self, url: str) -> pd.DataFrame:
        """
        Parses TOU table from webpage and converts to dataframe
        :param url: url of table to be parsed
        :return: returns single dataframe from parsed table from webpage
        """
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

    @staticmethod
    def download_tou_csv():
        """
        Downloads CSV file containing all-time TOU data, dumps data to CSV, parses out todays and tomorrows data and
        converts to respective csv files
        :return: None
        """
        response = requests.get('https://www.energy-stats.uk/wp-content/historic-data/csv_agile_C_London.csv')
        with open('./data/TOU_Data/full_data.csv', 'wb') as file:
            file.write(response.content)

        decoded_content = response.content.decode('utf-8')
        rows = list(csv.reader(decoded_content.splitlines()))

        tomorrows_rows = rows[-46:]
        todays_rows = rows[-94:-45]

        with open('./data/TOU_Data/tomorrows_data.csv', 'w', newline="") as tomorrow_csv:
            writer = csv.writer(tomorrow_csv)
            writer.writerows(tomorrows_rows)

        with open('./data/TOU_Data/todays_data.csv', 'w', newline="") as today_csv:
            writer = csv.writer(today_csv)
            writer.writerows(todays_rows)

if __name__ == "__main__":
    hi = GrabTOUData()
    hi.download_tou_csv()