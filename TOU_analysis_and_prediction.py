import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace import sarimax

class TOU(object):

    def __init__(self, file_name):

        self.file_name = file_name
        self.data = self.format_TOU_data()
        self.time_idx_TOU_price = self.create_time_idx_TOU_price()

    def load_xlsx_data(self, subdir=''):
        """
        Loads data from .xlsx file in to DataFrame

        :param file_name: .xlsx file name in string
        :return: extracted data in DataFrame
        """

        file_dir = os.path.realpath('./')
        for root, dirs, files in os.walk(file_dir):
            if root.endswith(subdir):
                for name in files:
                    if name == self.file_name:
                        file_path = os.path.join(root, name)

        df = pd.read_excel(file_path)

        return df

    def format_TOU_data(self):
        """
        Removes unwanted features (columns) and formats date and time

        :return: formatted and stripped DataFrame
        """

        df = self.load_xlsx_data()

        cols_to_drop = ['code', 'gsp', 'region_name']
        df = df.drop(columns=cols_to_drop)

        df['date'] = pd.to_datetime(df['date'])
        df['from'] = pd.to_timedelta(df['from'])
        df['to'] = pd.to_timedelta(df['to'])

        return df

    def create_time_idx_TOU_price(self):
        """
        Creates a DataFrame of the TOU prices with VAT included indexed by their timestamps (date and time)

        :return: created DataFrame
        """

        time_idx_TOU_price = self.data.copy()

        time_idx_TOU_price['timestamp'] = time_idx_TOU_price['date'] + time_idx_TOU_price['from']

        cols_to_drop = ['date', 'from', 'to', 'unit_rate_excl_vat']
        time_idx_TOU_price.drop(cols_to_drop, axis=1, inplace=True)

        time_idx_TOU_price = time_idx_TOU_price.set_index('timestamp')

        return time_idx_TOU_price

    def create_and_fit_model(self, fitted_model_filename='fitted_model.pickle'):
        """
        Creates Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model
        and fits with the VAT-included TOU price data

        :param fitted_model_filename: pickle file to save the fitted model
        :return: fitted model object (same as the one saved in the pickle file)
        """

        mod = sm.tsa.statespace.SARIMAX(self.time_idx_TOU_price,
                                        order=(1, 1, 1),
                                        seasonal_order=(1, 1, 0, 52),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        results = mod.fit()
        results.save(fitted_model_filename)

        return results

    def predict_and_compare(self, start, end, fitted_model_filename='fitted_model.pickle'):
        """
        Predicts the TOU prices throughout the specified range and plots against actual TOU prices

        :param start: start timestamp in the pandas datetime format (e.g. datetime64[ns])
        :param end: end timestamp in the pandas datetime format (e.g. datetime64[ns])
        :param fitted_model_filename: pickle file to load the fitted model
        :return: model prediction object
        """

        fitted_model = sarimax.SARIMAXResultsWrapper.load(fitted_model_filename)

        pred = fitted_model.get_prediction(start=start, end=end, dynamic=False)

        ax = self.time_idx_TOU_price[start:end].plot(label='actual')
        pred.predicted_mean.plot(ax=ax, label='predicted', figsize=(10, 5))
        plt.legend()

        if start.strftime('%Y-%m-%d') == end.strftime('%Y-%m-%d'):
            unique_file_name = start.strftime('%Y-%m-%d')
        else:
            unique_file_name = start.strftime('%Y-%m-%d') + '_to_' + end.strftime('%Y-%m-%d')

        plt.savefig('TOU_figures/TOU_actual_n_pred_'+unique_file_name+'.png')

        return pred

    def plot_daily_TOU(self, date):
        """
        Plots the TOU electricity price on a specific date

        :param date: specified date
        """

        data_for_selected_date = self.data.loc[self.data['date'] == date]

        data_for_selected_date.plot(x='from', y='unit_rate_incl_vat', figsize=(10, 5))
        plt.savefig('TOU_figures/TOU_' + date + '.png')

    def plot_multiple_TOU(self, date_list):
        """
        Plots TOU electricity prices for all the dates specified

        :param date_list: specified dates in a list
        """

        plt.figure(figsize=(10, 5))

        for date in date_list:
            data_for_selected_date = self.data.loc[self.data['date'] == date]
            plt.plot('from', 'unit_rate_incl_vat', data=data_for_selected_date, label=date)

        plt.legend()
        plt.savefig('TOU_figures/TOU_multiple_' + str(len(date_list)) + '.png')

    def plot_yearly_avg_TOU(self):
        """
        Plots yearly average of the TOU electricity price
        """

        avg = []
        for timestamp in self.data['from'].unique():
            timestamp_avg = self.data.loc[self.data['from'] == timestamp]['unit_rate_incl_vat'].mean()
            avg.append(timestamp_avg)

        time_and_avg = {'timestamp': self.data['from'].unique(), 'avg': avg}
        df_time_and_avg = pd.DataFrame(data=time_and_avg)
        df_time_and_avg.plot(x='timestamp', y='avg', figsize=(10, 5))
        plt.savefig('TOU_figures/TOU_yearly.png')
