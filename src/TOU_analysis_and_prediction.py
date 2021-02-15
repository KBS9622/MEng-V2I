import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace import sarimax
from pandas.tseries.offsets import DateOffset


class TOU(object):

    def __init__(self, file_name, subdir='TOU_data'):

        self.file_name = file_name
        self.subdir = subdir
        self.data = self.format_TOU_csv()
        self.time_idx_TOU_price = self.create_time_idx_TOU_price_csv()

    def load_xlsx_data(self, file_name, subdir=''):
        """
        Loads data from .xlsx file in to DataFrame

        :param file_name: .xlsx file name in string
        :return: extracted data in DataFrame
        """

        file_dir = os.path.realpath('../')
        for root, dirs, files in os.walk(file_dir):
            if root.endswith(subdir):
                for name in files:
                    if name == file_name:
                        file_path = os.path.join(root, name)
                        break

        df = pd.read_excel(file_path)

        return df

    def load_csv_data(self, file_name, subdir='', header_exists=False):
        """
        Loads data from .csv file in to DataFrame

        :param file_name: .csv file name in string
        :param subdir: optional parameter to specify the subdirectory of the file
        :return: extracted data in DataFrame
        """

        file_dir = os.path.realpath('./' + subdir)
        # print(file_dir)
        for root, dirs, files in os.walk(file_dir):
            if root.endswith(subdir):
                for name in files:
                    if name == file_name:
                        file_path = os.path.join(root, name)
        # print(file_path)
        if not header_exists:
            df = pd.read_csv(file_path, header=None)
        else:
            df = pd.read_csv(file_path)

        return df

    def format_TOU_data(self):
        """
        Removes unwanted features (columns) and formats date and time

        :return: formatted and stripped DataFrame
        """

        df = self.load_csv_data(self.file_name, self.subdir)

        cols_to_drop = ['code', 'gsp', 'region_name']
        df = df.drop(columns=cols_to_drop)

        df['date'] = pd.to_datetime(df['date'])
        df['from'] = pd.to_timedelta(df['from'])
        df['to'] = pd.to_timedelta(df['to'])

        return df

    def format_TOU_csv(self, header_exists=False):
        data = self.load_csv_data(self.file_name, self.subdir, header_exists=header_exists)

        if not header_exists:
            cols = ['date', 'from', 'code', 'region_name', 'unit_rate_incl_vat']
            data.columns = cols

        date_list = data['date'].str.split('T', n=1, expand=True)
        date_list[1] = date_list[1].str.replace('Z', '')

        data['date'] = date_list[0]
        data['from'] = date_list[1]
        data['date'] = pd.to_datetime(data['date'])
        data['from'] = pd.to_timedelta(data['from'])

        return data

    def create_time_idx_TOU_price_csv(self):
        time_idx_TOU_price = self.data.copy()

        # creates a new column named 'time_stamp' which is the sum of columns 'date' and 'from'
        time_idx_TOU_price['time_stamp'] = time_idx_TOU_price['date'] + time_idx_TOU_price['from']
        # time_idx_TOU_price['time_stamp'] += DateOffset(minutes=60)
        cols_to_drop = ['date', 'from', 'code', 'region_name']
        time_idx_TOU_price.drop(cols_to_drop, axis=1, inplace=True)
        time_idx_TOU_price = time_idx_TOU_price.set_index('time_stamp')

        return time_idx_TOU_price

    def create_time_idx_TOU_price(self):
        """
        Creates a DataFrame of the TOU prices with VAT included indexed by their timestamps (date and time)

        :return: created DataFrame
        """

        time_idx_TOU_price = self.data.copy()

        time_idx_TOU_price['time_stamp'] = time_idx_TOU_price['date'] + time_idx_TOU_price['from']

        cols_to_drop = ['date', 'from', 'to', 'unit_rate_excl_vat']
        time_idx_TOU_price.drop(cols_to_drop, axis=1, inplace=True)

        time_idx_TOU_price = time_idx_TOU_price.set_index('time_stamp')

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
                                        seasonal_order=(1, 1, 0, 12),
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

        pred = fitted_model.predict(start=start + DateOffset(minutes=30),
                                    end=end + DateOffset(minutes=30), dynamic=False)
        print(pred)
        pred = pred.to_frame(name='TOU')
        print(pred)
        pred = pred.set_index(pred.index - DateOffset(minutes=30))
        print(pred)
        ax = self.time_idx_TOU_price[start:end].plot(label='actual')
        pred.plot(ax=ax, label='predicted', figsize=(10, 5))
        plt.legend()

        if start.strftime('%Y-%m-%d') == end.strftime('%Y-%m-%d'):
            unique_file_name = start.strftime('%Y-%m-%d')
        else:
            unique_file_name = start.strftime('%Y-%m-%d') + '_to_' + end.strftime('%Y-%m-%d')


        plt.savefig('./data/TOU_figures/TOU_actual_n_pred_'+unique_file_name+'.png')
        return pred

    def plot_daily_TOU(self, date):
        """
        Plots the TOU electricity price on a specific date

        :param date: specified date
        """

        data_for_selected_date = self.data.loc[self.data['date'] == date]

        data_for_selected_date.plot(x='from', y='unit_rate_incl_vat', figsize=(10, 5))
        plt.savefig('./data/TOU_figures/TOU_' + date + '.png')

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
        plt.savefig('./data/TOU_figures/TOU_multiple_' + str(len(date_list)) + '.png')

    def plot_yearly_avg_TOU(self):
        """
        Plots yearly average of the TOU electricity price
        """

        avg = []
        for time_stamp in self.data['from'].unique():
            time_stamp_avg = self.data.loc[self.data['from'] == time_stamp]['unit_rate_incl_vat'].mean()
            avg.append(time_stamp_avg)

        time_and_avg = {'time_stamp': self.data['from'].unique(), 'avg': avg}
        df_time_and_avg = pd.DataFrame(data=time_and_avg)
        df_time_and_avg.plot(x='time_stamp', y='avg', figsize=(10, 5))
        plt.savefig('./data/TOU_figures/TOU_yearly.png')
