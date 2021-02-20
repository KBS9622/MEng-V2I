import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace import sarimax
from pandas.tseries.offsets import DateOffset


class DriveCycle(object):

    def __init__(self, file_name, preprocess_resample=False, subdir='combined/'):

        self.file_name = file_name
        self.subdir = subdir
        self.data = self.format_drive_cycle_csv(process=preprocess_resample)

    def load_csv_data(self, file_name, subdir='', header_exists=False):
        """
        Loads data from .csv file in to DataFrame

        :param file_name: .csv file name in string
        :param subdir: optional parameter to specify the subdirectory of the file
        :return: extracted data in DataFrame
        """
        
        # file_dir = os.path.realpath('./TimeSeriesPrediction/' + subdir)
        # print(os.getcwd())
        # for root, dirs, files in os.walk(file_dir):
        #     print('a:{}'.format(root))
        #     print('b:{}'.format(dirs))
        #     print('c:{}'.format(files))
        #     if root.endswith(subdir):
        #         print('in if loop')
        #         for name in files:
        #             if name == file_name:
        #                 file_path = os.path.join(root, name)
        file_path = os.path.realpath('./TimeSeriesPrediction/' + subdir + file_name)
        print(file_path)
        df = pd.read_csv(file_path, parse_dates=['timeStamp'])

        return df

    def format_drive_cycle_csv(self, process, header_exists=True):
        data = self.load_csv_data(self.file_name, self.subdir, header_exists=header_exists)

        if not header_exists:
            cols = ['timeStamp', 'speed']
            data.columns = cols
        data = data.loc[:,['timeStamp', 'speed']]
        data['time_stamp'] = pd.to_datetime(data['timeStamp'])
        cols_to_drop = ['timeStamp']

        if process:
            outlier_index = data.loc[data['speed'] > 200, :].index

            for x in outlier_index:
                data.loc[x, 'speed'] = data.loc[x - 1, 'speed']

        data.drop(cols_to_drop, axis=1, inplace=True)
        data = data.set_index('time_stamp')

        if process:
            data = data.resample("1S").first().ffill()

        return data

    def create_and_fit_model(self, seasonality=12, fitted_model_filename='fitted_model_dc.pickle'):
        """
        Creates Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model
        and fits with the VAT-included TOU price data

        :param fitted_model_filename: pickle file to save the fitted model
        :return: fitted model object (same as the one saved in the pickle file)
        """

        mod = sm.tsa.statespace.SARIMAX(self.data,
                                        order=(1, 1, 1),
                                        seasonal_order=(1, 1, 0, seasonality),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        print('line 141')
        results = mod.fit()
        print('line 143')
        results.save(fitted_model_filename)
        print('line 145')
        return results

    def predict_and_compare(self, start, end, fitted_model_filename='fitted_model_dc.pickle'):
        """
        Predicts the TOU prices throughout the specified range and plots against actual TOU prices

        :param start: start timestamp in the pandas datetime format (e.g. datetime64[ns])
        :param end: end timestamp in the pandas datetime format (e.g. datetime64[ns])
        :param fitted_model_filename: pickle file to load the fitted model
        :return: model prediction object
        """

        fitted_model = sarimax.SARIMAXResultsWrapper.load(fitted_model_filename)
        # pred = fitted_model.predict(dynamic=False)
        pred = fitted_model.predict(start=start, end=end, dynamic=False)
        print(pred)
        pred = pred.to_frame(name='DriveCycle')
        print(pred)
        # pred = pred.set_index(pred.index - DateOffset(minutes=30))
        # print(pred)
        ax = self.data[start:end].plot(label='actual')
        pred.plot(ax=ax, label='predicted', figsize=(10, 5))
        plt.legend()

        if start.strftime('%Y-%m-%d') == end.strftime('%Y-%m-%d'):
            unique_file_name = start.strftime('%Y-%m-%d')
        else:
            unique_file_name = start.strftime('%Y-%m-%d') + '_to_' + end.strftime('%Y-%m-%d')

        plt.savefig('./data/dc_figures/dc_actual_n_pred_' + unique_file_name + '.png')
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
