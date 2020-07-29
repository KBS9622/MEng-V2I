import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_xlsx_data(file_name):
    """
    Loads data from .xlsx file in to DataFrame

    :param file_name: .xlsx file name in string
    :return: extracted data in DataFrame
    """

    file_dir = os.path.realpath('./')
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            if name == file_name:
                file_path = os.path.join(root, name)

    df = pd.read_excel(file_path)

    return df

def format_TOU_data(df):
    """
    Removes unwanted features (columns) and formats date and time

    :param df: TOU data in DataFrame
    :return: formatted and stripped DataFrame
    """

    features_to_drop = ['code','gsp','region_name']
    df = df.drop(columns=features_to_drop)

    df['date'] = pd.to_datetime(df['date'])
    df['from'] = pd.to_timedelta(df['from'])
    df['to'] = pd.to_timedelta(df['to'])

    return df

def preprocessing(df):

    X = df['from']
    y = df['unit_rate_incl_vat']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values.reshape(-1, 1))
    X_test = scaler.transform(X_test.values.reshape(-1, 1))

    return X_train, X_test, y_train, y_test

def plot_daily_TOU(df, date):
    """
    Plots the TOU electricity price on a specific date

    :param df: data in DataFrame
    :param date: specified date
    """

    data_for_selected_date = data.loc[df['date'] == date]

    data_for_selected_date.plot(x='from', y='unit_rate_incl_vat',figsize=(10,5))
    plt.savefig('TOU_figures/TOU_'+date+'.png')

def plot_multiple_TOU(df, date_list):
    """
    Plots TOU electricity prices for all the dates specified

    :param df: data in DataFrame
    :param date_list: specified dates in a list
    """

    plt.figure(figsize=(10, 5))

    for date in date_list:
        data_for_selected_date = df.loc[df['date'] == date]
        plt.plot('from', 'unit_rate_incl_vat',data=data_for_selected_date, label=date)

    plt.legend()
    plt.savefig('TOU_figures/TOU_multiple_'+str(len(date_list))+'.png')

def plot_yearly_avg_TOU(df):
    """
    Plots yearly average of the TOU electricity price

    :param df: data in DataFrame
    """

    avg = []
    for timestamp in df['from'].unique():
        timestamp_avg = df.loc[df['from']==timestamp]['unit_rate_incl_vat'].mean()
        avg.append(timestamp_avg)

    time_and_avg = {'timestamp': df['from'].unique(), 'avg': avg}
    df_time_and_avg = pd.DataFrame(data=time_and_avg)
    df_time_and_avg.plot(x='timestamp', y='avg', figsize=(10, 5))
    plt.savefig('TOU_figures/TOU_yearly.png')


# file = 'agile_rates_2019.xlsx'
# data = load_xlsx_data(file)
# data = format_TOU_data(data)
#
# selected_date = '2019-01-31'
# plot_daily_TOU(data, selected_date)
#
# plot_yearly_avg_TOU(data)
#
# date_list = ['2019-01-31', '2019-02-01', '2019-07-01']
# plot_multiple_TOU(data, date_list)