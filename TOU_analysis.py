import pandas as pd
import matplotlib.pyplot as plt

def extract_data(file_name):
    """
    Removes unwanted features (columns) and formats date and time

    :param file_name: file name in string
    :return: formatted and stripped DataFrame
    """

    df = pd.read_excel(file_name)

    features_to_drop = ['code','gsp','region_name']
    df = df.drop(columns=features_to_drop)

    df['date'] = pd.to_datetime(df['date'])
    df['from'] = pd.to_timedelta(df['from'])

    return df

def plot_daily_TOU(df, date):
    """
    Plots the TOU electricity price on a specific date and saves the plot

    :param df: data in DataFrame
    :param date: specified date
    """

    mask = df['date'] == date
    data_for_selected_date = data.loc[mask]

    data_for_selected_date.plot(x='from', y='unit_rate_incl_vat',figsize=(10,5))
    plt.savefig('TOU_'+date)


file = 'agile_rates_2019.xlsx'
data = extract_data(file)

selected_date = '2019-01-31'

plot_daily_TOU(data, selected_date)

