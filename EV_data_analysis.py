import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv_data(file_name, subdir=''):
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

def EV_menu():
    """
    Menu for selection of EV to be used in system

    :param data: -
    :return: dataframe containing parameters for selected EV
    """

    file = 'EV_characteristics.csv'
    EV_selection = load_csv_data(file)

    print('****** EV SELECTION MENU ******')
    print(EV_selection['vehicle_model'])
    choice = input("""Please key in the number corresponding to the vehicle model : """)
    EV = EV_selection.iloc[int(choice)]

    return EV

def calculate_energy_consumption(data, EV):
    """
    Calculates the energy consumption trend and plots it against time

    :param data: data in DataFrame
    :return: data in DataFrame with four new columns: speed in m/s, acceleration in m/s^2,
             power at wheels in W and power at electric motor in W
    """

    m = EV['m_kg'] # mass (kg)
    g = 9.8066 # gravity (m/s)
    theta = 0 # road grade

    #rolling resistance parameters
    C_r = EV['C_r']
    c_1 = EV['c_1']
    c_2 = EV['c_2']

    rho_air = EV['rho_air'] # air mass density (kg/m3)
    A_f = EV['A_f'] # frontal area of the vehicle (m2)
    C_D = EV['C_D'] # aerodynamic drag coefficient of the vehicle

    mph_to_mps = 0.44704

    data['speed_mps'] = mph_to_mps * data['speed_mph']
    data['accel_mps2'] = mph_to_mps * data['accel_meters_ps']

    # Power required at the wheels
    data['P_wheels'] = (m * data['accel_mps2'] \
                       + m * g * np.cos(theta) * C_r * 1e-3 * (c_1 * data['speed_mps'] + c_2) \
                       + 0.5 * rho_air * A_f * C_D * (data['speed_mps']**2) \
                       + m * g * np.sin(theta)) * data['speed_mps']

    n_driveline = EV['n_driveline'] # driveline efficiency
    n_electric_motor = EV['n_electric_motor'] # electric motor efficiency (85%-95% for Nissan Leaf)

    # Power at electric motor 
    data['P_electric_motor'] = data['P_wheels'] / (n_driveline * n_electric_motor)

    return data

def regen_braking(data):
    """
    Calculates the energy consumption (with regenerative braking efficiency included)
    trend and plots it against time

    :param data: data in DataFrame
    :return: data in DataFrame with two new columns: regenerative braking efficiency
                                                     and power at electric motor adjusted with n_rb
    """

    alpha = 0.0411

    # regenerative braking efficiency is ZERO when acceleration >= 0
    data['n_rb'] = (np.exp(alpha / abs(data['accel_mps2']) ) )**-1
    data['n_rb'].where(data['accel_mps2'] < 0, other = 0, inplace = True)

    # calculate the energy being stored back to the battery whenever the car decelerates
    data['P_regen'] = data['P_electric_motor']
    data['P_regen'] *= data['n_rb']

    # add the energy consumption when the car accelerates
    pos_energy_consumption = data['P_electric_motor'].copy()
    pos_energy_consumption.where(data['accel_mps2']>=0, other=0, inplace = True)
    data['P_regen'] += pos_energy_consumption

    return data

def graph_plotter(data, x='timestamp', y='P_regen', file_name='energy_consumption_with_regen.png',
                  subdir='test', date='test'):
    """
    Plots a graph according to the specified x and y, and saves it to the specified file name

    :param data: data in DataFrame
    :param x: the name of the column for the x axis
    :param y: the name(s) of the column for the y axis
    :param file_name: the file name(s) to store the plot(s)
    """
    figure_folder = 'EV_figures'
    directory = figure_folder+'/'+subdir+'/'+date # directory to store the figures

    if not os.path.exists(directory):
        os.makedirs(directory)

    for col, name in zip(y, file_name):
        data.plot(x=x, y=col)
        plt.savefig(directory+'/'+name)


EV_chosen = EV_menu()

file = '2012-05-22.csv'
subdir = '1035198_1'
data = load_csv_data(file, subdir)

data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')

sliced_data = calculate_energy_consumption(data.loc[1:593], EV_chosen)

regen_sliced_data = regen_braking(sliced_data)

y = ['P_electric_motor', 'speed_mps', 'P_regen', 'n_rb']
file_name = ['energy_consumption.png', 'speed_profile.png', 'energy_consumption_with_regen.png', 'n_rb.png']
graph_plotter(regen_sliced_data, y=y, file_name=file_name, subdir=subdir, date=file.strip('.csv'))

print(sum(regen_sliced_data['P_regen'])) #calculate the final energy consumption, accounting for RB efficiency
print(sum(regen_sliced_data['P_electric_motor'])) #calculate the final energy consumption, NOT accounting for RB efficiency (therefore should be smaller)
