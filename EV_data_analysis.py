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

def calculate_energy_consumption(data):
    """
    Calculates the energy consumption trend and plots it against time

    :param data: data in DataFrame
    :return: data in DataFrame with four new columns: speed in m/s, acceleration in m/s^2, power at wheels in W and power at electric motor in W
    """

    m = 1521 # mass (kg)
    g = 9.8066 # gravity (m/s)
    theta = 0 # road grade

    #rolling resistance parameters
    C_r = 1.75
    c_1 = 0.0328
    c_2 = 4.575

    rho_air = 1.2256 # air mass density (kg/m3)
    A_f = 2.3316 # frontal area of the vehicle (m2)
    C_D = 0.28 # aerodynamic drag coefficient of the vehicle

    mph_to_mps = 0.44704

    data['speed_mps'] = mph_to_mps * data['speed_mph']
    data['accel_mps2'] = mph_to_mps * data['accel_meters_ps']

    # Power required at the wheels
    data['P_wheels'] = (m * data['accel_mps2'] \
                       + m * g * np.cos(theta) * C_r * 1e-3 * (c_1 * data['speed_mps'] + c_2) \
                       + 0.5 * rho_air * A_f * C_D * (data['speed_mps']**2) \
                       + m * g * np.sin(theta)) * data['speed_mps']

    n_driveline = 0.92 # driveline efficiency
    n_electric_motor = 0.91 # electric motor efficiency (85%-95% for Nissan Leaf)

    # Power at electric motor 
    data['P_electric_motor'] = data['P_wheels'] / (n_driveline * n_electric_motor)

    data.plot(x='timestamp', y='P_electric_motor')
    plt.savefig('energy_consumption.png')

    return data

def regen_braking(data):
    """
    Calculates the energy consumption (with regenerative braking efficiency included) trend and plots it against time

    :param data: data in DataFrame
    :return: data in DataFrame with two new columns: regenerative braking efficiency and power at electric motor adjusted with n_rb
    """

    alpha = 0.0411
    
    data['n_rb'] = (np.exp(alpha / abs(data['accel_mps2']) ) )**-1
    data['n_rb'].where(data['accel_mps2'] < 0, other = 0, inplace = True) #regenerative braking efficiency is ZERO when acceleration >= 0

    data['P_regen'] = data['P_electric_motor']
    for x in data['P_regen']:
        if x < 0:
            x *= data['n_rb']

    data.plot(x='timestamp', y='P_regen')
    plt.savefig('energy_consumption_with_regen.png')

    # data.plot(x='timestamp', y='n_rb')
    # plt.savefig('n_rb.png')

    return data


file = '2012-05-22.csv'
subdir = '1035198_1'
data = load_csv_data(file, subdir)

data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')

sliced_data = calculate_energy_consumption(data.loc[1:593])

regen_sliced_data = regen_braking(sliced_data)

sliced_data.plot(x='timestamp', y='speed_mps')
plt.savefig('speed_profile.png')

# file = 'OBD2_combined_data.csv'

