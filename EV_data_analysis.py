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
    :return: data in DataFrame with three new columns: speed in m/s, acceleration in m/s^2 and power at wheels in W
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

    data['P_wheels'] = (m * data['accel_mps2'] \
                       + m * g * np.cos(theta) * C_r * 1e-3 * (c_1 * data['speed_mps'] + c_2) \
                       + 0.5 * rho_air * A_f * C_D * (data['speed_mps']**2) \
                       + m * g * np.sin(theta)) * data['speed_mps']

    data.plot(x='timestamp', y='P_wheels')
    plt.savefig('energy_consumption.png')

    return data

# def regen_braking():



file = '2012-05-22.csv'
subdir = '1035198_1'
data = load_csv_data(file, subdir)

data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')

sliced_data = calculate_energy_consumption(data.loc[1:593])

sliced_data.plot(x='timestamp', y='speed_mps')
plt.savefig('speed_profile.png')

# file = 'OBD2_combined_data.csv'

