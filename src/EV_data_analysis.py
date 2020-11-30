import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EV(object):
    excess = None
    deficit = None

    g = 9.8066  # gravity (m/s)
    theta = 0  # road grade

    EV_model = None
    m_kg = None  # mass (kg)

    # rolling resistance parameters
    C_r = None
    c_1 = None
    c_2 = None

    rho_air = None  # air mass density (kg/m3)
    A_f = None  # frontal area of the vehicle (m2)
    C_D = None  # aerodynamic drag coefficient of the vehicle
    n_driveline = None  # driveline efficiency
    n_electric_motor = None  # electric motor efficiency (85%-95% for Nissan Leaf)

    capacity = None  # In Wh
    charge_lvl = None  # In Wh
    soc = None  # State of charge of the battery in %

    data = None

    def __init__(self, file_name, subdir='', c_file_name='EV_characteristics.csv', choice=None):

        self.file_name = file_name
        self.subdir = subdir
        self.c_file_name = c_file_name  # EV characteristics file name
        self.choice = choice
        self.data = self.format_EV_data()

        self.EV = self.EV_menu()

        self.EV_model = self.EV['vehicle_model']
        self.m = self.EV['m_kg']
        self.C_r = self.EV['C_r']
        self.c_1 = self.EV['c_1']
        self.c_2 = self.EV['c_2']
        self.rho_air = self.EV['rho_air']
        self.A_f = self.EV['A_f']
        self.C_D = self.EV['C_D']
        self.n_driveline = self.EV['n_driveline']
        self.n_electric_motor = self.EV['n_electric_motor']
        self.capacity = self.EV['capacity']
        self.charge_lvl = self.capacity * (50 / 100)  # battery is 50% charged initially

        self.calculate_energy_consumption()

    def load_csv_data(self, file_name, subdir=''):
        """
        Loads data from .csv file in to DataFrame
        :param file_name: .csv file name in string
        :param subdir: optional parameter to specify the subdirectory of the file
        :return: extracted data in DataFrame
        """

        file_dir = os.path.realpath('./'+subdir)
        print(file_dir)
        for root, dirs, files in os.walk(file_dir):
            if root.endswith(subdir):
                for name in files:
                    if name == file_name:
                        file_path = os.path.join(root, name)

        df = pd.read_csv(file_path)

        return df

    def format_EV_data(self):
        """
        Formats date and time in the 'timestamp' column
        :return: formatted DataFrame
        """

        df = self.load_csv_data(self.file_name, self.subdir)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M:%S')

        return df

    def EV_menu(self):
        """
        Menu for selection of EV to be used in system
        :return: dataframe containing parameters for selected EV
        """

        EV_selection = self.load_csv_data(self.c_file_name)
        if self.choice is None:
            while True:
                print('****** EV SELECTION MENU ******')
                print(EV_selection['vehicle_model'])

                try:
                    choice = int(input("""Please key in the number corresponding to the vehicle model : """))
                    EV = EV_selection.iloc[choice]
                except (IndexError, ValueError) as e:
                    print('Invalid input')
                    continue
                break
        else:
            EV = EV_selection.iloc[self.choice]

        return EV

    def calculate_energy_consumption(self):
        """
        Calculates the energy consumption trend and plots it against time
        :return: data in DataFrame with 4 new columns: speed in m/s, acceleration in m/s^2,
                                                        power at wheels in W and power at electric motor in W
        """
        # Comment below if using formatted yun solutions data
        # mph_to_mps = 0.44704

        # self.data['speed_mps'] = mph_to_mps * self.data['speed_mph']
        # self.data['accel_mps2'] = mph_to_mps * self.data['accel_meters_ps'] #the feature is named m/s but unit is mph/s

        # Power required at the wheels
        self.data['P_wheels'] = (self.m * self.data['accel_mps2'] + self.m * self.g * np.cos(self.theta) *
                                 self.C_r * 1e-3 * (
                                             self.c_1 * self.data['speed_mps'] + self.c_2)
                                 + 0.5 * self.rho_air * self.A_f * self.C_D * (self.data['speed_mps'] ** 2)
                                 + self.m * self.g * np.sin(self.theta)) * self.data['speed_mps']

        # Power at electric motor
        self.data['P_electric_motor'] = self.data['P_wheels'] / (self.n_driveline * self.n_electric_motor)

        self.regen_braking()

        return self.data

    def regen_braking(self):
        """
        Calculates the energy consumption (with regenerative braking efficiency, auxiliary loads and model error included)
        trend and plots it against time
        :param data: data in DataFrame
        :return: data in DataFrame with 3 new columns: regenerative braking efficiency,
                                                        power at electric motor adjusted with n_rb
                                                        and total power consumed including auxiliary loads and model error
        """
        alpha = 0.0411

        # regenerative braking efficiency is ZERO when acceleration >= 0
        self.data['n_rb'] = (np.exp(alpha / abs(self.data['accel_mps2']))) ** -1
        self.data['n_rb'].where(self.data['accel_mps2'] < 0, other=0, inplace=True)

        # calculate the energy being stored back to the battery whenever the car decelerates
        self.data['P_regen'] = self.data['P_electric_motor']
        self.data['P_regen'] *= self.data['n_rb']

        # add the energy consumption when the car accelerates
        pos_energy_consumption = self.data['P_electric_motor'].copy()
        pos_energy_consumption.where(self.data['accel_mps2'] >= 0, other=0, inplace=True)
        self.data['P_regen'] += pos_energy_consumption

        # add the energy consumption of auxiliary loads and model error
        auxiliary = 700  # Watts or Joules per second
        average_model_error = -5.9  # percent
        self.data['P_total'] = (self.data['P_regen'] + auxiliary) * (100 + average_model_error) / 100

    def charge(self, power_in_joules):
        """
        Method to charge EV battery, accounting for battery efficiency
        :param data: power_in_joules [power (because data is measured every second)]
        :return: new SOC for the EV object
        """
        # REMEMBER: battery efficiency needs to be factored in when determining amount to charge in recommend method of charging_recommendation.py
        # maximum SOC to ensure safe operation
        max_soc = 95  # in %
        max_charge_lvl = (max_soc / 100) * self.capacity

        n_battery = 90  # battery efficiency
        Wh_to_J = 3600
        power = (power_in_joules / Wh_to_J) / (n_battery / 100)  # convert joules to Wh

        if (max_charge_lvl - self.charge_lvl) >= power:
            self.charge_lvl += power
        else:
            temp = self.charge_lvl + power - max_charge_lvl
            self.excess += temp
            print('Battery is full, {} Wh of excess energy'.format(temp))
            self.charge_lvl = max_charge_lvl

        # calculates the new instantaneous SOC
        self.soc = (self.charge_lvl / self.capacity) * 100

    def discharge(self, power_in_joules):
        """
        Method to discharge EV battery, accounting for battery efficiency
        :param data: power (because data is measured every second)
        :return: new SOC for the EV object
        """

        # minimum SOC to ensure safe operation
        min_soc = 20
        min_charge_lvl = (min_soc / 100) * self.charge_lvl

        n_battery = 90  # battery efficiency
        Wh_to_J = 3600
        power = (power_in_joules / Wh_to_J) / (n_battery / 100)  # convert joules to Wh

        if power > (self.charge_lvl - min_charge_lvl):
            temp = power - (self.charge_lvl - min_charge_lvl)
            self.deficit += temp
            print('Battery is COMPLETELY drained, {} Wh of energy deficit'.format(temp))
            self.charge_lvl = min_charge_lvl
        else:
            self.charge_lvl -= power

        # calculates the new instantaneous SOC
        self.soc = (self.charge_lvl / self.capacity) * 100

    def soc_over_time(self):
        """
        Calculates the SOC and charge level of the vehicle over time
        :param data: -
        :return: data in DataFrame with 2 new columns: SOC of EV over time
                                                        and charge level of EV over time
        """
        timeseries_soc = []
        timeseries_charge_lvl = []
        for x in self.data['P_total']:
            self.discharge(x)
            timeseries_soc.append(self.soc)
            timeseries_charge_lvl.append(self.charge_lvl)
        self.data['soc'] = timeseries_soc
        self.data['charge_lvl'] = timeseries_charge_lvl

    def graph_plotter(self, x='timestamp', y='P_regen', file_name='energy_consumption_with_regen.png',
                      subdir='test', date='test'):
        """
        Plots a graph according to the specified x and y, and saves it to the specified file name
        :param data: data in DataFrame
        :param x: the name of the column for the x axis
        :param y: the name(s) of the column for the y axis
        :param file_name: the file name(s) to store the plot(s)
        """
        figure_folder = 'EV_figures'
        directory = figure_folder + '/' + subdir + '/' + self.EV_model + '/' + date  # directory to store the figures

        if not os.path.exists(directory):
            os.makedirs(directory)

        for col, name in zip(y, file_name):
            self.data.plot(x=x, y=col)
            plt.savefig(directory + '/' + name)