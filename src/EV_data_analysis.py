import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


class EV(object):
    excess = 0
    deficit = 0

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

    def __init__(self, file_name, subdir, config_path, c_file_name='EV_characteristics.csv', choice=None):
        self.file_name = file_name
        self.subdir = subdir
        self.config_path = config_path
        self.c_file_name = c_file_name  # EV characteristics file name
        self.choice = choice
        self.data = self.format_EV_data()
        self.pull_user_config()

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
        self.charging_battery_efficiency = self.config_dict["Charger_efficiency"]
        self.calculate_energy_consumption()

    def load_csv_data(self, file_name, subdir=''):
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
        Calculates the energy consumption (with regenerative braking efficiency, auxiliary loads, model error and battery efficiencies included)
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

        # add the energy consumption of auxiliary loads, model error and battery charging AND discharging efficiencies
        auxiliary = 700  # Watts or Joules per second
        average_model_error = -5.9  # percent
        discharging_battery_efficiency = 90
        self.data['P_total'] = ((self.data['P_regen'] + auxiliary) *
                                (100 + average_model_error) / 100) / ((self.charging_battery_efficiency / 100) *
                                                                      (discharging_battery_efficiency / 100))
        # P_total represents the amount of energy that is bought from the grid to ensure EV can perform (including ALL efficiencies)

    def pull_user_config(self):
        """
        Method to update the variable self.config_dict with the new user configuration

        :param data: config_path (the path for the JSON containing the user's system configuration)
        :return: - 
        """
        # open the json file and load the object into a python dictionary
        with open(self.config_path) as f:
            self.config_dict = json.load(f)
        # maybe have a seperate file to initialise config parameters based on user input, like calculating emergency reserves based on location

    def push_user_config(self):
        """
        Method to update the json file with the new user configuration

        :param data: - (because self.pull_user_config has already kept track of the file path)
        :return: -
        """
        with open(self.config_path, 'w') as f:
            json.dump(self.config_dict, f, indent=2)

    def charge(self, charge_time):
        """
        Method to charge EV battery, accounting for battery efficiency
        :param charge_time: charge_time in minutes
        :return: new SOC for the EV object
        """
        # get updated vehicle info
        self.pull_user_config()
        # maximum SOC to ensure safe operation (probably redundant unless recommend buffer fails)
        max_soc = 100  # in %
        max_charge_lvl = (max_soc / 100) * self.config_dict['EV_info']['Capacity']

        power_in_joules = self.config_dict['Charger_power'] * charge_time * 60

        n_battery = 90  # battery efficiency
        Wh_to_J = 3600
        power = (power_in_joules / Wh_to_J) * (n_battery / 100)  # convert joules to Wh

        if (max_charge_lvl - self.config_dict['Charge_level']) >= power:
            self.config_dict['Charge_level'] += power
        else:
            temp = self.config_dict['Charge_level'] + power - max_charge_lvl
            self.excess += temp
            print('Battery is full, {} Wh of excess energy'.format(temp))
            self.config_dict['Charge_level'] = max_charge_lvl

        # calculates the new instantaneous SOC
        self.config_dict['SOC'] = (self.config_dict['Charge_level'] / self.config_dict['EV_info']['Capacity']) * 100

        # updates json
        self.push_user_config()

    def discharge(self, power_in_joules):
        """
        Method to discharge EV battery, accounting for battery efficiency
        :param data: power including charging AND discharging efficiencies (because data is measured every second)
        :return: new SOC for the EV object
        """
        # get updated vehicle info
        self.pull_user_config()

        min_charge_lvl = 0

        # MAYBE have n_battery as a feature in the EV model database (csv)
        n_battery = 90  # battery efficiency
        Wh_to_J = 3600
        journey_power = power_in_joules * (n_battery / 100) * (
                    self.charging_battery_efficiency / 100)  # power needed to move EV excluding battery eff
        # power deducted from battery, accounting for n_battery
        power = (journey_power / Wh_to_J) / (n_battery / 100)  # convert joules to Wh

        if power > (self.config_dict['Charge_level'] - min_charge_lvl + 0.015):
            temp = power - (self.config_dict['Charge_level'] - min_charge_lvl)
            self.deficit += temp
            print('Battery is COMPLETELY drained, {} Wh of energy deficit'.format(temp))
            self.config_dict['Charge_level'] = min_charge_lvl
        elif power > (self.config_dict['Charge_level'] - min_charge_lvl):
            # to account for the difference in the decimal place that the JSON stores for charge level
            self.config_dict['Charge_level'] = min_charge_lvl
        else:
            self.config_dict['Charge_level'] -= power

        # calculates the new instantaneous SOC
        self.config_dict['SOC'] = (self.config_dict['Charge_level'] / self.config_dict['EV_info']['Capacity']) * 100

        # updates json
        self.push_user_config()

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
            timeseries_soc.append(self.config_dict['SOC'])
            timeseries_charge_lvl.append(self.config_dict['Charge_level'])
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
