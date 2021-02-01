import pandas as pd
from EV_data_analysis import EV
from TOU_analysis_and_prediction import TOU
from charging_recommendation import charging_recommendation
from pandas.tseries.offsets import DateOffset

class Simulation:
    def __init__(self, drive_cycle_file, drive_cycle_subdir, config_path, tou_file, tou_subdir, train_tou):
        self.min_tou_threshold = 0
        self.tou_obj = TOU(tou_file, tou_subdir)
        if train_tou:
            self.tou_obj.create_and_fit_model()
        self.ev_obj = EV(drive_cycle_file, drive_cycle_subdir, config_path)
        self.config_path = config_path
        self.beginning_of_time = pd.to_datetime('2019-10-07 00:00:00')
        self.start_next_day = self.beginning_of_time
        self.energy_bought = []
        self.energy_cost = []

        # If plot needed columns to drop in format_ev_data needs to be edited
        self.ev_obj.data = self.format_ev_data(beginning_of_time=pd.to_datetime('2019-09-25 00:00:00'))
        self.recommendation_obj = None
        # self.graph_plotter(drive_cycle_file, drive_cycle_subdir)

    def plugged_in(self):
        """
        Signifies that the EV is plugged in, adds 1 day to start_next_day index and call method run_recommendation_algorithm()
        :return: 
        """
        # Add 1 day to start_next_day index
        self.start_next_day += pd.DateOffset(1)
        #call method run_recommendation_algorithm()
        recommended_slots = self.run_recommendation_algorithm()
        # sum the charge time allocated in each slot and charge with method charge()
        total_charge_time = sum(recommended_slots)
        # print(total_charge_time)
        self.ev_obj.charge(total_charge_time)
        # calls method calculate_cost_and_energy()
        df_price_and_time = self.calculate_cost_and_energy(recommended_slots)
        # prints total energy bought for the session and the cost, then appends a list to keep track of charging sessions throughout simulation
        print(f"The total energy bought for charging session is: {sum(df_price_and_time['energy_per_time_slot (kWh)'])} kWh ")
        print(f"The total cost for charging session is: {sum(df_price_and_time['cost_per_time_slot (p)'])} p ")
        self.energy_bought.append(sum(df_price_and_time['energy_per_time_slot (kWh)']))
        self.energy_cost.append(sum(df_price_and_time['cost_per_time_slot (p)']))

    def calculate_cost_and_energy(self, time_slots_charging):
        """
        grabs corresponding energy prices for allocated charging slots and 
        calculates energy bought per slot with charger power and cost of the energy per slot
        :return: timestamp indexed dataframe with 4 columns (charging[charge time allocated], TOU[price of energy], energy_per_time_slot (kWh), cost_per_time_slot (p))
        """
        # determines the timeslots where charging is scheduled for and gets corresponding TOU prices for those timeslots
        time_to_grab = time_slots_charging.index.tolist()
        df_price_per_kwh = self.recommendation_obj.TOU_data.loc[time_to_grab]
        # concatenates allocated charging time and TOU prices into a new df variable
        df_price_and_time = pd.concat([time_slots_charging, df_price_per_kwh], axis=1)
        # uses charger power to determine energy bought per slot and then calculate cost of buying that energy per slot
        rated_charging_power = self.ev_obj.config_dict['Charger_power'] / 1000 #in kW
        df_price_and_time["energy_per_time_slot (kWh)"] = (df_price_and_time['charging'] / 60) * rated_charging_power
        df_price_and_time["cost_per_time_slot (p)"] = df_price_and_time["energy_per_time_slot (kWh)"] * df_price_and_time['TOU']

        print(df_price_and_time)
        # total_cost_day = sum(df_price_and_time["cost_per_time_slot (p)"])
        return df_price_and_time

    def trigger_discharge(self):
        """
        Reduce the charge level of battery by daily power consumption amount
        :return:
        """
        # calculates and print the total energy needed by EV to move as described by drive cycle, accounting for charging AND discharging efficiencies
        print("Total Energy Consumed for day: ", self.start_next_day)
        print(self.recommendation_obj.EV_data)
        total_energy = self.recommendation_obj.EV_data['P_total'].sum()
        print(total_energy)
        # converts the sum from above to Wh and removes battery charging efficiency to SHOW how many Wh to deduct from battery (as P_total accounts for both battery efficiencies)
        Wh_to_J = 3600
        power = (total_energy / Wh_to_J) * (self.ev_obj.charging_battery_efficiency / 100)
        print("Subtracting ", (power), "Wh from battery")
        # calls method discharge() from ev_obj (an object of class EV)
        # Note: the method below uses 'total_energy' instead of the variable 'power' that was calculated above
        self.ev_obj.discharge(total_energy)

    def create_recommendation_obj(self):
        """
        
        :return: 
        """
        previous_ev_data = self.get_ev_data(start_time=self.beginning_of_time,
                                            end_time=self.beginning_of_time + pd.offsets.Hour(24) - pd.offsets.Second(1))
        predicted_tou_data = self.get_tou_data(start_time=self.beginning_of_time,
                                               end_time=self.beginning_of_time + pd.offsets.Hour(48) - pd.offsets.Minute(30))
        ev_consumption_data = self.get_ev_data(start_time=self.beginning_of_time + pd.offsets.Hour(24),
                                               end_time=self.beginning_of_time + pd.offsets.Hour(48) - pd.offsets.Second(1))
        recommendation_obj = charging_recommendation(ev_consumption_data, predicted_tou_data, previous_ev_data, self.config_path)
        return recommendation_obj

    def run_recommendation_algorithm(self):
        """
        
        :return: 
        """
        start_time = self.start_next_day
        end_time = self.start_next_day + pd.offsets.Hour(24) - pd.offsets.Second(1)
        if self.recommendation_obj:
            self.recommendation_obj.set_EV_data(self.get_ev_data(
                start_time=start_time,
                end_time=end_time))

            tou_end_time = self.recommendation_obj.charging_time_start.replace(hour=23, minute=30, second=0) + \
                           pd.DateOffset(1)
            self.recommendation_obj.set_TOU_data(self.get_tou_data(
                start_time=self.recommendation_obj.charging_time_start,
                end_time=tou_end_time))
        else:
            self.recommendation_obj = self.create_recommendation_obj()
        self.recommendation_obj.pull_user_config()
        print('Manual Override: ',self.recommendation_obj.config_dict['Manual_override'])
        if self.recommendation_obj.config_dict['Manual_override']:
            return self.recommendation_obj.uncontrolled()
        else:
            return self.recommendation_obj.recommend()

    def get_ev_data(self, start_time, end_time):
        return self.ev_obj.data.loc[start_time:end_time, :]

    def format_ev_data(self, beginning_of_time):
        """
        
        :return: 
        """
        cols_to_drop = ['speed_mps', 'accel_mps2', 'P_wheels', 'P_electric_motor', 'n_rb', 'P_regen']

        p_total = self.ev_obj.data.copy()
        p_total = p_total.drop(columns=cols_to_drop)
        p_total = p_total.set_index('timestamp')
        p_total = p_total.set_index(p_total.index
                                    + DateOffset(days=(beginning_of_time.floor(freq='D')
                                                       - p_total.iloc[0].name.floor(freq='D')).days))
        return p_total

    def get_tou_data(self, start_time=pd.to_datetime('2019-01-31 00:00:00'),
                     end_time=pd.to_datetime('2019-01-31 23:30:00')):
        """
        
        :param start_time: 
        :param end_time: 
        :return: 
        """
        self.start_time = start_time
        self.end_time = end_time
        # predicted_tou = self.tou_obj.predict_and_compare(self.start_time, self.end_time)
        # not using predicted, using actual values ... complete line 95 to do so
        self.format_tou_data()
        predicted_tou = self.tou_obj.time_idx_TOU_price.loc[start_time:end_time, :]
        return predicted_tou

    def format_tou_data(self):
        """
        ONLY USE IF FEEDING IN REAL DATA and not predicted
        :param : 
        :return: 
        """
        self.tou_obj.time_idx_TOU_price.columns = ['TOU']

    def graph_plotter(self, file, subdir):
        """
        
        :return: 
        """
        y = ['P_electric_motor', 'speed_mps', 'P_regen', 'n_rb', 'soc', 'P_total']
        file_name = ['energy_consumption.png', 'speed_profile.png', 'energy_consumption_with_regen.png',
                     'n_rb.png', 'soc.png', 'total_energy_conumption.png']
        self.ev_obj.graph_plotter(y=y, file_name=file_name, subdir=subdir, date=file.strip('.csv'))

    def result_plotter(self):
        """
        
        :return: 
        """
        with_recommendation = ''
        without_recommendation = ''
        pass
