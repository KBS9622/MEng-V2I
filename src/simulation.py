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
        self.beginning_of_time = pd.to_datetime('2019-09-25 00:00:00')
        self.start_next_day = self.beginning_of_time
        self.energy_bought = []
        self.energy_cost = []
        self.charge_time = []
        self.charging_schedule = pd.DataFrame([])
        self.days_skipped = 0

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
        print(self.start_next_day.date())
        if self.start_next_day.date() in self.ev_obj.data.index.date:
            # call method run_recommendation_algorithm()
            recommended_slots = self.run_recommendation_algorithm()
            # calls method calculate_cost_and_energy()
            df_price_and_time = self.calculate_cost_and_energy(recommended_slots)
            # if there is any charging activity, print and save EV load profile
            if not df_price_and_time.empty:
                self.charging_schedule = self.charging_schedule.append(df_price_and_time)
                # print(self.charging_schedule)
                # saves the load profile for the allocated charge slots in a csv
                self.load_profile(df_price_and_time)
                # prints total energy bought for the session and the cost, then appends a list to keep track of charging sessions throughout simulation
                # print(
                #     f"The total energy bought for charging session is: {sum(df_price_and_time['energy_per_time_slot (kWh)'])} kWh ")
                # print(f"The total cost for charging session is: {sum(df_price_and_time['cost_per_time_slot (p)'])} p ")
                self.energy_bought.append(sum(df_price_and_time['energy_per_time_slot (kWh)']))
                self.energy_cost.append(sum(df_price_and_time['cost_per_time_slot (p)']))
                self.charge_time.append(sum(df_price_and_time['charging']))
            # sum the charge time allocated in each slot and charge with method charge()
            total_charge_time = sum(recommended_slots)
            self.ev_obj.charge(total_charge_time)
            print('Charge level at end of plugged_in',self.ev_obj.config_dict['Charge_level'])
            # if this loop condition is met, that means there is data for the day and it is not another inactive day
            # therefore we need to reset the days_skipped counter
            if self.days_skipped > 0:
                self.days_skipped = 0
        else:
            print('Skipping ahead to next day\n')
            self.days_skipped += 1
            self.plugged_in()

    def load_profile(self, df):
        """
        grabs corresponding EV load profile from battery_profile 
        based on the initial SOC/charge level and the total time allocated to charging
        :return: Exports a csv
        """
        # load the battery profile from csv
        battery_profile = pd.read_csv(self.ev_obj.config_dict['EV_info']['Battery_profile'])
        # get the index for the charge level value nearest to init_charge from the battery_profile df
        init_charge_idx = battery_profile.iloc[(battery_profile['Charge_level_based_on_SOC']-(self.ev_obj.config_dict['Charge_level'])).abs().argsort()[:1],-1].index.to_list()[0]
        # save to a new variable the charging time per timeslot in seconds (1 datapoint per second)
        charging_time_seconds = (df["charging"]*60).astype(int)
        # calculate the total charging time so that we know how many seconds of charging profile to get
        total_charge_time = sum(charging_time_seconds)
        # get the charger profile from the index of the initial charge to the initial charge index + the total charge time
        new_df = pd.DataFrame(battery_profile['Power'].iloc[init_charge_idx:(init_charge_idx+total_charge_time)]/(self.ev_obj.config_dict['Charger_efficiency']/100),columns=['Power'])
        # add the timeslot information for the 'power' column
        new_df['time_stamp'] = charging_time_seconds.loc[charging_time_seconds.index.repeat(charging_time_seconds)].index
        # reset the index
        new_df.reset_index(inplace=True)
        if self.recommendation_obj.config_dict['Manual_override']:
            mode = 'uncontrolled_'
        else:
            mode = 'IntelliCharga_'
        # csv file names based on the day of charging
        csv_name = mode+'EV_profile_for_'+str(self.start_next_day.date())+'.csv'
        # export the df to csv
        new_df.to_csv(csv_name)


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
        # load the battery profile from csv
        battery_profile = pd.read_csv(self.ev_obj.config_dict['EV_info']['Battery_profile'])
        # get the index for the charge level value nearest to init_charge from the battery_profile df
        init_charge_idx = battery_profile.iloc[(battery_profile['Charge_level_based_on_SOC']-(self.ev_obj.config_dict['Charge_level'])).abs().argsort()[:1],-1].index.to_list()[0]
        for x in range(len(df_price_and_time)):
            # iteratively go through all allocated slots and determine the energy bought within the time slot 
            # from the initial charge level and the time spent charging in that timeslot
            # find the amount of time allocated to charging in this timeslot, in seconds
            charge_time_seconds = int(60*df_price_and_time['charging'].iloc[x])
            # calculate the end index
            end_idx = init_charge_idx + charge_time_seconds
            # calculate the energy (in kWh) that would be bought (amount charged + loss) if the rest of the timeslot were to be allocated 
            df_price_and_time.loc[df_price_and_time.index[x],"energy_per_time_slot (kWh)"] = (battery_profile.iloc[end_idx,-1] - battery_profile.iloc[init_charge_idx,-1])/(1000*(self.ev_obj.config_dict['Charger_efficiency']/100))
            # calculate the cost to charge in that timeslot for the amount of energy bought
            df_price_and_time.loc[df_price_and_time.index[x],"cost_per_time_slot (p)"] = df_price_and_time["energy_per_time_slot (kWh)"].iloc[x] * df_price_and_time["TOU"].iloc[x]
            # update the inital charge index for next timeslot
            init_charge_idx = end_idx

        # print(df_price_and_time)
        return df_price_and_time

    def trigger_discharge(self):
        """
        Reduce the charge level of battery by daily power consumption amount\
        Will not be need if SOC data is received from OBD, if not available will run at the end of the day or as data is
        streamed in from OBD
        :return:
        """
        # calculates and print the total energy needed by EV to move as described by drive cycle, accounting for
        # charging AND discharging efficiencies
        print("Total Energy Consumed for day: ", self.start_next_day)
        # BOON: right now it uses data in recommendation obj, which is predicted and not actual, therefore should use
        # data from ev_obj
        start_time = self.start_next_day
        end_time = self.start_next_day + pd.offsets.Hour(24) - pd.offsets.Second(1)
        yesterdays_ev_data = self.get_ev_data(start_time=start_time, end_time=end_time)
        # print(yesterdays_ev_data)

        total_energy = yesterdays_ev_data['P_total'].sum()
        # print(total_energy)

        # converts the sum from above to Wh and removes battery charging efficiency to SHOW how many Wh to deduct
        # from battery (as P_total accounts for both battery efficiencies)
        wh_to_j = 3600
        power = (total_energy / wh_to_j) * (self.ev_obj.charging_battery_efficiency / 100)
        print("Subtracting ", power, "Wh from battery")
        # calls method discharge() from ev_obj (an object of class EV)
        # Note: the method below uses 'total_energy' instead of the variable 'power' that was calculated above
        self.ev_obj.discharge(total_energy)
        print('Charge level at end of trigger_discharge',self.ev_obj.config_dict['Charge_level'])

    def create_recommendation_obj(self):
        """
        creates a recommendation object which needs to load in the 'previous day' drive cycle, 'predicted' drive
        cycle and tou for those period * this method should only be called once, to create the initial object
        :return: object of class 'charging_recommendation'
        """
        # gets ev data for previous day (to determine when EV is home, for charging recommendation)
        # BOON: this will be where ACTUAL drive cycle will be loaded since it is historic data
        previous_ev_data = self.get_ev_data(start_time=self.beginning_of_time,
                                            end_time=self.beginning_of_time + pd.offsets.Hour(24) - pd.offsets.Second(
                                                1))
        # get tou data ranging from before EV reaches home until the end of the predicted drive cycle
        # BOON: this may be where we need to modify to integrate TOU forecast module
        predicted_tou_data = self.get_tou_data(start_time=self.beginning_of_time,
                                               end_time=self.beginning_of_time + pd.offsets.Hour(
                                                   48) - pd.offsets.Minute(30))
        # gets the next day drive cycle (which should be predicted by the drive cycle forecast module) BOON: this
        # will be where we need to call the drive cycle prediction function (new method called 'predict_ev_data'?)
        ev_consumption_data = self.get_ev_data(start_time=self.beginning_of_time + pd.offsets.Hour(24),
                                               end_time=self.beginning_of_time + pd.offsets.Hour(
                                                   48) - pd.offsets.Second(1))
        # calls constructor for class 'charging_recommendation'
        recommendation_obj = charging_recommendation(ev_consumption_data, predicted_tou_data, previous_ev_data,
                                                     self.config_path)
        return recommendation_obj

    def run_recommendation_algorithm(self):
        """
        creates/updates 'charging_recommendation' object, then run the recommendation algorithms based on user configuration
        :return: 'charging_recommendation' object with variables updated with allocated charging slots and etc
        """
        start_time = self.start_next_day
        end_time = self.start_next_day + pd.offsets.Hour(24) - pd.offsets.Second(1)
        previous_start_time = start_time - pd.DateOffset(1+self.days_skipped)
        previous_end_time = end_time - pd.DateOffset(1)
        # if recommendation object already exist
        if self.recommendation_obj:
            # calls method 'set_EV_data' to update 'predicted' drive cycle
            # BOON: this line will need to call the predict_ev_data as first input!!!
            self.recommendation_obj.set_EV_data(self.get_ev_data(start_time=start_time, end_time=end_time),
                                                self.get_ev_data(start_time=previous_start_time,
                                                                 end_time=previous_end_time))
            # determine the end range of TOU feed in
            tou_end_time = self.recommendation_obj.charging_time_start.replace(hour=23, minute=30, second=0) + \
                           pd.DateOffset(1+self.days_skipped)
            # calls method 'set_TOU_data' to update future TOU prices
            self.recommendation_obj.set_TOU_data(self.get_tou_data(
                start_time=self.recommendation_obj.charging_time_start,
                end_time=tou_end_time))
        else:
            # create a new recommendation object if it does not exist
            self.recommendation_obj = self.create_recommendation_obj()
        # gets user configuration for system by calling method 'pull_user_config()
        self.recommendation_obj.pull_user_config()
        print('Manual Override: ', self.recommendation_obj.config_dict['Manual_override'])
        # run uncontrolled charging if user chooses to manual override, otherwise run IntelliCharga algorithm
        if self.recommendation_obj.config_dict['Manual_override']:
            return self.recommendation_obj.uncontrolled()
        else:
            return self.recommendation_obj.recommend()

    def predict_ev_data(self, start_time, end_time):
        """
        predicts the slots of ev drive cycle data indicated by the start_time and end_time
        :return: 'EV' object with updated predicted drive cycle data
        """
        # BOON: this will probably be where we use Heejoon's prediction method/function
        # maybe have this method trigger the prediction, then create a method that will load in the ACTUAL data for the rest of simulation
        # eg: when actually DISCHARGING EV, it should be based on actual data and not predicted, as predicted is only for CHARGING
        pass
        # return self.ev_obj.data.loc[start_time:end_time, :]

    def get_ev_data(self, start_time, end_time):
        """
        gets the slots of ev drive cycle data indicated by the start_time and end_time
        :return: 'EV' object with updated drive cycle data
        """
        # this gets actual data, therefore may be useful to keep most of code the same and 
        # only swap method calls for when the use of drive cycle data is for prediction/estimation
        return self.ev_obj.data.loc[start_time:end_time, :]

    def format_ev_data(self, beginning_of_time):
        """
        removes unnecesary columns from dataframe 'data' of class 'EV' object
        :return: dataframe which contains only desired column(s) ('P_total' and timestamped index)
        """
        # determines columns to drop that are of no use for recommendation algorithm (columns that were used by ECM to calculate 'P_total')
        cols_to_drop = ['speed_mps', 'accel_mps2', 'P_wheels', 'P_electric_motor', 'n_rb', 'P_regen']
        # copies the current dataframe 'data' in ev_obj and drops columns then indexes with timestamps
        p_total = self.ev_obj.data.copy()
        p_total = p_total.drop(columns=cols_to_drop)
        p_total = p_total.set_index('time_stamp')
        p_total = p_total.set_index(p_total.index
                                    + DateOffset(days=(beginning_of_time.floor(freq='D')
                                                       - p_total.iloc[0].name.floor(freq='D')).days))
        return p_total

    def get_tou_data(self, start_time=pd.to_datetime('2019-01-31 00:00:00'),
                     end_time=pd.to_datetime('2019-01-31 23:30:00')):
        """
        gets the slots of tou price data indicated by the start_time and end_time
        :param start_time: 
        :param end_time: 
        :return: dataframe of tou prices from start_time to end_time
        """
        self.start_time = start_time
        self.end_time = end_time
        # BOON: this will probably be where we use Atom's prediction method/function
        # predicted_tou = self.tou_obj.predict_and_compare(self.start_time, self.end_time)
        # not using predicted, using actual values ... complete line 95 to do so
        self.format_tou_data()  # adds column names for when using actual prices, comment if using predicted prices
        # gets the tou prices from start_time to end_time
        predicted_tou = self.tou_obj.time_idx_TOU_price.loc[start_time:end_time, :]
        return predicted_tou

    def format_tou_data(self):
        """
        ONLY USE IF FEEDING IN REAL DATA and not predicted
        :param : 
        :return: 
        """
        # labels the column
        self.tou_obj.time_idx_TOU_price.columns = ['TOU']

    def graph_plotter(self, file, subdir):
        """
        plots all the features indicated by variable 'y' and saves the graphs
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
