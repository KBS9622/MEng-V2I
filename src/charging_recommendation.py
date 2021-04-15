import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta


class charging_recommendation(object):

    def __init__(self, predicted_EV_data, TOU_data, previous_EV_data, config_path):
        self.predicted_EV_data = predicted_EV_data
        # self.TOU_data = TOU_data
        self.journey_start, self.journey_end = self.find_journey_start_and_end_points(data=self.predicted_EV_data)
        # self.previous_end = None
        # if system inputs the starting scenario, then create these variables
        # must input starting scenario when creating object
        self.previous_EV_data = previous_EV_data
        self.previous_end = [self.previous_EV_data.iloc[-1, :].name]
        self.config_path = config_path
        # temp code:
        self.TOU_data = TOU_data.loc[self.previous_EV_data.iloc[-1, :].name:, :]

    def set_EV_data(self, new_EV_data, previous_EV_data):
        """
        Method to replace predicted EV data with ACTUAL EV data once available, to identify previous_end
        update the variable self.predicted_EV_data and use self.predicted_EV_data to determine reference point for charging availability
        
        :param data: new_EV_data (the predicted drive cycle)
        :return: -
        """
        # we could have the 'user_config.json' determine the span of prediction and load drive cycles based on the span 
        # BOON: there will be changes needed to this part so that charging_time_start is based on ACTUAL data but we
        # can load in predicted data
        self.previous_EV_data = previous_EV_data
        # earlist possible charge time is when the EV has just reached home, so the last data observation
        self.charging_time_start = self.previous_EV_data.iloc[-1, :].name
        self.predicted_EV_data = new_EV_data
        self.journey_start, self.journey_end = self.find_journey_start_and_end_points(data=self.predicted_EV_data)
        self.charging_time_end = self.journey_start[0]
        # may need some work

    def set_TOU_data(self, new_TOU_data):
        """
        Method to update the variable self.TOU_Data (which always starts at the first availability of charging)

        :param data: new_TOU_data (the actual and possibly predicted TOU prices appended together)
        :return: -
        """
        self.TOU_data = new_TOU_data

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

    def find_journey_start_and_end_points(self, data, min_gap=30):
        """
        Method to indetify sub-journey start/end times (Classifier Module)

        :param data: min_gap (minimum time gap to identify as two seperate sub-journey)
        :return: journey_start varible (with sub-journey start times) and journey_end variable (with corresponding sub-journey end times)
        """

        P_total_data = data.copy()
        min_gap = pd.Timedelta(min_gap * 60, unit='sec')

        journey_start = [P_total_data.iloc[0, :].name]
        journey_end = [P_total_data.iloc[-1, :].name]

        for i in range(0, len(P_total_data) - 1):
            time_diff = P_total_data.iloc[i + 1, :].name - P_total_data.iloc[i, :].name
            if time_diff >= min_gap:
                journey_start.append(P_total_data.iloc[i + 1, :].name)
                journey_end.insert(len(journey_end) - 1, P_total_data.iloc[i, :].name)

        return journey_start, journey_end

    def charging_slot_availability(self, pred):
        """
        Method to generate df with charging slot availability (Pre-Scheduler Module)
        - Value within each cell represents the unavailable time length (in min) for that slot

        :param data: pred (dataframe with timestamp and 3 columns [TOU price, charging, journey])
        :return: Filled out dataframe with timestamp and 3 columns (TOU price, charging, journey)
        """
        prev_end = None

        for start, end in zip(self.journey_start, self.journey_end):

            # find correct time slots for start and end points
            start_time_slot = start.floor(freq='30min')
            end_time_slot = end.floor(freq='30min')

            # fully fill any time slots between start_time_slot and end_time_slot
            pred.loc[
                np.logical_and(pred.index > start_time_slot, pred.index < end_time_slot), ['charging', 'journey']] = 30

            # Partially fill start_time_slot and end_time_slot based on journey time
            if start_time_slot == end_time_slot:
                pred.loc[start_time_slot, ['charging', 'journey']] += (end - start).seconds / 60
            else:
                pred.loc[start_time_slot, ['charging', 'journey']] += (start.ceil(freq='30min') - start).seconds / 60
                pred.loc[end_time_slot, ['charging', 'journey']] += (end - end_time_slot).seconds / 60

            # TEMPORARY CODE:
            # UNCOMMENT line below if (assumption: EV available to charge BEFORE first journey and AFTER last journey of the day)
            if prev_end:
                if prev_end.date() == start.date():
                    pred.loc[np.logical_and(pred.index >= prev_end_time_slot, pred.index < start.ceil(freq='30min')), [
                        'charging', 'journey']] = 30

            # TEMPORARY CODE:
            # COMMENT lines below if (assumption: EV available to charge when EV is stationary)
            prev_end = end
            prev_end_time_slot = end_time_slot

        return pred

    def fill_in_timeslot(self, charge_time, free_time_slots, pred):
        """
        Method to fill in charging time slots based on charge_time

        :param data: charge_time (in mins), free_time_slots (the available charging timeslots), pred (all timeslots of the prediction span)
        :return: pred with the charging time allocated
        """
        # calculate number of time slots needed to charge EV
        quotient = int(charge_time // 30)  # full slots
        remainder = charge_time % 30

        # fill in slots based on the quotient and remainder
        remainder += sum(free_time_slots.iloc[list(range(0, quotient))]['charging'])
        if quotient > 0: pred.loc[free_time_slots.iloc[list(range(0, quotient))].index, 'charging'] = 30
        idx_offset = 0
        while remainder != 0:
            remainder += free_time_slots.iloc[quotient + idx_offset]['charging']
            if remainder >= 30:
                pred.loc[free_time_slots.iloc[quotient + idx_offset].name, 'charging'] = 30
                remainder -= 30
            else:
                pred.loc[free_time_slots.iloc[quotient + idx_offset].name, 'charging'] = remainder
                remainder = 0

            idx_offset += 1

        return pred

    def uncontrolled(self):
        """
        Method to immitate uncontrolled charging

        :param data: -
        :return: uncontroled charging slots
        """
        # load the battery profile from csv
        battery_profile = pd.read_csv(self.config_dict['EV_info']['Battery_profile'])
        Wh_to_J = 3600
        # SOC and charge level below which uncontrolled charging starts
        SOC_threshold = 50
        charge_threshold = (SOC_threshold / 100) * self.config_dict['EV_info']['Capacity']
        # system should stop charging 30 min prior to next journey, for error purposes
        time_to_stop_charging = 30  

        # Copy the TOU_data df and create two new columns to be filled
        temp_pred = self.TOU_data.copy()
        temp_pred['charging'] = 0
        temp_pred['journey'] = 0
        pred = self.charging_slot_availability(pred=temp_pred)

        if self.config_dict['Charge_level'] < charge_threshold:

            # get the index for the 'energy' value closest to the expected_initial_charge value as a reference index
            nearest_start = battery_profile.iloc[(battery_profile['Charge_level_based_on_SOC']-self.config_dict['Charge_level']).abs().argsort()[:1],-1].index.to_list()[0]
            # get the index for the 'energy' value closest to the expected_charge value as a reference index
            nearest_end = battery_profile.iloc[(battery_profile['Charge_level_based_on_SOC']-self.config_dict['EV_info']['Capacity']).abs().argsort()[:1],-1].index.to_list()[0]
            # the time needed to charge up to the expected charge value from the expected initil charge value is just the difference of the index values, then convert from s to min
            charge_time = (nearest_end - nearest_start)/60

            start = self.journey_start[0]
            # ignore any full slots
            free_time_slots = pred.loc[np.logical_and(pred.index < start - timedelta(minutes=time_to_stop_charging),
                                                      pred['charging'] < 30)].copy()

            # Note: if not enough time slots, charge until plug out time
            # exception handling
            if charge_time + sum(free_time_slots['charging']) > 30 * len(free_time_slots):
                print('Not enough time slots to charge')
                # calculate how much time could be allocated for charging
                charge_time = (30 * len(free_time_slots)) - sum(free_time_slots['charging'])
                print('charging for : {} mins'.format(charge_time))
                # allocate the timeslots
                pred = self.fill_in_timeslot(charge_time=charge_time, free_time_slots=free_time_slots, pred=pred)
                # subtract journey time from charging time so that the df column only contains charging time
                pred['charging'] -= pred['journey']
                # save the column in the object df variable
                self.predicted_EV_data['charging'] = pred['charging']

                return pred.loc[pred['charging'] > 0, 'charging']

            # allocate the timeslots
            pred = self.fill_in_timeslot(charge_time=charge_time, free_time_slots=free_time_slots, pred=pred)

        # subtract journey time from charging time so that the df column only contains charging time
        pred['charging'] -= pred['journey']
        # save the column in the object df variable
        self.predicted_EV_data['charging'] = pred['charging']

        return pred.loc[pred['charging'] > 0, 'charging']
    
    def recommend(self):
        """
        Method to recommend charging times (Scheduler Module)

        :param data: -
        :return: recommmended charging slots
        """
        # load the battery profile from csv
        battery_profile = pd.read_csv(self.config_dict['EV_info']['Battery_profile'])
        # **** maybe before the system runs the scheduler, it should use API to request up to date info from EV, like SOC and charge level
        # could add limit to when the EV is actually home and ready to charge (by reading the current passing through charger when initially plugged in)
        Wh_to_J = 3600
        initial_charge = self.config_dict['Charge_level'] * Wh_to_J
        battery_capacity = self.config_dict['EV_info']['Capacity'] * Wh_to_J
        # lower limit can also include the buffer the manufacturers set *if the charge level obtained from EV has not considered that yet
        lower_limit = battery_capacity * (
                    self.config_dict['Lower_buffer'] + self.config_dict['Emergency_reserves']) / 100
        # does not consider the manufacturer upper buffer yet
        upper_limit = battery_capacity * (self.config_dict['Upper_buffer'] / 100)
        available_charge = initial_charge - lower_limit
        expected_charge = initial_charge

        # Copy the TOU_data df and create two new columns to be filled
        temp_pred = self.TOU_data.copy()
        temp_pred['charging'] = 0
        temp_pred['journey'] = 0

        # UNCOMMENT line below if (assumption: EV available to charge BEFORE first journey and AFTER last journey of the day)
        pred = self.charging_slot_availability(pred=temp_pred)

        # iterate through all pairs of journey start and end points
        for start, end in zip(self.journey_start, self.journey_end):

            # calculate total energy consumption for the journey
            sum_of_P_total = sum(self.predicted_EV_data.loc[start:end]['P_total'])  # given in Joules
            journey_energy_consumption = sum_of_P_total * (self.config_dict[
                                                               'Charger_efficiency'] / 100)  # given in Joules, this value is the value to be deducted from the battery
            # print('journey energy consumption including discharging efficiency: {} Wh'.format(journey_energy_consumption/3600))

            # -> this is where the SOC (charge level) consideration takes place (Boon)
            if available_charge >= journey_energy_consumption:
                # reduce the available charge of EV by the amount needed for this journey
                available_charge = available_charge - journey_energy_consumption
                continue
            else:
                # reduce the additional charge needed to allow EV to complete this journey
                journey_energy_consumption = journey_energy_consumption - available_charge
                available_charge = 0
                if journey_energy_consumption < 0.015:
                    # this is to account for the difference in the decimal place that the JSON stores for charge level
                    journey_energy_consumption = 0
            expected_initial_charge = expected_charge # this is the charge level, assuming previously allocated slots have already charged (for the purpose of battery_profile)
            expected_charge = expected_charge + journey_energy_consumption  # keep track of expected charge level after charging
            # print('expected charge for {}: {} Wh'.format(start, expected_charge/3600))

            # get the index for the 'energy' value closest to the expected_initial_charge value as a reference index
            nearest_start = battery_profile.iloc[(battery_profile['Charge_level_based_on_SOC']-(expected_initial_charge/3600)).abs().argsort()[:1],-1].index.to_list()[0]
            # get the index for the 'energy' value closest to the expected_charge value as a reference index
            nearest_end = battery_profile.iloc[(battery_profile['Charge_level_based_on_SOC']-(expected_charge/3600)).abs().argsort()[:1],-1].index.to_list()[0]
            # the time needed to charge up to the expected charge value from the expected initil charge value is just the difference of the index values, then convert from s to min
            charge_time = (nearest_end - nearest_start)/60

            # ignore any full slots and sort TOU slots by price
            free_time_slots = pred.loc[np.logical_and(pred.index < start, pred['charging'] < 30)].copy()
            free_time_slots = free_time_slots.sort_values(by=['TOU'])
            # print(free_time_slots)

            # exception handling
            if charge_time + sum(free_time_slots['charging']) > 30 * len(free_time_slots):
                print('Not enough time slots to charge')
                return None

            pred = self.fill_in_timeslot(charge_time=charge_time, free_time_slots=free_time_slots, pred=pred)

        # charge the EV if the additional charge does not cause charge level to exceed limit AND if the price of the timeslot is below threshold
        while expected_charge < upper_limit:
            # ignore any full slots and sort TOU slots by price
            free_time_slots = pred.loc[pred['charging'] < 30].copy()
            free_time_slots = free_time_slots.sort_values(by=['TOU'])

            # if the cheapest TOU slot is below threshold, charge for that slot
            if free_time_slots.iloc[0]['TOU'] <= self.config_dict['TOU_threshold']:
                # determine how much time has already been allocated for the cheapest available timeslot
                remainder = free_time_slots.iloc[0]['charging']
                # calculate how much time (in minutes) left in the timeslot to allocate charge
                charge_time = 30 - remainder
                # calculate the charge time in seconds so that we can look up the time in battery_profile
                charge_time_seconds = int(charge_time*60)
                # find the amount of energy that could be charged in this timeslot based on expected charge level after all previous charging allocations
                # get the index for the 'energy' value closest to the expected charge value as a reference index
                expected_charge_idx = battery_profile.iloc[(battery_profile['Charge_level_based_on_SOC']-(expected_charge/3600)).abs().argsort()[:1],-1].index.to_list()[0]
                # calculate the energy (in joules) that would be charged if the rest of the timeslot were to be allocated 
                charge_for_timeslot = 3600*(battery_profile.iloc[(expected_charge_idx+charge_time_seconds),-1] - battery_profile.iloc[(expected_charge_idx),-1])

                # if charging for the remainder of the slot will NOT exceed upper limit, allocate full slot
                if (upper_limit - expected_charge) >= charge_for_timeslot:
                    pred.loc[free_time_slots.iloc[[0]].index, 'charging'] = 30
                    expected_charge += charge_for_timeslot
                # if charging for the remainder of the slot WILL exceed upper limit, charge up to upper limit
                else:
                    # get the index for the 'energy' value closest to the upper limit value as a reference index
                    upper_limit_idx = battery_profile.iloc[(battery_profile['Charge_level_based_on_SOC']-(upper_limit/3600)).abs().argsort()[:1],-1].index.to_list()[0]
                    charge_time = (upper_limit_idx - expected_charge_idx)/60
                    pred.loc[free_time_slots.iloc[[0]].index, 'charging'] = charge_time + remainder
                    expected_charge = upper_limit

            else:
                print('TOU exceeds threshold, no additional charge slots allocated.')
                break

        # subtract journey time from charging time
        pred['charging'] -= pred['journey']
        self.predicted_EV_data['charging'] = pred['charging']

        return pred.loc[pred['charging'] > 0, 'charging']
