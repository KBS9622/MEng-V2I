import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta


class charging_recommendation(object):

    def __init__(self, EV_data, TOU_data, previous_EV_data):
        self.EV_data = EV_data
        # self.TOU_data = TOU_data
        self.journey_start, self.journey_end = self.find_journey_start_and_end_points(data=self.EV_data)
        # self.previous_end = None
        # if system inputs the starting scenario, then create these variables
        # must input starting scenario when creating object
        self.previous_end = [previous_EV_data.iloc[-1, :].name]
        #temp code:
        self.TOU_data = TOU_data.loc[previous_EV_data.iloc[-1, :].name:, :]

    def set_EV_data(self, new_EV_data):
        """
        Method to update the variable self.EV_Data and use self.EV_data to determine reference point for charging availability
        
        :param data: new_EV_data (the predicted drive cycle)
        :return: -
        """
        # we could have the simulation file determine the span of prediction and load drive cycles based on the span
        self.previous_end = [self.EV_data.iloc[-1, :].name]
        self.charging_time_start = self.journey_end[-1]
        self.EV_data = new_EV_data
        self.journey_start, self.journey_end = self.find_journey_start_and_end_points(data=self.EV_data)
        self.charging_time_end = self.journey_start[0]
        #may need some work

    def set_TOU_data(self, new_TOU_data):
        """
        Method to update the variable self.TOU_Data (which always starts at the first availability of charging)

        :param data: new_TOU_data (the actual and possibly predicted TOU prices appended together)
        :return: -
        """
        self.TOU_data = new_TOU_data

    def update_user_config(self, config_path):
        """
        Method to update the variable self.config_dict with the new user configuration

        :param data: config_path (the path for the JSON containing the user's system configuration)
        :return: -
        """
        # open the json file and load the object into a python dictionary
        with open(config_path) as f:
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

            # TEMPORARY CODE:
            # UNCOMMENT line below if (assumption: EV available to charge BEFORE first journey and AFTER last journey of the day)
            if prev_end:
                if prev_end.date() == start.date():
                    pred.loc[np.logical_and(pred.index >= prev_end_time_slot, pred.index < start.ceil(freq='30min')), ['charging', 'journey']] = 30

            # find correct time slots for start and end points
            start_time_slot = start.floor(freq='30min')
            end_time_slot = end.floor(freq='30min')

            # fully fill any time slots between start_time_slot and end_time_slot
            pred.loc[np.logical_and(pred.index > start_time_slot, pred.index < end_time_slot), ['charging', 'journey']] = 30

            # Partially fill start_time_slot and end_time_slot based on journey time
            if start_time_slot == end_time_slot:
                pred.loc[start_time_slot, ['charging', 'journey']] += (end - start).seconds / 60
            else:
                pred.loc[start_time_slot, ['charging', 'journey']] += (start.ceil(freq='30min') - start).seconds / 60
                pred.loc[end_time_slot, ['charging', 'journey']] += (end - end_time_slot).seconds / 60
            
            # TEMPORARY CODE:
            # COMMENT lines below if (assumption: EV available to charge when EV is stationary)
            prev_end = end
            prev_end_time_slot = end_time_slot

        return pred

    def recommend(self):
        """
        Method to recommend charging times (Scheduler Module)

        :param data: JSON containing configuration parameters
        :return: recommmended charging slots
        """
        # **** maybe before the system runs the scheduler, it should use API to request up to date info from EV, like SOC and charge level
        # could add limit to when the EV is actually home and ready to charge (by reading the current passing through charger when initially plugged in)
        Wh_to_J = 3600
        initial_charge = self.config_dict['Charge_level'] * Wh_to_J
        battery_capacity = self.config_dict['EV_info']['Capacity'] * Wh_to_J
        # lower limit can also include the buffer the manufacturers set *if the charge level obtained from EV has not considered that yet
        lower_limit = battery_capacity * (self.config_dict['Lower_buffer'] + self.config_dict['Emergency_reserves']) / 100  
        # does not consider the manufacturer upper buffer yet
        upper_limit = battery_capacity * (self.config_dict['Upper_buffer'] / 100)  
        available_charge = initial_charge - lower_limit
        expected_charge = initial_charge

        # Copy the TOU_data df and create two new columns to be filled
        temp_pred = self.TOU_data.copy()
        temp_pred['charging'] = 0
        temp_pred['journey'] = 0

        # UNCOMMENT line below if (assumption: EV available to charge BEFORE first journey and AFTER last journey of the day)
        pred = self.charging_slot_availability(pred = temp_pred)
        
        # iterate through all pairs of journey start and end points
        for start, end in zip(self.journey_start, self.journey_end):

            # calculate total energy consumption for the journey
            journey_energy_consumption = sum(self.EV_data.loc[start:end]['P_total'])  # given in Joules
            print('journey energy consumption: {}'.format(journey_energy_consumption))

            # -> this is where the SOC (charge level) consideration takes place (Boon)
            if available_charge >= journey_energy_consumption:
                # reduce the available charge of EV by the amount needed for this journey
                available_charge = available_charge - journey_energy_consumption
                continue
            else:
                # reduce the additional charge needed to allow EV to complete this journey
                journey_energy_consumption = journey_energy_consumption - available_charge
                available_charge = 0
            expected_charge = expected_charge + journey_energy_consumption  # keep track of expected charge level after charging
            print('expected charge: {}'.format(expected_charge))

            charge_time = journey_energy_consumption / (self.config_dict['Charger_power'] * 60)  # gives charging time in minutes

            # calculate number of time slots needed to charge EV
            quotient = int(charge_time // 30)  # full slots
            remainder = charge_time % 30

            # ignore any full slots and sort TOU slots by price
            free_time_slots = pred.loc[np.logical_and(pred.index < start, pred['charging'] < 30)].copy()
            free_time_slots = free_time_slots.sort_values(by=['TOU'])
            print(free_time_slots)

            # exception handling
            if charge_time + sum(free_time_slots['charging']) > 30 * len(free_time_slots):
                print('Not enough time slots to charge')
                return None

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

        # add TOU and SOC (charge level) consideration here (Boon)
        # ASSUMPTION: charging rate is not dependent on current SOC. For more accurate result, implement a charging curve for the specific vehicle
        charge_per_timeslot = self.config_dict['Charger_power'] * 60 * 30  # in J for each 30 min timeslot
        # charge the EV if the additional charge does not cause charge level to exceed limit AND if the price of the timeslot is below threshold
        while expected_charge < upper_limit:
            # ignore any full slots and sort TOU slots by price
            free_time_slots = pred.loc[pred['charging'] < 30].copy()
            free_time_slots = free_time_slots.sort_values(by=['TOU'])
            # if the cheapest TOU slot is below threshold, charge for that slot
            if free_time_slots.iloc[0]['TOU'] <= self.config_dict['TOU_threshold']:
                # if the cheapest slot is partially filled, then calculate how much charge the system would add for the rest of the timeslot
                if free_time_slots.iloc[0]['charging'] != 0:
                    remainder = free_time_slots.iloc[0]['charging']
                    charge_time = 30 - remainder
                    # if charging for the remainder of the slot will NOT exceed upper limit, allocate full slot
                    if (upper_limit - expected_charge) >= (self.config_dict['Charger_power'] * 60 * charge_time):
                        pred.loc[free_time_slots.iloc[[0]].index, 'charging'] = 30
                        expected_charge += (self.config_dict['Charger_power'] * 60 * charge_time)
                    # if charging for the remainder of the slot WILL exceed upper limit, charge up to upper limit
                    else:
                        charge_time = (upper_limit - expected_charge) / (self.config_dict['Charger_power'] * 60)
                        pred.loc[free_time_slots.iloc[[0]].index, 'charging'] = charge_time + remainder
                        expected_charge = upper_limit
                else:
                    # if charging for the full slot will NOT exceed upper limit, allocate full slot
                    if (upper_limit - expected_charge) >= charge_per_timeslot:
                        # fill up entire empty slot
                        pred.loc[free_time_slots.iloc[[0]].index, 'charging'] = 30
                        expected_charge += charge_per_timeslot
                    # if charging for the full slot WILL exceed upper limit, allocate partial slot to charge up to upper limit
                    else:
                        charge_time = (upper_limit - expected_charge) / (self.config_dict['Charger_power'] * 60)
                        pred.loc[free_time_slots.iloc[[0]].index, 'charging'] = charge_time
                        expected_charge = upper_limit
                # include the charge curve in the loop to update charge_per_timeslot according to current SOC
            else:
                print('TOU exceeds threshold, no additional charge slots allocated.')
                break

        # TOU threshold charging (old implementation without SOC consideration)
        # pred.loc[pred['TOU'] <= threshold, 'charging'] = 30

        # subtract journey time from charging time
        pred['charging'] -= pred['journey']
        self.EV_data['charging'] = pred['charging']

        return pred.loc[pred['charging'] > 0, 'charging']