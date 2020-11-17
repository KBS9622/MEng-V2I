import pandas as pd
import numpy as np
import json

class charging_recommendation(object):

    # def __init__(self, initial_charge, EV_data, TOU_data):

    def __init__(self, EV_data, TOU_data):
        # self.available_charge = initial_charge
        self.EV_data = EV_data
        self.TOU_data = TOU_data
        self.journey_start, self.journey_end = self.find_journey_start_and_end_points()

    def find_journey_start_and_end_points(self, min_gap=30):
        """
        Method to indetify sub-journey start/end times (Classifier Module)

        :param data: min_gap (minimum time gap to identify as two seperate sub-journey)
        :return: journey_start varible (with sub-journey start times) and journey_end variable (with corresponding sub-journey end times)
        """

        P_total_data = self.EV_data.copy()
        min_gap = pd.Timedelta(min_gap*60, unit='sec')

        journey_start = [P_total_data.iloc[0, :].name]
        journey_end = [P_total_data.iloc[-1, :].name]

        for i in range(0, len(P_total_data) - 1):
            time_diff = P_total_data.iloc[i + 1, :].name - P_total_data.iloc[i, :].name
            if time_diff >= min_gap:
                journey_start.append(P_total_data.iloc[i + 1, :].name)
                journey_end.insert(len(journey_end)-1, P_total_data.iloc[i, :].name)

        return journey_start, journey_end

    def recommend(self, config_path):
        """
        Method to recommend charging times (Scheduler Module)

        :param data: JSON containing configuration parameters
        :return: recommmended charging slots
        """
        # **** maybe before the system runs the scheduler, it should use API to request up to date info from EV, like SOC and charge level
        # could add limit to when the EV is actually home and ready to charge (by reading the current passing through charger when initially plugged in)
        # open the json file and load the object into a python dictionary
        with open(config_path) as f:
            config_dict = json.load(f)
        threshold = config_dict['TOU_threshold']
        charger_power = config_dict['Charger_power']
        lower_buffer = config_dict['Lower_buffer']
        upper_buffer = config_dict['Upper_buffer']
        emergency_reserves = config_dict['Emergency_reserves']
        home_location = config_dict['Home_location'] #maybe have a seperate file to initialise config parameters based on user input, like calculating emergency reserves based on location
        manual_override = config_dict['Manual_override']
        initial_charge = config_dict['Charge_level']
        battery_capacity = config_dict['Capacity']

        lower_limit = lower_buffer + emergency_reserves # lower limit can also include the buffer the manufacturers set *if the charge level obtained from EV has not considered that yet
        upper_limit = battery_capacity - upper_buffer #does not consider the manufacturer upper buffer yet
        available_charge = initial_charge - lower_limit
        expected_charge = initial_charge

        pred = self.TOU_data.copy()
        pred['charging'] = 0
        pred['journey'] = 0

        # iterate through all pairs of journey start and end points
        for start, end in zip(self.journey_start, self.journey_end):

            # find correct time slots for start and end points
            start_time_slot = start.floor(freq='30min')
            end_time_slot = end.floor(freq='30min')

            # fully fill any time slots between start_time_slot and end_time_slot
            pred.loc[np.logical_and(pred.index > start_time_slot, pred.index < end_time_slot), ['charging', 'journey']] = 30

            # fill start_time_slot and end_time_slot based on journey time
            if start_time_slot == end_time_slot:
                pred.loc[start_time_slot, ['charging', 'journey']] += (end - start).seconds / 60
            else:
                pred.loc[start_time_slot, ['charging', 'journey']] += (start.ceil(freq='30min') - start).seconds / 60
                pred.loc[end_time_slot, ['charging', 'journey']] += (end - end_time_slot).seconds / 60

            # calculate total energy consumption for the journey
            journey_energy_consumption = sum(self.EV_data.loc[start:end]['P_total'])

            # -> this is where the SOC (charge level) consideration takes place (Boon)
            if available_charge >= journey_energy_consumption:
                # reduce the available charge of EV by the amount needed for this journey
                available_charge = available_charge - journey_energy_consumption
                continue
            else:
                # reduce the additional charge needed to allow EV to complete this journey
                journey_energy_consumption = journey_energy_consumption - available_charge
            
            expected_charge = expected_charge + journey_energy_consumption #keep track of expected charge level after charging
            charge_time = journey_energy_consumption / (charger_power * 60)

            # calculate number of time slots needed to charge EV
            quotient = int(charge_time // 30) # full slots
            remainder = charge_time % 30

            # ignore any full slots and sort TOU slots by price
            free_time_slots = pred.loc[np.logical_and(pred.index < start, pred['charging'] < 30)].copy()
            free_time_slots = free_time_slots.sort_values(by=['TOU'])

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

        # add SOC (charge level) consideration here (Boon)
        # ASSUMPTION: charging rate is not dependent on current SOC. For more accurate result, implement a charging curve for the specific vehicle
        charge_per_timeslot = charger_power * 30 / 60 #in Wh
        # charge the EV if the additional charge does not cause charge level to exceed limit AND if the price of the timeslot is below threshold
        while expected_charge + charge_per_timeslot <= upper_limit:
            # ignore any full slots and sort TOU slots by price
            free_time_slots = pred.loc[np.logical_and(pred.index < start, pred['charging'] < 30)].copy()
            free_time_slots = free_time_slots.sort_values(by=['TOU'])
            # if the cheapest TOU slot is below threshold, charge for that slot
            if free_time_slots['TOU'].iloc[[0]] <= threshold:
                pred.loc[free_time_slots.iloc[[0]].index, 'charging'] = 30
                #include the charge curve in the loop to update charge_per_timeslot according to current SOC
            else:
                break


        # TOU threshold charging (old implementation without SOC consideration)
        # pred.loc[pred['TOU'] <= threshold, 'charging'] = 30

        # subtract journey time from charging time
        pred['charging'] -= pred['journey']
        self.EV_data['charging'] = pred['charging']

        return pred.loc[pred['charging'] > 0, 'charging']

    # def recommend(self, threshold=0, charger_power=3e3):
    #     """
    #     Method to recommend charging times (Scheduler Module)

    #     :param data: threshold (user-defined/system defined TOU threshold) and charger_power
    #     :return: recommmended charging slots
    #     """

    #     pred = self.TOU_data.copy()
    #     pred['charging'] = 0
    #     pred['journey'] = 0

    #     # iterate through all pairs of journey start and end points
    #     for start, end in zip(self.journey_start, self.journey_end):

    #         # find correct time slots for start and end points
    #         start_time_slot = start.floor(freq='30min')
    #         end_time_slot = end.floor(freq='30min')

    #         # fully fill any time slots between start_time_slot and end_time_slot
    #         pred.loc[np.logical_and(pred.index > start_time_slot, pred.index < end_time_slot), ['charging', 'journey']] = 30

    #         # fill start_time_slot and end_time_slot based on journey time
    #         if start_time_slot == end_time_slot:
    #             pred.loc[start_time_slot, ['charging', 'journey']] += (end - start).seconds / 60
    #         else:
    #             pred.loc[start_time_slot, ['charging', 'journey']] += (start.ceil(freq='30min') - start).seconds / 60
    #             pred.loc[end_time_slot, ['charging', 'journey']] += (end - end_time_slot).seconds / 60

    #         # calculate total charging time for the journey
    #         journey_energy_consumption = sum(self.EV_data.loc[start:end]['P_total'])
    #         # -> this is where the SOC consideration takes place (Boon)
    #         charge_time = journey_energy_consumption / (charger_power * 60)

    #         # calculate number of time slots needed to charge EV
    #         quotient = int(charge_time // 30) # full slots
    #         remainder = charge_time % 30

    #         # ignore any full slots and sort TOU slots by price
    #         free_time_slots = pred.loc[np.logical_and(pred.index < start, pred['charging'] < 30)].copy()
    #         free_time_slots = free_time_slots.sort_values(by=['TOU'])

    #         # exception handling
    #         if charge_time + sum(free_time_slots['charging']) > 30 * len(free_time_slots):
    #             print('Not enough time slots to charge')
    #             return None

    #         # fill in slots based on the quotient and remainder
    #         remainder += sum(free_time_slots.iloc[list(range(0, quotient))]['charging'])
    #         if quotient > 0: pred.loc[free_time_slots.iloc[list(range(0, quotient))].index, 'charging'] = 30
    #         idx_offset = 0
    #         while remainder != 0:
    #             remainder += free_time_slots.iloc[quotient + idx_offset]['charging']
    #             if remainder >= 30:
    #                 pred.loc[free_time_slots.iloc[quotient + idx_offset].name, 'charging'] = 30
    #                 remainder -= 30
    #             else:
    #                 pred.loc[free_time_slots.iloc[quotient + idx_offset].name, 'charging'] = remainder
    #                 remainder = 0

    #             idx_offset += 1

    #     # TOU threshold charging
    #     pred.loc[pred['TOU'] <= threshold, 'charging'] = 30

    #     # subtract journey time from charging time
    #     pred['charging'] -= pred['journey']
    #     self.EV_data['charging'] = pred['charging']

    #     return pred.loc[pred['charging'] > 0, 'charging']

