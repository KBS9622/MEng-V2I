import pandas as pd
import numpy as np

class charging_recommendation(object):

    def __init__(self, EV_data, TOU_data):

        self.EV_data = EV_data
        self.TOU_data = TOU_data
        self.journey_start, self.journey_end = self.find_journey_start_and_end_points()

    def find_journey_start_and_end_points(self, min_gap=30):

        P_total_data = self.EV_data.copy()
        min_gap = pd.Timedelta(min_gap*60, unit='sec')

        journey_start = [P_total_data.iloc[0, :].name]
        journey_end = [P_total_data.iloc[-1, :].name]

        for i in range(0, len(P_total_data) - 1):
            time_diff = P_total_data.iloc[i + 1, :].name - P_total_data.iloc[i, :].name
            if time_diff >= min_gap:
                journey_start.append(P_total_data.iloc[i + 1, :].name)
                journey_end.insert(0, P_total_data.iloc[i, :].name)

        return journey_start, journey_end

    def recommend(self, charger_power=6.6e3):

        pred = self.TOU_data.copy()
        pred['charging'] = 0
        pred['journey'] = 0

        for start, end in zip(self.journey_start, self.journey_end):

            start_time_slot = start.floor(freq='30min')
            end_time_slot = end.floor(freq='30min')

            pred.loc[np.logical_and(pred.index > start_time_slot, pred.index < end_time_slot), 'charging'] = 30

            if start_time_slot == end_time_slot:
                pred.loc[start_time_slot, ['charging', 'journey']] += (end - start).seconds / 60
            else:
                pred.loc[start_time_slot, ['charging', 'journey']] += (start.ceil(freq='30min') - start).seconds / 60
                pred.loc[end_time_slot, ['charging', 'journey']] += (end - end_time_slot).seconds / 60

            journey_energy_consumption = sum(self.EV_data.loc[start:end]['P_total'])
            charge_time = journey_energy_consumption / (charger_power * 60)

            #number of time slots needed to charge EV
            quotient = int(charge_time // 30) 
            remainder = charge_time % 30 

            # --------- and sorting TOU by price
            free_time_slots = pred.loc[np.logical_and(pred.index < start, pred['charging'] < 30)].copy()
            free_time_slots = free_time_slots.sort_values(by=['TOU'])

            if charge_time + sum(free_time_slots['charging']) > 30 * len(free_time_slots):
                print('Not enough time slots to charge')
                return None

            remainder += sum(free_time_slots.iloc[list(range(0, quotient))]['charging'])
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

            if quotient > 0: pred.loc[free_time_slots.iloc[list(range(0, quotient))].index, 'charging'] = 30

        pred['charging'] -= pred['journey']
        self.EV_data['charging'] = pred['charging']

        return pred.loc[pred['charging'] > 0, 'charging']
