import pandas as pd
from src.EV_data_analysis import EV
from src.TOU_analysis_and_prediction import TOU
from src.charging_recommendation import charging_recommendation
from pandas.tseries.offsets import DateOffset


class Simulation:
    def __init__(self, drive_cycle_file, drive_cycle_subdir, tou_file, train_tou):
        self.min_tou_threshold = 0
        self.tou_obj = TOU(tou_file)
        if train_tou:
            self.tou_obj.create_and_fit_model()
        self.ev_obj = EV(drive_cycle_file, drive_cycle_subdir)
        self.recommendation_obj = self.create_recommendation_obj()
        self.graph_plotter(drive_cycle_file, drive_cycle_subdir)

    def plugged_in(self):
        """
        
        :return: 
        """
        # self.start_time = time.
        # self.run_recommendation_algorithm()
        pass

    def create_recommendation_obj(self):
        predicted_tou_data = self.get_tou_data(start_time=pd.to_datetime('2019-01-31 00:00:00') ,
                                               end_time=pd.to_datetime('2019-01-31 00:00:00'))
        ev_consumption_data = self.format_ev_data()
        recommendation_obj = charging_recommendation(ev_consumption_data, predicted_tou_data, ev_consumption_data)
        return recommendation_obj


    def run_recommendation_algorithm(self):
        """
        
        :return: 
        """
        self.recommendation_obj.set_TOU_data(self.get_tou_data())
        self.recommendation_obj.set_EV_data(self.format_ev_data())
        json_path = './utils/user_config.json'
        self.recommendation_obj.update_user_config(json_path)
        return self.recommendation_obj.recommend()

    def format_ev_data(self):
        """
        
        :return: 
        """
        cols_to_drop = ['cycle_sec', 'timestep', 'speed_mph', 'accel_meters_ps', 'speed_mps',
                        'accel_mps2', 'P_wheels', 'P_electric_motor', 'n_rb', 'P_regen']

        p_total = self.ev_obj.data.copy()
        p_total = p_total.drop(columns=cols_to_drop)
        p_total = p_total.set_index('timestamp')
        p_total = p_total.set_index(p_total.index
                                    + DateOffset(days=(self.start_time.floor(freq='D')
                                                       - p_total.iloc[0].name.floor(freq='D')).days))
        return p_total

    def get_tou_data(self, start_time=pd.to_datetime('2019-01-31 00:00:00'), end_time=pd.to_datetime('2019-01-31 23:30:00') ):
        """
        
        :param start_time: 
        :param end_time: 
        :return: 
        """
        self.start_time = start_time
        self.end_time = end_time
        predicted_tou = self.tou_obj.predict_and_compare(self.start_time, self.end_time)
        return predicted_tou

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
        without_recommendation = ''
        without_recommendation = ''
        pass

