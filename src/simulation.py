import pandas as pd
from src.EV_data_analysis import EV
from src.TOU_analysis_and_prediction import TOU
from src.charging_recommendation import charging_recommendation
from pandas.tseries.offsets import DateOffset

class Simulation:
    def __init__(self, drive_cycle_file, drive_cycle_subdir, tou_file):
        tou_model_trained = False
        if not tou_model_trained:
            self.train_tou_model()

        self.ev_cycle_file = drive_cycle_file
        self.ev_subdirectory = drive_cycle_subdir

    def run_recommendation_algorithm(self):
        ev = self.create_ev()
        tou = create_tou()
        recommendation_results = charging_recommendation()

    def create_ev(self):
        ev_obj = EV(self.ev_cycle_file, self.ev_subdirectory)
        cols_to_drop = ['cycle_sec', 'timestep', 'speed_mph', 'accel_meters_ps', 'speed_mps',
                        'accel_mps2', 'P_wheels', 'P_electric_motor', 'n_rb', 'P_regen']

        P_total = ev_obj.data.copy()
        P_total = P_total.drop(columns=cols_to_drop)
        P_total = P_total.set_index('timestamp')

    def create_tou(self):
        pass

    def train_tou_model(self):
        pass



    def run_recommendation_algorithm(self):
        pass

    def graph_plotter(self):
        pass

