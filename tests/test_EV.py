import os
import sys
import unittest
import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

test_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(os.path.normpath(test_dir))
sys.path.insert(0, main_dir)

from EV_data_analysis import EV

class EV_tests(unittest.TestCase):

    def setup_load_csv_data_expected_df(self):

        data = [['2012-05-22 07:19:51',0,1,0.0,0.0],
                ['2012-05-22 07:19:52',1,1,0.0,0.0],
                ['2012-05-22 07:19:53',2,1,0.0,0.0],
                ['2012-05-22 07:19:54',3,1,0.0,0.0],
                ['2012-05-22 07:19:55',4,1,0.0,0.0],
                ['2012-05-22 07:19:56',5,1,0.0,0.0],
                ['2012-05-22 07:19:57',6,1,1.38697917336,1.38697917336],
                ['2012-05-22 07:19:58',7,1,2.05818452976,0.671205356399],
                ['2012-05-22 07:19:59',8,1,2.39650294717,0.338318417411]]

        col_names = ['timestamp','cycle_sec','timestep','speed_mph','accel_meters_ps']

        df = pd.DataFrame(data=data, columns=col_names)

        return df

    def setup_format_EV_data_expected_df(self):

        df = self.setup_load_csv_data_expected_df()

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

        return df

    def setup_expected_Nissan_Leaf_characteristics(self):

        data = ['Nissan Leaf 2013',1521,1.75,0.0328,4.575,1.2256,2.3316,0.28,0.92,0.91,24000]

        col_names = ['vehicle_model','m_kg','C_r','c_1','c_2','rho_air',
                     'A_f','C_D','n_driveline','n_electric_motor','capacity']

        series = pd.Series(data=data, index=col_names, name=0)

        return series

    def setup_calculate_energy_consumption_and_regen_braking_expected_df(self):

        data = [['2012-05-22 07:19:51',0,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,700.0],
                ['2012-05-22 07:19:52',1,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,700.0],
                ['2012-05-22 07:19:53',2,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,700.0],
                ['2012-05-22 07:19:54',3,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,700.0],
                ['2012-05-22 07:19:55',4,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,700.0],
                ['2012-05-22 07:19:56',5,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,700.0],
                ['2012-05-22 07:19:57',6,1,1.38697917336,1.38697917336,0.6200351696588543,
                 0.6200351696588543,659.2078032868252,787.3958472131213,0.0,787.3958472131213,1487.3958472131212],
                ['2012-05-22 07:19:58',7,1,2.05818452976,0.671205356399,0.9200908121839102,
                 0.300055642524609,530.828912833186,634.0526909139822,0.0,634.0526909139822,1334.0526909139821],
                ['2012-05-22 07:19:59',8,1,2.39650294717,0.338318417411,1.0713326775028769,
                 0.15124186531941344,375.86124448935703,448.950363699662,0.0,448.950363699662,1148.950363699662]]

        col_names = ['timestamp','cycle_sec','timestep','speed_mph','accel_meters_ps',
                     'speed_mps','accel_mps2','P_wheels','P_electric_motor','n_rb','P_regen','P_total']

        df = pd.DataFrame(data=data, columns=col_names)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

        return df

    def setup_expected_EV_obj(self):

        file_name = 'test_2012-05-22.csv'
        c_file_name = 'test_EV_characteristics.csv'
        choice = 0
        EV_obj = EV(file_name, c_file_name=c_file_name, choice=choice)

        EV_obj.file_name = file_name
        EV_obj.subdir = ''
        EV_obj.c_file_name = c_file_name
        EV_obj.choice = choice
        EV_obj.data = self.setup_calculate_energy_consumption_and_regen_braking_expected_df()

        EV_obj.EV = self.setup_expected_Nissan_Leaf_characteristics()

        EV_obj.EV_model = EV_obj.EV['vehicle_model']
        EV_obj.m = EV_obj.EV['m_kg']
        EV_obj.C_r = EV_obj.EV['C_r']
        EV_obj.c_1 = EV_obj.EV['c_1']
        EV_obj.c_2 = EV_obj.EV['c_2']
        EV_obj.rho_air = EV_obj.EV['rho_air']
        EV_obj.A_f = EV_obj.EV['A_f']
        EV_obj.C_D = EV_obj.EV['C_D']
        EV_obj.n_driveline = EV_obj.EV['n_driveline']
        EV_obj.n_electric_motor = EV_obj.EV['n_electric_motor']
        EV_obj.capacity = EV_obj.EV['capacity']
        EV_obj.charge_lvl = EV_obj.EV['capacity'] * (50/100)

        return EV_obj

    def test_EV_returns_correct_obj(self):

        file_name = 'test_2012-05-22.csv'
        c_file_name = 'test_EV_characteristics.csv'
        choice = 0
        EV_obj = EV(file_name, c_file_name=c_file_name, choice=choice)

        expected_EV_obj = self.setup_expected_EV_obj()

        self.assertEqual(EV_obj.file_name, expected_EV_obj.file_name)
        self.assertEqual(EV_obj.subdir, expected_EV_obj.subdir)
        self.assertEqual(EV_obj.c_file_name, expected_EV_obj.c_file_name)
        self.assertEqual(EV_obj.choice, expected_EV_obj.choice)
        assert_frame_equal(EV_obj.data, expected_EV_obj.data)

        assert_series_equal(EV_obj.EV, expected_EV_obj.EV)

        self.assertEqual(EV_obj.EV_model, expected_EV_obj.EV_model)
        self.assertEqual(EV_obj.m, expected_EV_obj.m)
        self.assertEqual(EV_obj.C_r, expected_EV_obj.C_r)
        self.assertEqual(EV_obj.c_1, expected_EV_obj.c_1)
        self.assertEqual(EV_obj.c_2, expected_EV_obj.c_2)
        self.assertEqual(EV_obj.rho_air, expected_EV_obj.rho_air)
        self.assertEqual(EV_obj.A_f, expected_EV_obj.A_f)
        self.assertEqual(EV_obj.C_D, expected_EV_obj.C_D)
        self.assertEqual(EV_obj.n_driveline, expected_EV_obj.n_driveline)
        self.assertEqual(EV_obj.n_electric_motor, expected_EV_obj.n_electric_motor)
        self.assertEqual(EV_obj.capacity, expected_EV_obj.capacity)
        self.assertEqual(EV_obj.charge_lvl, expected_EV_obj.charge_lvl)

    def test_load_csv_data(self):

        file_name = 'test_2012-05-22.csv'
        c_file_name = 'test_EV_characteristics.csv'
        EV_obj = EV(file_name, c_file_name=c_file_name, choice=0)

        df = EV_obj.load_csv_data(file_name, subdir='test_files')

        expected_df = self.setup_load_csv_data_expected_df()

        assert_frame_equal(df, expected_df)

    def test_format_EV_data(self):

        file_name = 'test_2012-05-22.csv'
        c_file_name = 'test_EV_characteristics.csv'
        EV_obj = EV(file_name, c_file_name=c_file_name, choice=0)

        df = EV_obj.format_EV_data()

        expected_df = self.setup_format_EV_data_expected_df()

        assert_frame_equal(df, expected_df)

    def test_EV_menu_with_choice_input_in_script(self):

        file_name = 'test_2012-05-22.csv'
        c_file_name = 'test_EV_characteristics.csv'
        EV_obj = EV(file_name, c_file_name=c_file_name, choice=0)

        series = EV_obj.EV_menu()

        expected_series = self.setup_expected_Nissan_Leaf_characteristics()

        assert_series_equal(series, expected_series)

    def test_calculate_energy_consumption_and_regen_braking(self):

        file_name = 'test_2012-05-22.csv'
        c_file_name = 'test_EV_characteristics.csv'
        EV_obj = EV(file_name, c_file_name=c_file_name, choice=0)

        df = EV_obj.calculate_energy_consumption()

        expected_df = self.setup_calculate_energy_consumption_and_regen_braking_expected_df()

        assert_frame_equal(df, expected_df)


if __name__ == "__main__":
    unittest.main()
