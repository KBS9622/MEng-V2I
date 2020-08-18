import os
import sys
import unittest
import pandas as pd
from pandas._testing import assert_frame_equal

test_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(os.path.normpath(test_dir))
sys.path.insert(0, main_dir)

from TOU_analysis_and_prediction import TOU

class TOU_tests(unittest.TestCase):

    def setup_load_xlsx_data_expected_df(self):

        data = [['2019-01-31', '00:00:00', '00:30:00', 'E-1R-AGILE-18-02-21-B', '_B', 'East Midlands', 9.66, 10.143],
                ['2019-01-31', '00:30:00', '01:00:00', 'E-1R-AGILE-18-02-21-B', '_B', 'East Midlands', 10.94, 11.487],
                ['2019-01-31', '01:00:00', '01:30:00', 'E-1R-AGILE-18-02-21-B', '_B', 'East Midlands', 11.36, 11.928],
                ['2019-01-31', '01:30:00', '02:00:00', 'E-1R-AGILE-18-02-21-B', '_B', 'East Midlands', 10.6, 11.13],
                ['2019-01-31', '02:00:00', '02:30:00', 'E-1R-AGILE-18-02-21-B', '_B', 'East Midlands', 10.6, 11.13],
                ['2019-01-31', '02:30:00', '03:00:00', 'E-1R-AGILE-18-02-21-B', '_B', 'East Midlands', 10.6, 11.13],
                ['2019-01-31', '03:00:00', '03:30:00', 'E-1R-AGILE-18-02-21-B', '_B', 'East Midlands', 10.2, 10.71],
                ['2019-01-31', '03:30:00', '04:00:00', 'E-1R-AGILE-18-02-21-B', '_B', 'East Midlands', 10.2, 10.71],
                ['2019-01-31', '04:00:00', '04:30:00', 'E-1R-AGILE-18-02-21-B', '_B', 'East Midlands', 10, 10.5]]

        col_names = ['date', 'from', 'to', 'code', 'gsp', 'region_name', 'unit_rate_excl_vat', 'unit_rate_incl_vat']

        df = pd.DataFrame(data=data, columns=col_names)

        df['date'] = pd.to_datetime(df['date'])

        return df

    def setup_format_TOU_data_expected_df(self):

        df = self.setup_load_xlsx_data_expected_df()

        cols_to_drop = ['code', 'gsp', 'region_name']
        df = df.drop(columns=cols_to_drop)

        df['from'] = pd.to_timedelta(df['from'])
        df['to'] = pd.to_timedelta(df['to'])

        return df

    def setup_create_time_idx_TOU_price_expected_df(self):

        data = [['2019-01-31 00:00:00', 10.143],
                ['2019-01-31 00:30:00', 11.487],
                ['2019-01-31 01:00:00', 11.928],
                ['2019-01-31 01:30:00', 11.13],
                ['2019-01-31 02:00:00', 11.13],
                ['2019-01-31 02:30:00', 11.13],
                ['2019-01-31 03:00:00', 10.71],
                ['2019-01-31 03:30:00', 10.71],
                ['2019-01-31 04:00:00', 10.5]]

        col_names = ['timestamp', 'unit_rate_incl_vat']

        df = pd.DataFrame(data=data, columns=col_names)

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        df = df.set_index('timestamp')

        return df

    def setup_expected_TOU_obj(self):
        file_name = 'test_agile_rates_2019.xlsx'
        TOU_obj = TOU(file_name)

        TOU_obj.file_name = 'test_agile_rates_2019.xlsx'
        TOU_obj.data = self.setup_format_TOU_data_expected_df()
        TOU_obj.time_idx_TOU_price = self.setup_create_time_idx_TOU_price_expected_df()

        return TOU_obj

    def test_TOU_returns_correct_obj(self):

        file_name = 'test_agile_rates_2019.xlsx'
        TOU_obj = TOU(file_name)

        expected_TOU_obj = self.setup_expected_TOU_obj()

        self.assertEqual(TOU_obj.file_name, expected_TOU_obj.file_name)
        assert_frame_equal(TOU_obj.data, expected_TOU_obj.data)
        assert_frame_equal(TOU_obj.time_idx_TOU_price, expected_TOU_obj.time_idx_TOU_price)

    def test_load_xlsx_data(self):

        file_name = 'test_agile_rates_2019.xlsx'
        TOU_obj = TOU(file_name)

        df = TOU_obj.load_xlsx_data()

        expected_df = self.setup_load_xlsx_data_expected_df()

        assert_frame_equal(df, expected_df)

    def test_load_xlsx_data_with_subdir(self):

        file_name = 'test_agile_rates_2019.xlsx'
        TOU_obj = TOU(file_name)

        subdir = 'test_files'
        df = TOU_obj.load_xlsx_data(subdir)

        expected_df = self.setup_load_xlsx_data_expected_df()

        assert_frame_equal(df, expected_df)

    def test_format_TOU_data(self):

        file_name = 'test_agile_rates_2019.xlsx'
        TOU_obj = TOU(file_name)

        df = TOU_obj.format_TOU_data()

        expected_df = self.setup_format_TOU_data_expected_df()

        assert_frame_equal(df, expected_df)

    def test_create_time_idx_TOU_price(self):

        file_name = 'test_agile_rates_2019.xlsx'
        TOU_obj = TOU(file_name)

        df = TOU_obj.create_time_idx_TOU_price()

        expected_df = self.setup_create_time_idx_TOU_price_expected_df()

        assert_frame_equal(df, expected_df)



if __name__ == "__main__":
    unittest.main()