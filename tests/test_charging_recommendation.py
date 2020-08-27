import os
import sys
import unittest
import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

test_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(os.path.normpath(test_dir))
sys.path.insert(0, main_dir)

from charging_recommendation import charging_recommendation

class charging_recommendation_tests(unittest.TestCase):

    def setup_EV_data_one_journey(self):

        data=[['2012-03-01 12:10:25',725.6652975859237],
              ['2012-03-01 12:10:26',1257.8565322939758],
              ['2012-03-01 12:10:27',1614.6213222627302],
              ['2012-03-01 12:10:28',1646.2014645982658],
              ['2012-03-01 12:10:29',1422.8805052671455],
              ['2012-03-01 12:10:30',1097.7222297835308],
              ['2012-03-01 12:10:31',668.1959896387149],
              ['2012-03-01 12:10:32',130.7977777486384],
              ['2012-03-01 12:10:33',-271.7970997268379],
              ['2012-03-01 12:10:34',-25.602265461532852],
              ['2012-03-01 12:10:35',700.0]]

        col_names = ['timestamp', 'P_total']

        df = pd.DataFrame(data=data, columns=col_names)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

        df = df.set_index('timestamp')

        return df

    def setup_start_and_end_one_journey(self):

        df = self.setup_EV_data_one_journey()

        journey_start = [df.iloc[0,:].name]
        journey_end = [df.iloc[-1, :].name]

        return journey_start, journey_end

    def setup_expected_suggestion_one_journey(self):

        data = [['2012-03-01 04:30:00',0.02264278220704685]]

        col_names = ['timestamp', 'charging']

        df = pd.DataFrame(data=data, columns=col_names)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

        df = df.set_index('timestamp')

        return df['charging']

    def setup_EV_data_two_journeys(self):

        data=[['2012-03-01 12:10:25',725.6652975859237],
              ['2012-03-01 12:10:26',1257.8565322939758],
              ['2012-03-01 12:10:27',1614.6213222627302],
              ['2012-03-01 12:10:28',1646.2014645982658],
              ['2012-03-01 12:10:29',1422.8805052671455],
              ['2012-03-01 12:10:30',1097.7222297835308],
              ['2012-03-01 12:10:31',668.1959896387149],
              ['2012-03-01 12:10:32',130.7977777486384],
              ['2012-03-01 12:10:33',-271.7970997268379],
              ['2012-03-01 12:10:34',-25.602265461532852],
              ['2012-03-01 12:10:35',700.0],
              ['2012-03-01 15:06:28',700.0],
              ['2012-03-01 15:06:29',3380.0239798070857],
              ['2012-03-01 15:06:30',3223.8939157828513],
              ['2012-03-01 15:06:31',3039.811891741142],
              ['2012-03-01 15:06:32',1681.704175918081],
              ['2012-03-01 15:06:33',324.6514419168423],
              ['2012-03-01 15:06:34',-365.8002966864635],
              ['2012-03-01 15:06:35',-186.5440995596856],
              ['2012-03-01 15:06:36',428.6289494237993],
              ['2012-03-01 15:06:37',1296.9412807524639],
              ['2012-03-01 15:06:38',2648.400085705571],
              ['2012-03-01 15:06:39',3534.5359973467],
              ['2012-03-01 15:06:40',2901.1930599100783],
              ['2012-03-01 15:06:41',1383.3227659771262],
              ['2012-03-01 15:06:42',451.62946409097833],
              ['2012-03-01 15:06:43',321.0564126992313],
              ['2012-03-01 15:06:44',454.44225589937344],
              ['2012-03-01 15:06:45',387.08457288154614],
              ['2012-03-01 15:06:46',133.897362211565],
              ['2012-03-01 15:06:47',41.010885689294355],
              ['2012-03-01 15:06:48',222.65882483519766],
              ['2012-03-01 15:06:49',511.7796784183827]]

        col_names = ['timestamp', 'P_total']

        df = pd.DataFrame(data=data, columns=col_names)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

        df = df.set_index('timestamp')

        return df

    def setup_start_and_end_two_journeys(self):

        df = self.setup_EV_data_two_journeys()

        journey_start = [df.iloc[0,:].name, df.iloc[11,:].name]
        journey_end = [df.iloc[10, :].name, df.iloc[-1, :].name]

        return journey_start, journey_end

    def setup_expected_suggestion_two_journeys(self):

        data = [['2012-03-01 04:30:00',0.089598]]

        col_names = ['timestamp', 'charging']

        df = pd.DataFrame(data=data, columns=col_names)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

        df = df.set_index('timestamp')

        return df['charging']

    def setup_TOU_data(self):

        data = [['2012-03-01 00:00:00',10.143],
                ['2012-03-01 00:30:00',11.487],
                ['2012-03-01 01:00:00',11.928],
                ['2012-03-01 01:30:00',11.13],
                ['2012-03-01 02:00:00',11.13],
                ['2012-03-01 02:30:00',11.13],
                ['2012-03-01 03:00:00',10.71],
                ['2012-03-01 03:30:00',10.71],
                ['2012-03-01 04:00:00',10.5],
                ['2012-03-01 04:30:00',10.08],
                ['2012-03-01 05:00:00',11.55],
                ['2012-03-01 05:30:00',11.508],
                ['2012-03-01 06:00:00',14.028],
                ['2012-03-01 06:30:00',11.76],
                ['2012-03-01 07:00:00',13.692],
                ['2012-03-01 07:30:00',15.288],
                ['2012-03-01 08:00:00',15.225],
                ['2012-03-01 08:30:00',14.679],
                ['2012-03-01 09:00:00',14.91],
                ['2012-03-01 09:30:00',14.112],
                ['2012-03-01 10:00:00',13.818],
                ['2012-03-01 10:30:00',12.894],
                ['2012-03-01 11:00:00',13.146],
                ['2012-03-01 11:30:00',12.18],
                ['2012-03-01 12:00:00',12.621],
                ['2012-03-01 12:30:00',12.054],
                ['2012-03-01 13:00:00',13.44],
                ['2012-03-01 13:30:00',12.6],
                ['2012-03-01 14:00:00',12.6],
                ['2012-03-01 14:30:00',12.18],
                ['2012-03-01 15:00:00',11.655],
                ['2012-03-01 15:30:00',13.02],
                ['2012-03-01 16:00:00',26.208],
                ['2012-03-01 16:30:00',28.98],
                ['2012-03-01 17:00:00',28.14],
                ['2012-03-01 17:30:00',28.56],
                ['2012-03-01 18:00:00',28.035],
                ['2012-03-01 18:30:00',27.258],
                ['2012-03-01 19:00:00',13.545],
                ['2012-03-01 19:30:00',13.923],
                ['2012-03-01 20:00:00',12.789],
                ['2012-03-01 20:30:00',11.655],
                ['2012-03-01 21:00:00',11.025],
                ['2012-03-01 21:30:00',10.29],
                ['2012-03-01 22:00:00',9.975],
                ['2012-03-01 22:30:00',9.744],
                ['2012-03-01 23:00:00',9.66],
                ['2012-03-01 23:30:00',9.555]]

        col_names = ['timestamp','TOU']

        df = pd.DataFrame(data=data, columns=col_names)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

        df = df.set_index('timestamp')

        return df

    def test_find_journey_start_and_end_points_one_journey(self):

        EV_data = self.setup_EV_data_one_journey()
        TOU_data = self.setup_TOU_data()

        charging_recom_obj = charging_recommendation(EV_data, TOU_data)

        journey_start = charging_recom_obj.journey_start
        journey_end = charging_recom_obj.journey_end

        expected_journey_start, expected_journey_end = self.setup_start_and_end_one_journey()

        self.assertListEqual(journey_start, expected_journey_start)
        self.assertListEqual(journey_end, expected_journey_end)

    def test_find_journey_start_and_end_points_two_journeys(self):

        EV_data = self.setup_EV_data_two_journeys()
        TOU_data = self.setup_TOU_data()

        charging_recom_obj = charging_recommendation(EV_data, TOU_data)

        journey_start = charging_recom_obj.journey_start
        journey_end = charging_recom_obj.journey_end

        expected_journey_start, expected_journey_end = self.setup_start_and_end_two_journeys()

        self.assertListEqual(journey_start, expected_journey_start)
        self.assertListEqual(journey_end, expected_journey_end)

    def test_recommend_one_journey(self):

        EV_data = self.setup_EV_data_one_journey()
        TOU_data = self.setup_TOU_data()

        charging_recom_obj = charging_recommendation(EV_data, TOU_data)
        suggestion = charging_recom_obj.recommend()

        expected_suggestion = self.setup_expected_suggestion_one_journey()

        assert_series_equal(suggestion, expected_suggestion)

    def test_recommend_two_journeys(self):

        EV_data = self.setup_EV_data_two_journeys()
        TOU_data = self.setup_TOU_data()

        charging_recom_obj = charging_recommendation(EV_data, TOU_data)
        suggestion = charging_recom_obj.recommend()

        expected_suggestion = self.setup_expected_suggestion_two_journeys()

        assert_series_equal(suggestion, expected_suggestion)

    def test_recommend_returns_None(self):

        EV_data = self.setup_EV_data_two_journeys()
        TOU_data = self.setup_TOU_data()

        charging_recom_obj = charging_recommendation(EV_data, TOU_data)
        suggestion = charging_recom_obj.recommend(charger_power=1e-3)

        expected_suggestion = None

        self.assertEqual(suggestion, expected_suggestion)


if __name__ == "__main__":
    unittest.main()
