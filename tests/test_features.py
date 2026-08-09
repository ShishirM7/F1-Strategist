import unittest
import pandas as pd
import numpy as np
from src.features import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        self.engineer = FeatureEngineer(pit_loss_seconds=20.0)
        # Construct synthetic lap data frame matching FastF1 output format
        self.mock_laps = pd.DataFrame({
            'LapNumber': [1, 1, 2, 2],
            'Driver': ['VER', 'HAM', 'VER', 'HAM'],
            'Position': [1, 2, 1, 2],
            'LapTime': [pd.Timedelta(seconds=90), pd.Timedelta(seconds=91), pd.Timedelta(seconds=89), pd.Timedelta(seconds=89.5)],
            'Sector1Time': [pd.Timedelta(seconds=30)] * 4,
            'Sector2Time': [pd.Timedelta(seconds=30)] * 4,
            'Sector3Time': [pd.Timedelta(seconds=30)] * 4,
            'PitOutTime': [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
            'PitInTime': [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
            'TrackStatus': ['1', '1', '1', '1'],
            'Compound': ['MEDIUM', 'MEDIUM', 'MEDIUM', 'MEDIUM'],
            'TyreLife': [1, 1, 2, 2]
        })

    def test_transform_lap_data_output_columns(self):
        df_transformed = self.engineer.transform_lap_data(self.mock_laps)
        
        self.assertIn('LapTimeSeconds', df_transformed.columns)
        self.assertIn('IsCleanLap', df_transformed.columns)
        self.assertIn('Compound_MEDIUM', df_transformed.columns)
        self.assertEqual(df_transformed['LapTimeSeconds'].iloc[0], 90.0)
        self.assertTrue(df_transformed['IsCleanLap'].all())

if __name__ == '__main__':
    unittest.main()