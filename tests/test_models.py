import unittest
import pandas as pd
import numpy as np
from src.models import LapTimePredictor

class TestLapTimePredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = LapTimePredictor()
        
        # Build synthetic dataset representing degradation (1.5s slower over 20 laps)
        np.random.seed(42)
        n_samples = 100
        tyre_life = np.random.randint(1, 25, size=n_samples)
        lap_times = 85.0 + (tyre_life * 0.08) + np.random.normal(0, 0.1, size=n_samples)

        self.mock_df = pd.DataFrame({
            'TyreLife': tyre_life,
            'Compound_SOFT': [0] * n_samples,
            'Compound_MEDIUM': [1] * n_samples,
            'Compound_HARD': [0] * n_samples,
            'Compound_INTERMEDIATE': [0] * n_samples,
            'Compound_WET': [0] * n_samples,
            'TrackTemp': [30.0] * n_samples,
            'AirTemp': [20.0] * n_samples,
            'Position': [2] * n_samples,
            'LapTimeSeconds': lap_times,
            'IsCleanLap': [True] * n_samples
        })

    def test_training_and_prediction(self):
        metrics = self.predictor.train(self.mock_df)
        self.assertIn('mae', metrics)
        self.assertTrue(self.predictor.is_trained)

        # Predict 5 laps on fresh Mediums
        forecast = self.predictor.predict_stint_trajectory(
            starting_tyre_life=1,
            compound='MEDIUM',
            horizon_laps=5
        )
        
        self.assertEqual(len(forecast), 5)
        # Lap time on older tyres should be greater than fresh tyres
        self.assertGreater(forecast[-1], forecast[0])

if __name__ == '__main__':
    unittest.main()