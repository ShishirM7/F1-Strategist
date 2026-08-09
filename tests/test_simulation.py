import unittest
import pandas as pd
import numpy as np
from src.models import LapTimePredictor
from src.simulation import RaceStrategySimulator, StrategyRecommendation

class TestRaceStrategySimulator(unittest.TestCase):

    def setUp(self):
        # Build synthetic degradation data where tyres degrade heavily after 15 laps
        np.random.seed(42)
        n_samples = 150
        tyre_life = np.random.randint(1, 30, size=n_samples)
        # 0.25s degradation per lap
        lap_times = 80.0 + (tyre_life * 0.25) + np.random.normal(0, 0.05, size=n_samples)

        train_df = pd.DataFrame({
            'TyreLife': tyre_life,
            'Compound_SOFT': [1] * n_samples,
            'Compound_MEDIUM': [0] * n_samples,
            'Compound_HARD': [0] * n_samples,
            'Compound_INTERMEDIATE': [0] * n_samples,
            'Compound_WET': [0] * n_samples,
            'Position': [1] * n_samples,
            'LapTimeSeconds': lap_times,
            'IsCleanLap': [True] * n_samples
        })

        self.predictor = LapTimePredictor()
        self.predictor.train(train_df)
        self.simulator = RaceStrategySimulator(predictor=self.predictor, default_pit_loss=20.0)

    def test_evaluate_pit_decision_heavily_degraded(self):
        """When tyres are old (age 25), pitting should be recommended over staying out for 15 remaining laps."""
        result = self.simulator.evaluate_pit_decision(
            remaining_laps=15,
            current_tyre_life=25,
            current_compound='SOFT',
            new_compound='SOFT',
            pit_loss=20.0
        )

        self.assertIsInstance(result, StrategyRecommendation)
        self.assertEqual(result.decision, "PIT_NOW")
        self.assertGreater(result.time_delta_seconds, 0)

    def test_evaluate_pit_decision_fresh_tyres(self):
        """When tyres are almost fresh (age 2), driver should stay out."""
        result = self.simulator.evaluate_pit_decision(
            remaining_laps=10,
            current_tyre_life=2,
            current_compound='SOFT',
            new_compound='SOFT',
            pit_loss=20.0
        )

        self.assertEqual(result.decision, "STAY_OUT")
        self.assertLessEqual(result.time_delta_seconds, 1.0)

if __name__ == '__main__':
    unittest.main()