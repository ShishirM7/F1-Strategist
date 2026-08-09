import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Any, List

# Fuel effect per lap (car gets lighter by ~1.5kg/lap -> ~0.035s faster per lap)
FUEL_EFFECT_PER_LAP = -0.035

# Compound wear rates calibrated to ensure positive degradation slopes
# Linear wear MUST exceed |FUEL_EFFECT_PER_LAP| (0.035s) for lap times to rise with age
COMPOUND_DEG_RATES = {
    'SOFT':         {'linear': 0.120, 'exponential': 0.0040, 'pace_offset': -0.5},
    'MEDIUM':       {'linear': 0.075, 'exponential': 0.0025, 'pace_offset':  0.0},
    'HARD':         {'linear': 0.055, 'exponential': 0.0015, 'pace_offset':  0.4},
    'INTERMEDIATE': {'linear': 0.090, 'exponential': 0.0030, 'pace_offset':  2.0},
    'WET':          {'linear': 0.130, 'exponential': 0.0050, 'pace_offset':  5.0}
}

class LapTimePredictor:
    """
    ML Lap Time Predictor enhanced with physics-informed stint trajectory forecasting.
    """

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.feature_cols: List[str] = []
        self.is_trained: bool = False

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Trains the GBDT model on engineered lap features."""
        clean_df = df.dropna(subset=['LapTimeSeconds']).copy()
        clean_df = clean_df[np.isfinite(clean_df['LapTimeSeconds'])]

        if clean_df.empty:
            raise ValueError("No valid lap records remaining after dropping NaNs in LapTimeSeconds.")

#        ignore_cols = ['LapTimeSeconds', 'Driver', 'Team', 'Compound', 'TrackStatus']
        ignore_cols = ['LapTimeSeconds', 'Sector1TimeSeconds', 'Sector2TimeSeconds', 'Sector3TimeSeconds',
                       'Driver', 'Team', 'Compound', 'TrackStatus', 'LapNumber', 'Stint',
                       'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
        self.feature_cols = [
            col for col in clean_df.columns 
            if col not in ignore_cols and pd.api.types.is_numeric_dtype(clean_df[col])
        ]
        
        X = clean_df[self.feature_cols].fillna(0)
        y = clean_df['LapTimeSeconds']

        self.model.fit(X, y)
        self.is_trained = True

        preds = self.model.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        return {'mae': mae, 'rmse': rmse}

    def predict_stint_trajectory(
        self,
        starting_tyre_life: int,
        compound: str,
        horizon_laps: int,
        position: int = 3
    ) -> np.ndarray:
        """
        Predicts lap times over a stint horizon using baseline ML inference
        combined with compound-specific physical tyre degradation curves.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        compound_upper = compound.upper()
        deg_config = COMPOUND_DEG_RATES.get(
            compound_upper, 
            {'linear': 0.065, 'exponential': 0.002, 'pace_offset': 0.0}
        )

        # Baseline fresh pace (TyreLife = 1) from ML model
        dummy_features = pd.DataFrame(0.0, index=[0], columns=self.feature_cols)
        if 'TyreLife' in dummy_features.columns:
            dummy_features['TyreLife'] = 1
        if 'Position' in dummy_features.columns:
            dummy_features['Position'] = position
        if 'IsCleanLap' in dummy_features.columns:
            dummy_features['IsCleanLap'] = 1.0  # Force clean racing lap state
            
        # Activate the active tyre compound flag (One-Hot binary column)
        target_compound_col = f'Compound_{compound_upper}'
        if target_compound_col in dummy_features.columns:
            dummy_features[target_compound_col] = 1.0

        base_fresh_pace = float(self.model.predict(dummy_features)[0])

        trajectory = []
        for i in range(horizon_laps):
            current_age = starting_tyre_life + i
            age_wear = max(0, current_age - 1)
            
            # Tyre wear penalty (accumulated from fresh age 1)
            deg_penalty = (
                deg_config['linear'] * age_wear +
                deg_config['exponential'] * (age_wear ** 1.5)
            )
            
            # Fuel burn effect over the stint projection
            fuel_delta = FUEL_EFFECT_PER_LAP * i
            
            # Projected Lap Time = Base Pace + Compound Offset + Wear Penalty + Fuel Benefit
            projected_lap_time = (
                base_fresh_pace + 
                deg_config['pace_offset'] + 
                deg_penalty + 
                fuel_delta
            )
            trajectory.append(projected_lap_time)

        return np.array(trajectory)