import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Transforms raw FastF1 lap data into clean, ML-ready feature sets for F1 race strategy modeling.
    """

    STANDARD_COMPOUNDS: List[str] = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']

    def __init__(self, pit_loss_seconds: float = 22.0):
        """
        Args:
            pit_loss_seconds (float): Average estimated time lost during a pit stop (pit lane drive + stationary time).
        """
        self.pit_loss_seconds = pit_loss_seconds

    def transform_lap_data(self, laps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes and enriches a session's lap DataFrame with strategy features.

        Args:
            laps_df (pd.DataFrame): Raw lap dataframe from RaceDataLoader.

        Returns:
            pd.DataFrame: Cleaned dataframe enriched with engineered features.
        """
        if laps_df.empty:
            raise ValueError("Provided laps_df is empty.")

        df = laps_df.copy()

        # Convert timedelta columns to float seconds for ML modeling
        df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
        df['Sector1TimeSeconds'] = df['Sector1Time'].dt.total_seconds()
        df['Sector2TimeSeconds'] = df['Sector2Time'].dt.total_seconds()
        df['Sector3TimeSeconds'] = df['Sector3Time'].dt.total_seconds()

        # Ensure correct sorting by lap number and position
        df = df.sort_values(by=['LapNumber', 'Position']).reset_index(drop=True)

        # Flag non-representative green-flag pace laps
        df = self._flag_clean_laps(df)

        # Calculate time gaps to drivers ahead and behind on each lap
        df = self._compute_driver_gaps(df)

        # One-hot encode tyre compounds
        df = self._encode_compounds(df)

        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        return df

    def _flag_clean_laps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies laps affected by Safety Cars, In/Out laps, or track anomalies.
        """
        # OutLaps and PitInLaps contain non-racing time lost in the pit lane
        is_pit_lap = df['PitOutTime'].notnull() | df['PitInTime'].notnull()
        
        # Track status flags: '1' indicates normal track conditions (green flag)
        # FastF1 TrackStatus codes: 1=AllClear, 2=Yellow, 4=SC, 5=Red, 6=VSC
        is_green_flag = df['TrackStatus'].astype(str) == '1'

        # Flag true racing laps suitable for degradation modeling
        df['IsCleanLap'] = (~is_pit_lap) & is_green_flag
        return df

    def _compute_driver_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes time gaps to the car directly ahead and behind for each driver per lap.
        """
        df['GapToAhead'] = np.nan
        df['GapToBehind'] = np.nan
        
        # Calculate cumulative race elapsed time per driver
        df = df.sort_values(by=['Driver', 'LapNumber']).reset_index(drop=True)
        df['CumulativeRaceTime'] = df.groupby('Driver')['LapTimeSeconds'].cumsum()

        # Re-sort back to lap and position order for gap evaluation
        df = df.sort_values(by=['LapNumber', 'Position']).reset_index(drop=True)

        # Process frame lap by lap to determine intervals
        for lap_num, lap_group in df.groupby('LapNumber'):
            sorted_lap = lap_group.sort_values(by='Position')
            cum_times = sorted_lap['CumulativeRaceTime'].values
            
            # Cumulative lap time delta proxy for gap calculation across field
            lap_times = sorted_lap['LapTimeSeconds'].values
            positions = sorted_lap['Position'].values

            gaps_ahead = [np.nan]
            for i in range(1, len(sorted_lap)):
                # Approximate gap from lap time differences if telemetry gap isn't direct
                #gap = lap_times[i] - lap_times[i-1] if not np.isnan(lap_times[i]) and not np.isnan(lap_times[i-1]) else np.nan
                gap = cum_times[i] - cum_times[i-1] if not np.isnan(cum_times[i]) and not np.isnan(cum_times[i-1]) else np.nan
                gaps_ahead.append(gap)

            df.loc[sorted_lap.index, 'GapToAhead'] = gaps_ahead
            
            # Shift gaps ahead down to calculate gap to car behind
            #df.loc[sorted_lap.index, 'GapToBehind'] = pd.Series(gaps_ahead).shift(-1).values
            df.loc[sorted_lap.index, 'GapToBehind'] = pd.Series(gaps_ahead, index=sorted_lap.index).shift(-1).values
        return df

    def _encode_compounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes tyre compound string names into standardized binary columns.
        """
        df['Compound'] = df['Compound'].str.upper()
        for compound in self.STANDARD_COMPOUNDS:
            df[f'Compound_{compound}'] = (df['Compound'] == compound).astype(int)
        return df