import os
import logging
import pandas as pd
import fastf1
from typing import Optional

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RaceDataLoader:
    """
    Handles the ingestion and caching of Formula 1 data using the FastF1 library.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initializes the RaceDataLoader and configures the FastF1 cache.

        Args:
            cache_dir (str): The local directory path for storing FastF1 cache files.
        """
        self.cache_dir = cache_dir
        self._setup_cache()

    def _setup_cache(self) -> None:
        """Creates the cache directory if it doesn't exist and enables FastF1 caching."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory at {self.cache_dir}")
        
        fastf1.Cache.enable_cache(self.cache_dir)
        logger.info(f"FastF1 cache enabled at {self.cache_dir}")

    def load_session_laps(self, year: int, race: str, session_type: str = 'R') -> Optional[pd.DataFrame]:
        """
        Loads a specific F1 session and extracts the lap-level data.

        Args:
            year (int): The championship year (e.g., 2023).
            race (str): The race name or round number (e.g., 'Monza' or 14).
            session_type (str): The session identifier ('R' for Race, 'Q' for Qualifying, etc.).

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing lap data, or None if loading fails.
        """
        try:
            logger.info(f"Loading session data for {year} {race} (Session: {session_type})...")
            
            # Load the session; telemetry=False speeds up loading if we only need lap times
            session = fastf1.get_session(year, race, session_type)
            session.load(telemetry=False, weather=True)
            
            laps_df = session.laps
            
            if laps_df.empty:
                logger.warning(f"No lap data found for {year} {race}.")
                return None
            
            logger.info(f"Successfully loaded {len(laps_df)} laps.")
            return pd.DataFrame(laps_df)

        except Exception as e:
            logger.error(f"Failed to load session data for {year} {race}: {str(e)}")
            return None