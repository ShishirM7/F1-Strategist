import os
import shutil
import unittest
from unittest.mock import patch
from src.ingestion import RaceDataLoader

class TestRaceDataLoader(unittest.TestCase):
    
    def setUp(self):
        """Set up a temporary cache directory path before each test."""
        self.test_cache_dir = os.path.join("tests", "temp_cache")
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir, ignore_errors=True)

    def tearDown(self):
        """Clean up temporary directory after each test."""
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir, ignore_errors=True)

    @patch("fastf1.Cache.enable_cache")
    def test_cache_initialization(self, mock_enable_cache):
        """Test that data loader creates cache directory and enables FastF1 cache."""
        loader = RaceDataLoader(cache_dir=self.test_cache_dir)
        
        # Verify directory was created
        self.assertTrue(os.path.exists(self.test_cache_dir))
        
        # Verify FastF1 enable_cache was called with correct path
        mock_enable_cache.assert_called_once_with(self.test_cache_dir)

if __name__ == '__main__':
    unittest.main()
    