import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class StrategyVisualizer:
    """
    Generates plots for F1 race strategy decisions and stint trajectory analysis.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Configures Matplotlib style settings for visual output.
        """
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')

    def plot_strategy_comparison(
        self,
        stay_out_laps: np.ndarray,
        pit_now_laps: np.ndarray,
        pit_loss: float,
        current_lap: int,
        driver_code: str = "DRIVER",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plots comparative lap times and cumulative delta for 'Pit Now' vs 'Stay Out'.

        Args:
            stay_out_laps (np.ndarray): Projected lap times if staying out.
            pit_now_laps (np.ndarray): Projected lap times post-pit stop.
            pit_loss (float): Applied pit lane delta.
            current_lap (int): Active decision lap.
            driver_code (str): Driver identifier string (e.g. 'VER').
            save_path (Optional[str]): Path to save output plot PNG file.
        """
        horizon = len(stay_out_laps)
        laps_range = np.arange(current_lap + 1, current_lap + horizon + 1)

        # Calculate cumulative time deltas
        cum_stay_out = np.cumsum(stay_out_laps)
        
        # Pit now scenario: first lap adds pit loss delta
        pit_laps_adjusted = np.copy(pit_now_laps)
        pit_laps_adjusted[0] += pit_loss
        cum_pit_now = np.cumsum(pit_laps_adjusted)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        # Plot 1: Lap Times Trajectory
        ax1.plot(laps_range, stay_out_laps, label="Stay Out (Current Tyres)", color='red', linestyle='--', marker='o')
        ax1.plot(laps_range, pit_now_laps, label="Pit Now (Fresh Tyres)", color='green', linestyle='-', marker='s')
        ax1.set_ylabel("Predicted Lap Time (s)")
        ax1.set_title(f"AI Strategy Projection | Driver: {driver_code} | Lap {current_lap}")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Cumulative Race Time Delta (Stay Out minus Pit Now)
        # Positive values mean Pit Now is faster overall
        net_gain = cum_stay_out - cum_pit_now
        
        colors = ['green' if g > 0 else 'red' for g in net_gain]
        ax2.bar(laps_range, net_gain, color=colors, alpha=0.6, edgecolor='black')
        ax2.axhline(0, color='black', linewidth=1, linestyle='--')
        ax2.set_xlabel("Race Lap Number")
        ax2.set_ylabel("Net Time Advantage (s)\n[Positive = Pit Now Gains Time]")
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Strategy plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close()