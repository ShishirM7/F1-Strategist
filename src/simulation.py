import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from src.models import LapTimePredictor

logger = logging.getLogger(__name__)

# Maximum realistic stint length per compound
COMPOUND_MAX_LIFE = {
    'SOFT': 18,
    'MEDIUM': 26,
    'HARD': 38,
    'INTERMEDIATE': 30,
    'WET': 22
}

# Preferred compound transition default map
DEFAULT_NEXT_COMPOUND = {
    'SOFT': 'MEDIUM',
    'MEDIUM': 'HARD',
    'HARD': 'MEDIUM',
    'INTERMEDIATE': 'INTERMEDIATE',
    'WET': 'INTERMEDIATE'
}


@dataclass
class StrategyRecommendation:
    """Dataclass holding strategy decision results and explainability metrics."""
    decision: str                      # 'PIT_NOW' or 'STAY_OUT'
    time_delta_seconds: float          # Time gained (>0) or lost (<0) by pitting now
    stay_out_total_time: float         # Projected total seconds if staying out to planned window
    pit_now_total_time: float          # Projected total seconds if pitting immediately
    pit_loss_seconds: float            # Applied pit stop loss penalty
    remaining_laps: int                # Laps evaluated in horizon
    explanation: str                   # Plain text human-readable justification

class RaceStrategySimulator:
    """
    Simulates race scenarios and evaluates pit window decisions.
    """

    def __init__(
        self,
        predictor: LapTimePredictor,
        default_pit_loss: float = 22.0,
        margin_threshold: float = 1.0
    ):
        self.predictor = predictor
        self.default_pit_loss = default_pit_loss
        self.margin_threshold = margin_threshold

    def evaluate_pit_decision(
        self,
        remaining_laps: int,
        current_tyre_life: int,
        current_compound: str,
        new_compound: Optional[str] = None,
        pit_loss: Optional[float] = None,
        position: int = 3
    ) -> StrategyRecommendation:
        """
        Evaluates 'Pit Now' vs 'Stay Out to Planned Window' over remaining race distance.
        """
        if not self.predictor.is_trained:
            raise RuntimeError("Cannot run simulation without a trained LapTimePredictor.")

        if remaining_laps <= 1:
            return StrategyRecommendation(
                decision="STAY_OUT",
                time_delta_seconds=0.0,
                stay_out_total_time=0.0,
                pit_now_total_time=0.0,
                pit_loss_seconds=0.0,
                remaining_laps=remaining_laps,
                explanation="Final lap or race complete. Pitting is non-optimal."
            )

        current_compound_upper = current_compound.upper()
        
        # Dynamically infer logical target compound if not provided
        if new_compound is None:
            target_compound = DEFAULT_NEXT_COMPOUND.get(current_compound_upper, 'HARD')
        else:
            target_compound = new_compound.upper()

        applied_pit_loss = pit_loss if pit_loss is not None else self.default_pit_loss
        max_life = COMPOUND_MAX_LIFE.get(current_compound_upper, 25)

        # -------------------------------------------------------------
        # SCENARIO 1: PIT NOW
        # Complete 1 in-lap on current tyres, then pit for target compound
        # -------------------------------------------------------------
        in_lap_time = float(self.predictor.predict_stint_trajectory(
            starting_tyre_life=current_tyre_life,
            compound=current_compound_upper,
            horizon_laps=1,
            position=position
        )[0])

        fresh_stint_laps = self.predictor.predict_stint_trajectory(
            starting_tyre_life=1,
            compound=target_compound,
            horizon_laps=remaining_laps - 1,
            position=position
        )
        pit_now_total_time = in_lap_time + applied_pit_loss + float(fresh_stint_laps.sum())

        # -------------------------------------------------------------
        # SCENARIO 2: STAY OUT
        # Continue current stint until optimal pit lap (max_life), then pit
        # -------------------------------------------------------------
        laps_until_planned_pit = max(0, max_life - current_tyre_life)

        #if laps_until_planned_pit >= remaining_laps or laps_until_planned_pit == 0:
        if laps_until_planned_pit >= remaining_laps:
            stay_out_laps = self.predictor.predict_stint_trajectory(
                starting_tyre_life=current_tyre_life,
                compound=current_compound_upper,
                horizon_laps=remaining_laps,
                position=position
            )
            stay_out_total_time = float(stay_out_laps.sum())
        else:
            first_stint = self.predictor.predict_stint_trajectory(
                starting_tyre_life=current_tyre_life,
                compound=current_compound_upper,
                horizon_laps=laps_until_planned_pit,
                position=position
            )
            second_stint_laps = remaining_laps - laps_until_planned_pit
            second_stint = self.predictor.predict_stint_trajectory(
                starting_tyre_life=1,
                compound=target_compound,
                horizon_laps=second_stint_laps,
                position=position
            )
            stay_out_total_time = float(first_stint.sum()) + applied_pit_loss + float(second_stint.sum())

        # -------------------------------------------------------------
        # DELTA & RECOMMENDATION
        # -------------------------------------------------------------
        time_delta = stay_out_total_time - pit_now_total_time

        if time_delta > self.margin_threshold:
            decision = "PIT_NOW"
            explanation = (
                f"BOX THIS LAP. Pitting now for fresh {target_compound} yields a net gain of "
                f"{time_delta:.2f}s over extending current {current_compound_upper} stint."
            )
        else:
            decision = "STAY_OUT"
            explanation = (
                f"STAY OUT. Pitting now for {target_compound} is suboptimal by {abs(time_delta):.2f}s. "
                f"Extend current {current_compound_upper} stint."
            )

        logger.info(f"Simulation result: Decision={decision} | Delta={time_delta:+.2f}s")

        return StrategyRecommendation(
            decision=decision,
            time_delta_seconds=round(time_delta, 3),
            stay_out_total_time=round(stay_out_total_time, 3),
            pit_now_total_time=round(pit_now_total_time, 3),
            pit_loss_seconds=applied_pit_loss,
            remaining_laps=remaining_laps,
            explanation=explanation
        )