import logging
import argparse
import numpy as np
import pandas as pd
from src.ingestion import RaceDataLoader
from src.features import FeatureEngineer
from src.models import LapTimePredictor
from src.simulation import RaceStrategySimulator, DEFAULT_NEXT_COMPOUND
from src.visualization import StrategyVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("F1StrategyAssistant")

def run_pipeline(
    year: int = 2023,
    race: str = 'Monza',
    target_driver: str = 'VER',
    decision_lap: int = 20,
    target_compound: str = None
):
    logger.info("==================================================")
    logger.info(f"STARTING AI RACE STRATEGY ASSISTANT: {year} {race}")
    logger.info("==================================================")

    # 1. Ingestion
    loader = RaceDataLoader(cache_dir="data/cache")
    raw_laps = loader.load_session_laps(year=year, race=race, session_type='R')

    if raw_laps is None or raw_laps.empty:
        logger.error("Data ingestion failed. Exiting pipeline.")
        return

    # 2. Feature Engineering
    engineer = FeatureEngineer(pit_loss_seconds=22.0)
    features_df = engineer.transform_lap_data(raw_laps)

    # 3. Model Training
    predictor = LapTimePredictor()
    metrics = predictor.train(features_df)
    logger.info(f"Model Training Complete. MAE: {metrics['mae']:.3f}s | RMSE: {metrics['rmse']:.3f}s")

    # 4. Extract Target Driver Context
    driver_laps = features_df[(features_df['Driver'] == target_driver) & (features_df['LapNumber'] <= decision_lap)]
    
    if driver_laps.empty:
        logger.error(f"No lap records found for driver {target_driver} up to lap {decision_lap}.")
        return

    current_state = driver_laps.iloc[-1]
    current_tyre_life = int(current_state['TyreLife'])
    #current_compound = str(current_state['Compound']).upper()
    current_compound = "HARD"
    total_race_laps = int(features_df['LapNumber'].max())
    remaining_laps = total_race_laps - decision_lap

    if target_compound is None:
        new_compound = DEFAULT_NEXT_COMPOUND.get(current_compound, 'HARD')
    else:
        new_compound = target_compound.upper()

    logger.info(f"Driver: {target_driver} | Lap: {decision_lap}/{total_race_laps}")
    logger.info(f"Active Compound: {current_compound} | Tyre Age: {current_tyre_life} laps")
    logger.info(f"Target Compound if Pitting: {new_compound}")

    # 5. Simulation
    simulator = RaceStrategySimulator(predictor=predictor, default_pit_loss=22.0)
    recommendation = simulator.evaluate_pit_decision(
        remaining_laps=remaining_laps,
        current_tyre_life=current_tyre_life,
        current_compound=current_compound,
        new_compound=new_compound,
        position=int(current_state['Position'])
    )

    logger.info("\n" + "="*40)
    logger.info("STRATEGY RECOMMENDATION RESULT")
    logger.info("="*40)
    logger.info(f"DECISION:           {recommendation.decision}")
    logger.info(f"PROJECTED DELTA:    {recommendation.time_delta_seconds:+.2f} seconds")
    logger.info(f"EXPLANATION:        {recommendation.explanation}")
    logger.info("="*40)

    # 6. Visualization Data Construction
    visualizer = StrategyVisualizer()
    
    stay_out_trajectory = predictor.predict_stint_trajectory(
        starting_tyre_life=current_tyre_life,
        compound=current_compound,
        horizon_laps=remaining_laps
    )

    in_lap_time = predictor.predict_stint_trajectory(
        starting_tyre_life=current_tyre_life,
        compound=current_compound,
        horizon_laps=1
    )
    fresh_stint_laps = predictor.predict_stint_trajectory(
        starting_tyre_life=1,
        compound=new_compound,
        horizon_laps=remaining_laps - 1
    )
    pit_now_trajectory = np.concatenate([in_lap_time, fresh_stint_laps])

    visualizer.plot_strategy_comparison(
        stay_out_laps=stay_out_trajectory,
        pit_now_laps=pit_now_trajectory,
        pit_loss=22.0,
        current_lap=decision_lap,
        driver_code=target_driver,
        save_path="strategy_recommendation.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 AI Race Strategy Assistant")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--race", type=str, default="Silverstone")
    parser.add_argument("--driver", type=str, default="VER")
    parser.add_argument("--lap", type=int, default=30)
    parser.add_argument("--compound", type=str, default="SOFT")
    #parser.add_argument("--initial-compound", type=str, default=None, help="Override active tyre compound")

    args = parser.parse_args()

    run_pipeline(
        year=args.year,
        race=args.race,
        target_driver=args.driver,
        decision_lap=args.lap,
        target_compound=args.compound
#        initial_compound=args.initial_compound
    )