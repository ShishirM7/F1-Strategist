# F1 Strategist

## Overview
F1 Strategist is an analytical tool designed to model and optimize Formula 1 race strategies. Strategic decisions during a Grand Prix rely on balancing tire degradation, pit-stop window traffic, and pace differentials. This tool processes session telemetry to predict compound performance and calculate optimal stint lengths for race scenarios.

## Features
* Automated extraction and parsing of F1 lap timing and telemetry data.
* Tire degradation modeling across soft, medium, and hard compounds.
* Race pace simulation and pit-window optimization.
* Visualization of driver pace deltas and stint progression.

## Technologies
* Programming Language: Python
* Data Libraries: FastF1, Pandas, NumPy
* Visualization: Matplotlib, Seaborn

## Project Structure
```text
F1-Strategist/
├── data/               # Telemetry cache and session data
├── modules/            # Degradation models and timing calculators
├── main.py             # Main execution script
└── notebooks/          # Exploratory analysis and strategy plots

```
## Results
* Accurately derived tire degradation coefficients across various stint lengths.
* Produced clear strategy comparison plots identifying optimal pit windows under clear-air conditions.

## Future Improvements
* Implementation of real-time Safety Car and Virtual Safety Car probability weighting.
* Dynamic track evolution modeling based on ambient and track temperature shifts.
