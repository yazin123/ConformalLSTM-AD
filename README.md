# Conformal LSTM Anomaly Detection for NAB

This project implements a conformal LSTM classifier for anomaly detection using the Numenta Anomaly Benchmark (NAB) dataset. The system combines Long Short-Term Memory (LSTM) neural networks with Conformal Prediction to provide uncertainty estimates and robust anomaly detection.

## Overview

The project focuses on detecting anomalies in time series data by:
1. Predicting future values using neural networks
2. Generating statistically valid prediction intervals with conformal prediction
3. Identifying data points that fall outside these intervals as anomalies
4. Evaluating performance against the NAB benchmark

## Requirements

- Julia 1.6 or higher
- Required packages:
  - ConformalPrediction.jl
  - MLJ.jl
  - Lux.jl
  - CSV.jl
  - DataFrames.jl
  - Plots.jl
  - StatsBase.jl
  - Dates.jl
  - JSON.jl
  - Zygote.jl
  - Optimisers.jl
  - MLUtils.jl
  - ArgParse.jl

## Project Structure

- `preprocessing.jl`: Functions for data normalization and sliding window creation
- `lstm_model.jl`: LSTM model implementation using Lux.jl
- `conformal_prediction.jl`: Conformal prediction implementation
- `main.jl`: Main pipeline for single dataset analysis
- `nab_evaluation.jl`: Functions for evaluating results using NAB metrics
- `benchmark_multiple_datasets.jl`: Script for benchmarking on multiple NAB datasets
- `run.jl`: Command-line interface for running the project

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yazin123/ConformalLSTM-AD.git
cd conformal-lstm-nab

```

2. if there is no NAB folder, clone it
```bash
git clone https://github.com/numenta/NAB.git
```

3. Install required Julia packages:
```bash
using Pkg
Pkg.add(["ConformalPrediction", "MLJ", "Lux", "CSV", "DataFrames", "Plots", "StatsBase", "Dates", "JSON", "Zygote", "Optimisers", "MLUtils", "ArgParse"])
```

4. Run with default parameters
```bash
julia run.jl
```

5. Run on a specific dataset
```bash
julia run.jl -d NAB/data/realKnownCause/ambient_temperature_system_failure.csv
```

6. Run with custom parameters
```bash
julia run.jl -w 120 -e 100 -H 128 -a 0.05
```

7. Run benchmark on multiple datasets
```bash
julia run.jl -m benchmark
```

Methods

1. Data Preprocessing: Time series data is normalized and transformed into sliding windows.
2. LSTM Model: Neural network trained to predict future values in time series.
3. Conformal Prediction: Provides prediction intervals with calibrated confidence levels.
4. Anomaly Detection: Points outside prediction intervals are flagged as anomalies.
5. Evaluation: Results are evaluated using NAB metrics including precision, recall, and F1 score.


