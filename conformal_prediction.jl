# Updated conformal_prediction.jl - Custom implementation

using Statistics
using Random

# Custom implementation of an inductive conformal predictor
mutable struct InductiveConformalPredictor
    model
    model_params
    model_state
    calibration_scores
    alpha
end

# Function to create a conformal predictor based on model predictions
function create_conformal_predictor(model, ps, st, X_calib, y_calib, alpha=0.1)
    # Make predictions on calibration set
    y_pred, _ = model(X_calib, ps, st)
    
    # Compute absolute errors (nonconformity scores)
    calibration_scores = vec(abs.(y_pred .- y_calib))
    
    # Create and return conformal predictor
    return InductiveConformalPredictor(model, ps, st, calibration_scores, alpha)
end

# Function to generate prediction intervals using conformal prediction
function predict_intervals(icp, model, ps, st, X, alpha=0.1)
    # Get point predictions
    y_pred, _ = model(X, ps, st)
    
    # Get the quantile from calibration scores
    # For 90% confidence (alpha=0.1), we use the 90th percentile of calibration scores
    q = quantile(icp.calibration_scores, 1 - alpha)
    
    # Create prediction intervals
    intervals = [(pred - q, pred + q) for pred in vec(y_pred)]
    
    return y_pred, intervals
end

# Function to detect anomalies based on prediction intervals
function detect_anomalies(y_true, y_pred, intervals)
    # Make sure y_true is a vector for consistent comparison
    y_true_vec = vec(y_true)
    
    # An observation is flagged as an anomaly if it falls outside the prediction interval
    lower_bounds = [interval[1] for interval in intervals]
    upper_bounds = [interval[2] for interval in intervals]
    
    # Make sure dimensions match
    anomalies = [y_true_vec[i] < lower_bounds[i] || y_true_vec[i] > upper_bounds[i] for i in 1:length(y_true_vec)]
    
    return anomalies
end