# main.jl - Main pipeline for anomaly detection

using CSV
using DataFrames
using Plots
using Dates
using Random
using Statistics
using MLUtils

# Include our previously created modules
include("preprocessing.jl")
include("lstm_model.jl")
include("conformal_prediction.jl")
include("nab_evaluation.jl")

# Set random seed for reproducibility
Random.seed!(123)

function main(dataset_path="NAB/data/realKnownCause/ambient_temperature_system_failure.csv",
    window_size=60, 
    hidden_size=64, 
    epochs=50, 
    alpha=0.1)
    # Step 1: Load and preprocess a NAB dataset
    println("Loading dataset...")
    
    # Normalize path (remove any leading ./)
    if startswith(dataset_path, "./")
        dataset_path = dataset_path[3:end]
    end
    
    # Check if file exists
    if !isfile(dataset_path)
        error("File not found: $dataset_path")
    end
    
    # Try reading with absolute path
    abs_path = abspath(dataset_path)
    println("Trying to load file from: $abs_path")
    
    df = CSV.read(abs_path, DataFrame)
    
    # Step 2: Prepare data for LSTM
    println("Preparing data...")
    horizon = 1       # Predict 1 step ahead
    
    # We need to split data into train, calibration, and test sets
    # First, normalize the data
    values = df[!, "value"]
    normalized_values, μ, σ = normalize_series(values)
    
    # Create sliding windows
    X, y = create_windows(normalized_values, window_size, horizon)
    
    # Determine split indices (60% train, 20% calibration, 20% test)
    n_samples = size(X, 2)
    train_idx = Int(floor(n_samples * 0.6))
    calib_idx = Int(floor(n_samples * 0.8))
    
    # Split the data
    X_train = X[:, 1:train_idx]
    y_train = y[:, 1:train_idx]
    X_calib = X[:, (train_idx+1):calib_idx]
    y_calib = y[:, (train_idx+1):calib_idx]
    X_test = X[:, (calib_idx+1):end]
    y_test = y[:, (calib_idx+1):end]
    
    # Reshape for our neural network model
    # For a simple feedforward network, we want (features, samples)
    X_train = reshape(X_train, (window_size, train_idx))
    y_train = reshape(y_train, (horizon, train_idx))
    
    X_calib = reshape(X_calib, (window_size, calib_idx - train_idx))
    y_calib = reshape(y_calib, (horizon, calib_idx - train_idx))
    
    X_test = reshape(X_test, (window_size, n_samples - calib_idx))
    y_test = reshape(y_test, (horizon, n_samples - calib_idx))
    
    # Step 3: Create and train LSTM model
    println("Creating and training LSTM model...")
    model, ps, st = create_lstm_model(window_size, hidden_size, horizon)  # Input size is window_size, output size is horizon
    
    # Train the model
    ps, st, losses = train_lstm(model, ps, st, X_train, y_train, epochs)  # dynamic epochs for demonstration
    
    # Plot training loss
    p_loss = plot(losses, title="Training Loss", xlabel="Epoch", ylabel="MSE", legend=false)
    display(p_loss)
    savefig(p_loss, "training_loss.png")
    println("Training loss plot saved as training_loss.png")
    
    # Step 4: Create conformal predictor
    println("Creating conformal predictor...")
    icp = create_conformal_predictor(model, ps, st, X_calib, y_calib, alpha)
    
    # Step 5: Make predictions and detect anomalies
    println("Making predictions and detecting anomalies...")
    y_pred, intervals = predict_intervals(icp, model, ps, st, X_test)
    
    # Denormalize predictions and actual values for plotting
    y_test_denorm = denormalize_series(y_test, μ, σ)
    y_pred_denorm = denormalize_series(y_pred, μ, σ)
    
    # Denormalize intervals
    intervals_denorm = [(denormalize_series(interval[1], μ, σ), 
                        denormalize_series(interval[2], μ, σ)) for interval in intervals]
    
    # Detect anomalies
    anomalies = detect_anomalies(y_test, y_pred, intervals)
    
    # Step 6: Visualize results
    println("Visualizing results...")
    
    # Create a time index for the test set (just for visualization)
    test_size = size(y_test, 2)
    test_indices = (calib_idx+1):(calib_idx+test_size)
    
    # Ensure all data is in vector form for plotting
    y_test_plot = vec(y_test_denorm)
    y_pred_plot = vec(y_pred_denorm)
    
    # Plot predictions vs actual
    p = plot(test_indices, y_test_plot, label="Actual", linewidth=2)
    plot!(test_indices, y_pred_plot, label="Predicted", linewidth=2)
    
    # Plot prediction intervals
    lower_bounds = [interval[1] for interval in intervals_denorm]
    upper_bounds = [interval[2] for interval in intervals_denorm]
    plot!(test_indices, lower_bounds, fillrange=upper_bounds, alpha=0.3, 
         label="Prediction Interval", color=:blue)
    
    # Highlight anomalies
    if any(anomalies)
        # Convert the anomalies to indices first
        anomaly_idx = findall(anomalies)
        
        # Check if there are any valid indices
        if !isempty(anomaly_idx)
            # Make sure indices are in range
            valid_idx = filter(i -> i <= length(test_indices), anomaly_idx)
            
            if !isempty(valid_idx)
                # Then use these indices to access the test_indices
                anomaly_indices = test_indices[valid_idx]
                
                # Extract the corresponding y values, ensuring dimensions match
                if size(y_test_denorm, 1) == 1
                    anomaly_values = vec(y_test_denorm[:, valid_idx])
                else
                    anomaly_values = y_test_denorm[valid_idx]
                end
                
                scatter!(anomaly_indices, anomaly_values, markersize=6, 
                        markershape=:circle, color=:red, label="Anomalies")
            end
        end
    else
        println("No anomalies detected.")
    end
    
    # Set plot attributes
    title!("Conformal Prediction - Anomaly Detection")
    xlabel!("Time Index")
    ylabel!("Value")
    
    # Display and save the plot
    display(p)
    savefig(p, "anomaly_detection_results.png")
    println("Anomaly detection plot saved as anomaly_detection_results.png")
    
    # Print statistics
    n_anomalies = sum(anomalies)
    anomaly_ratio = n_anomalies / length(anomalies)
    println("Number of detected anomalies: $n_anomalies ($(round(anomaly_ratio * 100, digits=2))% of test data)")

    # Step 7: Evaluate on NAB benchmark
    println("Evaluating on NAB benchmark...")
    metrics, results_df = evaluate_model_on_nab(
        dataset_path,
        anomalies,
        window_size,
        calib_idx
    )
    
    # Save results to CSV - use full path in current directory
    output_file = "anomaly_detection_results.csv"
    println("Saving results to: $(joinpath(pwd(), output_file))")
    CSV.write(output_file, results_df)
    
    # Save metrics to a separate CSV
    metrics_df = DataFrame(
        Metric = ["Precision", "Recall", "F1_Score", "True_Positives", "False_Positives", "False_Negatives"],
        Value = [metrics["precision"], metrics["recall"], metrics["f1_score"], 
                metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]]
    )
    metrics_file = "evaluation_metrics.csv"
    CSV.write(metrics_file, metrics_df)
    println("Metrics saved to: $(joinpath(pwd(), metrics_file))")
    
    println("Evaluation completed.")
    
    return model, ps, st, icp, anomalies
end

# Only run if this script is directly executed (not when included)
if abspath(PROGRAM_FILE) == @__FILE__
    model, ps, st, icp, anomalies = main()
end