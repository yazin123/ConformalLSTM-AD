# Updated benchmark_multiple_datasets.jl

using CSV
using DataFrames
using Statistics
using Plots

include("preprocessing.jl")
include("lstm_model.jl")
include("conformal_prediction.jl")
include("nab_evaluation.jl")

function benchmark_multiple_datasets(window_size=60, 
                                     hidden_size=64, 
                                     epochs=50, 
                                     alpha=0.1)
    # List of NAB datasets to benchmark
    datasets = [
        "NAB/data/realKnownCause/ambient_temperature_system_failure.csv",
        "./NAB/data/realKnownCause/cpu_utilization_asg_misconfiguration.csv",
        "./NAB/data/realKnownCause/ec2_request_latency_system_failure.csv",
        "./NAB/data/realKnownCause/nyc_taxi.csv",
        "./NAB/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"
    ]
    
    # Store results for each dataset
    all_results = DataFrame(
        Dataset = String[],
        Precision = Float64[],
        Recall = Float64[],
        F1Score = Float64[],
        AnomalyCount = Int[]
    )
    
    # Run anomaly detection on each dataset
    for dataset_path in datasets
        println("\n=== Processing dataset: $(basename(dataset_path)) ===")
        
        try
            # Load dataset
            df = CSV.read(dataset_path, DataFrame)
            
            # Prepare data
            values = df[!, "value"]
            normalized_values, μ, σ = normalize_series(values)
            X, y = create_windows(normalized_values, window_size, 1)
            
            # Split data
            n_samples = size(X, 2)
            train_idx = Int(floor(n_samples * 0.6))
            calib_idx = Int(floor(n_samples * 0.8))
            
            X_train = X[:, 1:train_idx]
            y_train = y[:, 1:train_idx]
            X_calib = X[:, (train_idx+1):calib_idx]
            y_calib = y[:, (train_idx+1):calib_idx]
            X_test = X[:, (calib_idx+1):end]
            y_test = y[:, (calib_idx+1):end]
            
            # Reshape for neural network model
            X_train = reshape(X_train, (window_size, train_idx))
            y_train = reshape(y_train, (1, train_idx))
            
            X_calib = reshape(X_calib, (window_size, calib_idx - train_idx))
            y_calib = reshape(y_calib, (1, calib_idx - train_idx))
            
            X_test = reshape(X_test, (window_size, n_samples - calib_idx))
            y_test = reshape(y_test, (1, n_samples - calib_idx))
            
            # Create and train model
            println("Training LSTM model...")
            model, ps, st = create_lstm_model(window_size, hidden_size, 1)
            ps, st, _ = train_lstm(model, ps, st, X_train, y_train, epochs)
            
            # Create conformal predictor
            println("Creating conformal predictor...")
            icp = create_conformal_predictor(model, ps, st, X_calib, y_calib)
            
            # Make predictions and detect anomalies
            println("Detecting anomalies...")
            y_pred, intervals = predict_intervals(icp, model, ps, st, X_test)
            anomalies = detect_anomalies(y_test, y_pred, intervals)
            
            # Evaluate on NAB benchmark
            println("Evaluating...")
            metrics, _ = evaluate_model_on_nab(dataset_path, anomalies, window_size, calib_idx)
            
            # Store results
            push!(all_results, (
                basename(dataset_path),
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                sum(anomalies)
            ))
            
        catch e
            println("Error processing dataset $(basename(dataset_path)): $e")
        end
    end
    
    # Print summary of results
    println("\n=== Summary of Results ===")
    println(all_results)
    
    # Calculate average metrics
    if nrow(all_results) > 0
        avg_precision = mean(all_results.Precision)
        avg_recall = mean(all_results.Recall)
        avg_f1 = mean(all_results.F1Score)
        
        println("\nAverage Metrics:")
        println("  Precision: $(round(avg_precision, digits=4))")
        println("  Recall: $(round(avg_recall, digits=4))")
        println("  F1 Score: $(round(avg_f1, digits=4))")
        
        # Create a bar chart for F1 scores
        if nrow(all_results) > 0
            p = bar(all_results.Dataset, all_results.F1Score,
                title="F1 Scores Across NAB Datasets",
                xlabel="Dataset",
                ylabel="F1 Score",
                legend=false,
                rotation=45,
                xtickfontsize=8)
            
            # Add a horizontal line for average F1
            hline!([avg_f1], linestyle=:dash, color=:red, label="Average F1")
            
            display(p)
            savefig(p, "benchmark_results.png")
        end
    else
        println("No successful dataset evaluations to average.")
    end
    
    # Save results to CSV
    CSV.write("benchmark_results.csv", all_results)
    
    return all_results
end