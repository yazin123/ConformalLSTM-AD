# Create a new file named "nab_evaluation.jl" and add the following code:

using CSV
using DataFrames
using JSON
using Dates

"""
Function to load NAB ground truth labels from the labels file
"""
function load_nab_labels(dataset_name, labels_path="./NAB/labels/combined_labels.json")
    # Try to find the labels file
    if !isfile(labels_path)
        # Try without the leading ./
        alt_path = replace(labels_path, "./" => "")
        if isfile(alt_path)
            labels_path = alt_path
        else
            # Try going up one directory
            parent_path = joinpath("..", labels_path)
            if isfile(parent_path)
                labels_path = parent_path
            else
                # Try with absolute path based on NAB location
                nab_dir = ""
                for dir in [pwd(), dirname(pwd())]
                    if isdir(joinpath(dir, "NAB"))
                        nab_dir = joinpath(dir, "NAB")
                        break
                    end
                end
                
                if !isempty(nab_dir)
                    abs_path = joinpath(nab_dir, "labels", "combined_labels.json")
                    if isfile(abs_path)
                        labels_path = abs_path
                    end
                end
            end
        end
    end
    
    # Check if we found a valid path
    if !isfile(labels_path)
        println("Warning: Labels file not found at $labels_path")
        println("Current directory: $(pwd())")
        println("Creating empty labels for evaluation")
        return Dict{String, Vector{String}}()
    end
    
    # Load the labels JSON file
    println("Loading labels from: $labels_path")
    labels_json = JSON.parsefile(labels_path)
    
    # Get the labels for the specified dataset
    if haskey(labels_json, dataset_name)
        return labels_json[dataset_name]
    else
        println("Dataset $dataset_name not found in labels file")
        return []
    end
end

"""
Function to convert our anomaly predictions to the NAB format
"""
function convert_to_nab_format(df, anomalies, window_size, test_start_idx)
    # Create a copy of the original dataframe
    results_df = copy(df)
    
    # Add an anomaly score column (initialize with zeros)
    results_df.anomaly_score = zeros(nrow(results_df))
    
    # Map our binary anomalies to the test portion of the data
    # The test set starts at test_start_idx in the original data
    # We need to account for the window_size offset
    test_indices = (test_start_idx+window_size):(test_start_idx+window_size+length(anomalies)-1)
    
    # Set anomaly scores for detected anomalies (1.0 for anomaly, 0.0 for normal)
    for (i, idx) in enumerate(test_indices)
        if idx <= nrow(results_df)
            results_df[idx, :anomaly_score] = anomalies[i] ? 1.0 : 0.0
        end
    end
    
    return results_df
end

"""
Function to calculate basic evaluation metrics
"""
function calculate_metrics(predictions, ground_truth)
    # Extract timestamps where anomalies were detected
    detected_anomalies = Set(ground_truth[predictions .== 1])
    actual_anomalies = Set(ground_truth)
    
    # Calculate metrics
    true_positives = length(intersect(detected_anomalies, actual_anomalies))
    false_positives = length(setdiff(detected_anomalies, actual_anomalies))
    false_negatives = length(setdiff(actual_anomalies, detected_anomalies))
    
    # Compute precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return Dict(
        "precision" => precision,
        "recall" => recall,
        "f1_score" => f1,
        "true_positives" => true_positives,
        "false_positives" => false_positives,
        "false_negatives" => false_negatives
    )
end

"""
Main evaluation function
"""
function evaluate_model_on_nab(dataset_path, anomalies, window_size, test_start_idx)
    # Get the dataset name from the path
    dataset_name = basename(dataset_path)
    
    # Load the dataset
    df = CSV.read(dataset_path, DataFrame)
    
    # Load ground truth labels
    ground_truth_timestamps = load_nab_labels(dataset_name)
    
    # Convert string timestamps to DateTime objects
    ground_truth_datetimes = DateTime.(ground_truth_timestamps)
    
    # Create a binary ground truth vector
    ground_truth = zeros(Bool, nrow(df))
    for gt_time in ground_truth_datetimes
        # Find the index in the dataframe where the timestamp matches
        idx = findfirst(t -> DateTime(t) == gt_time, df[!, "timestamp"])
        if !isnothing(idx)
            ground_truth[idx] = true
        end
    end
    
    # Convert our anomaly predictions to NAB format
    results_df = convert_to_nab_format(df, anomalies, window_size, test_start_idx)
    
    # Calculate evaluation metrics
    metrics = calculate_metrics(results_df.anomaly_score, ground_truth)
    
    # Print evaluation results
    println("Evaluation Results for $dataset_name:")
    println("  Precision: $(round(metrics["precision"], digits=4))")
    println("  Recall: $(round(metrics["recall"], digits=4))")
    println("  F1 Score: $(round(metrics["f1_score"], digits=4))")
    println("  True Positives: $(metrics["true_positives"])")
    println("  False Positives: $(metrics["false_positives"])")
    println("  False Negatives: $(metrics["false_negatives"])")
    
    return metrics, results_df
end

# Example usage (to be called from main.jl):
# metrics, results_df = evaluate_model_on_nab(
#     dataset_path, 
#     anomalies, 
#     window_size, 
#     calib_idx
# )