# Create a new file named "preprocessing.jl" and add the following code:

using DataFrames
using Statistics
using Random
using StatsBase

# Function to normalize the data
function normalize_series(data)
    μ = mean(data)
    σ = std(data)
    return (data .- μ) ./ σ, μ, σ
end

# Function to denormalize the data
function denormalize_series(normalized_data, μ, σ)
    return normalized_data .* σ .+ μ
end

# Function to create sliding windows for time series prediction
function create_windows(data, window_size, horizon=1)
    X, y = [], []
    for i in 1:(length(data) - window_size - horizon + 1)
        push!(X, data[i:(i + window_size - 1)])
        push!(y, data[(i + window_size):(i + window_size + horizon - 1)])
    end
    return hcat(X...), hcat(y...)
end

# Function to prepare data for LSTM
function prepare_data_for_lstm(df, value_column, window_size=60, horizon=1, train_ratio=0.7)
    # Extract the value column
    values = df[!, value_column]
    
    # Normalize the data
    normalized_values, μ, σ = normalize_series(values)
    
    # Create sliding windows
    X, y = create_windows(normalized_values, window_size, horizon)
    
    # Determine train/test split index
    split_idx = Int(floor(size(X, 2) * train_ratio))
    
    # Split into training and testing sets
    X_train = X[:, 1:split_idx]
    y_train = y[:, 1:split_idx]
    X_test = X[:, (split_idx+1):end]
    y_test = y[:, (split_idx+1):end]
    
    # Reshape for LSTM: (sequence_length, features, batch_size)
    X_train_reshaped = reshape(X_train, (window_size, 1, size(X_train, 2)))
    X_test_reshaped = reshape(X_test, (window_size, 1, size(X_test, 2)))
    
    # Return all necessary components
    return X_train_reshaped, y_train, X_test_reshaped, y_test, μ, σ
end

# Example usage:
# df = your_dataframe
# X_train, y_train, X_test, y_test, μ, σ = prepare_data_for_lstm(df, "value")