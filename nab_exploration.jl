# Update the nab_exploration.jl file with this code:

using Pkg
using CSV
using DataFrames
using Plots
using Dates

# Function to load and explore a NAB dataset
function explore_nab_dataset(file_path)
    # Load the data
    df = CSV.read(file_path, DataFrame)
    
    # Print basic information
    println("Dataset: ", file_path)
    println("Number of rows: ", nrow(df))
    println("Columns: ", names(df))
    
    # Extract timestamp and value columns (standard in NAB datasets)
    timestamp_col = names(df)[1]
    value_col = names(df)[2]
    
    # Sample timestamp to see its format
    println("Sample timestamp: ", df[1, timestamp_col])
    
    # Convert timestamps to DateTime objects with the correct format
    # NAB typically uses ISO 8601 format
    try
        df[!, timestamp_col] = DateTime.(df[!, timestamp_col], "yyyy-mm-dd HH:MM:SS.s")
    catch
        try
            df[!, timestamp_col] = DateTime.(df[!, timestamp_col], "yyyy-mm-dd'T'HH:MM:SS.s")
        catch
            println("Could not parse datetime. Keeping as string for plotting.")
        end
    end
    
    # Plot the time series
    p = plot(1:nrow(df), df[!, value_col], 
        title="Time Series Data",
        xlabel="Time Index",
        ylabel="Value",
        legend=false)
    
    # Return the DataFrame and plot
    return df, p
end

# Example usage:
dataset_path = "NAB/data/realKnownCause/ambient_temperature_system_failure.csv"
df, p = explore_nab_dataset(dataset_path)

# Display the first few rows
println(first(df, 5))

# Display the plot
display(p)