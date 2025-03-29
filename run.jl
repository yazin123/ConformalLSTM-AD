# run.jl - Command-line interface for the anomaly detection system

using ArgParse

include("main.jl")
include("benchmark_multiple_datasets.jl")

function parse_commandline()
    s = ArgParseSettings(description="Conformal LSTM Anomaly Detection for NAB")
    
    @add_arg_table s begin
        "--mode", "-m"
            help = "Operation mode: 'single' for single dataset, 'benchmark' for multiple datasets"
            default = "single"
        "--dataset", "-d"
            help = "Path to NAB dataset CSV for single mode"
            default = "NAB/data/realKnownCause/ambient_temperature_system_failure.csv"
        "--window", "-w"
            help = "Sliding window size for LSTM"
            arg_type = Int
            default = 60
        "--epochs", "-e"
            help = "Number of training epochs"
            arg_type = Int
            default = 50
        "--hidden", "-H"  # Changed from -h to -H
            help = "Hidden layer size for LSTM"
            arg_type = Int
            default = 64
        "--alpha", "-a"
            help = "Significance level for conformal prediction (lower = wider intervals)"
            arg_type = Float64
            default = 0.1
        "--output", "-o"
            help = "Output directory for results"
            default = "results"
    end
    
    return parse_args(s)
end

function run_single_dataset(args)
    println("Running anomaly detection on a single dataset:")
    println("  Window size: $(args["window"])")
    println("  Hidden size: $(args["hidden"])")
    println("  Epochs: $(args["epochs"])")
    println("  Alpha: $(args["alpha"])")
    
    # Get current directory
    current_dir = pwd()
    println("Current working directory: $(current_dir)")
    
    # Construct full path to dataset
    dataset_path = joinpath(current_dir, args["dataset"])
    println("Full dataset path: $(dataset_path)")
    
    # Check if file exists
    if !isfile(dataset_path)
        println("ERROR: File not found: $(dataset_path)")
        return
    end
    
    # Create output directory if it doesn't exist
    output_dir = joinpath(current_dir, args["output"])
    mkpath(output_dir)
    println("Output directory: $(output_dir)")
    
    # Save current directory and change to output directory
    original_dir = pwd()
    cd(output_dir)
    println("Changed working directory to: $(pwd())")
    
    try
        # Call main from here with the output directory as the working directory
        model, ps, st, icp, anomalies = main(
            dataset_path,
            args["window"],
            args["hidden"],
            args["epochs"],
            args["alpha"]
        )
        
        println("Results saved to $(pwd())")
    finally
        # Change back to original directory
        cd(original_dir)
    end
end

function run_benchmark(args)
    println("Running benchmark on multiple NAB datasets:")
    println("  Window size: $(args["window"])")
    println("  Hidden size: $(args["hidden"])")
    println("  Epochs: $(args["epochs"])")
    println("  Alpha: $(args["alpha"])")
    
    # Create output directory if it doesn't exist
    output_dir = joinpath(pwd(), args["output"])
    mkpath(output_dir)
    println("Output directory: $(output_dir)")
    
    # Save current directory and change to output directory
    original_dir = pwd()
    cd(output_dir)
    println("Changed working directory to: $(pwd())")
    
    try
        # Call benchmark_multiple_datasets()
        results = benchmark_multiple_datasets(
            args["window"],
            args["hidden"],
            args["epochs"],
            args["alpha"]
        )
        
        # Save a summary of the benchmark results
        summary_file = "benchmark_summary.csv"
        CSV.write(summary_file, results)
        println("Benchmark summary saved to: $(joinpath(pwd(), summary_file))")
        
        println("Benchmark results saved to $(pwd())")
    finally
        # Change back to original directory
        cd(original_dir)
    end
end

function main_cli()
    args = parse_commandline()
    
    if args["mode"] == "single"
        run_single_dataset(args)
    elseif args["mode"] == "benchmark"
        run_benchmark(args)
    else
        println("Invalid mode: $(args["mode"]). Use 'single' or 'benchmark'.")
    end
end

# Run the CLI
main_cli()