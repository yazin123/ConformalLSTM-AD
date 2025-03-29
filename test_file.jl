# test_file.jl
using CSV
using DataFrames

function test_file_access()
    # Test 1: Check if file exists using Julia's isfile
    file_path = "NAB/data/realKnownCause/ambient_temperature_system_failure.csv"
    println("Test 1: Checking if file exists with isfile")
    println("File exists: $(isfile(file_path))")
    println("Current directory: $(pwd())")
    
    # Test 2: Try to read the file using readlines
    println("\nTest 2: Trying to read file with readlines")
    try
        lines = readlines(file_path, keep=true)
        println("Successfully read file with readlines")
        println("First few lines:")
        for i in 1:min(5, length(lines))
            println(lines[i])
        end
    catch e
        println("Error reading file with readlines: $e")
    end
    
    # Test 3: Try to read with CSV.read
    println("\nTest 3: Trying to read file with CSV.read")
    try
        df = CSV.read(file_path, DataFrame)
        println("Successfully read file with CSV.read")
        println("DataFrame shape: $(size(df))")
        println("Column names: $(names(df))")
        println("First few rows:")
        println(first(df, 5))
    catch e
        println("Error reading file with CSV.read: $e")
    end
    
    # Test 4: Try with absolute path
    println("\nTest 4: Trying with absolute path")
    abs_path = abspath(file_path)
    println("Absolute path: $abs_path")
    try
        df = CSV.read(abs_path, DataFrame)
        println("Successfully read file with absolute path")
    catch e
        println("Error reading file with absolute path: $e")
    end
end

test_file_access()