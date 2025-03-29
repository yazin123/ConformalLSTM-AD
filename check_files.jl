# create a file called check_files.jl
function check_nab_files()
    # Check if NAB directory exists in current directory
    if isdir("NAB")
        println("NAB directory found in current directory")
        
        # Check for a specific dataset file
        test_file = "NAB/data/realKnownCause/ambient_temperature_system_failure.csv"
        if isfile(test_file)
            println("Found test file: $test_file")
        else
            println("Test file not found at: $test_file")
            
            # Try to find data directory
            if isdir("NAB/data")
                println("NAB data directory exists")
                
                # List subdirectories in data
                println("Available directories in NAB/data:")
                for dir in readdir("NAB/data")
                    println("  - $dir")
                    
                    # List files in this directory
                    if isdir(joinpath("NAB/data", dir))
                        files = readdir(joinpath("NAB/data", dir))
                        if length(files) > 0
                            println("    Files: $(files[1:min(3, length(files))])...")
                        else
                            println("    No files found")
                        end
                    end
                end
            else
                println("NAB/data directory not found")
            end
        end
    else
        println("NAB directory not found in current directory")
        
        # Check parent directory
        if isdir("../NAB")
            println("NAB directory found in parent directory")
        else
            println("NAB directory not found in parent directory")
        end
    end
end

check_nab_files()