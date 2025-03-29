# check_dir.jl
function check_directory_state()
    println("Current directory: $(pwd())")
    
    # Check if NAB directory exists
    println("NAB directory exists: $(isdir("NAB"))")
    
    # Check file path
    file_path = "NAB/data/realKnownCause/ambient_temperature_system_failure.csv"
    println("File exists: $(isfile(file_path))")
    
    # Try to generate an absolute path
    abs_path = abspath(file_path)
    println("Absolute path: $(abs_path)")
    println("Absolute path exists: $(isfile(abs_path))")
    
    # List directories in current path
    println("Directories in current path:")
    for dir in readdir()
        if isdir(dir)
            println("  - $dir")
        end
    end
end

check_directory_state()