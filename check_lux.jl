# Create a file named check_lux.jl
using Lux
using Pkg

# Print package version
println("Lux package version: ", Pkg.installed()["Lux"])

# List all exported names from Lux
println("\nExported names from Lux:")
for name in names(Lux)
    println(name)
end

# Specifically check for RNN components
println("\nLooking for RNN components:")
for name in names(Lux, all=true)
    if occursin("RNN", string(name)) || occursin("LSTM", string(name)) || occursin("GRU", string(name))
        println(name)
    end
end