# Create a new file named "test_pipeline.jl" and add the following code:

using Test

include("main.jl")

@testset "Anomaly Detection Pipeline Tests" begin
    # Test data preprocessing
    @testset "Preprocessing" begin
        # Generate synthetic data
        data = sin.(1:100) .+ 0.1 .* randn(100)
        
        # Test normalization
        norm_data, μ, σ = normalize_series(data)
        @test isapprox(mean(norm_data), 0.0, atol=1e-10)
        @test isapprox(std(norm_data), 1.0, atol=1e-10)
        
        # Test denormalization
        denorm_data = denormalize_series(norm_data, μ, σ)
        @test isapprox(data, denorm_data, rtol=1e-10)
        
        # Test window creation
        window_size = 10
        X, y = create_windows(data, window_size, 1)
        @test size(X, 1) == window_size
        @test size(X, 2) == length(data) - window_size - 1 + 1
        @test size(y, 1) == 1
        @test size(y, 2) == length(data) - window_size - 1 + 1
    end
    
    # Test LSTM model creation
    @testset "LSTM Model" begin
        model, ps, st = create_lstm_model(1, 16, 1)
        
        # Test that model parameters exist
        @test !isempty(ps)
        
        # Create a small batch of data
        x = reshape(randn(10, 5), (10, 1, 5))  # (seq_len, features, batch_size)
        
        # Test forward pass
        y, st_new = model(x, ps, st)
        
        @test size(y) == (1, 5)  # (output_size, batch_size)
        @test st_new != st       # State should change after forward pass
    end
    
    # Test conformal prediction
    @testset "Conformal Prediction" begin
        # Create small synthetic datasets
        X_calib = reshape(randn(10, 20), (10, 1, 20))
        y_calib = randn(1, 20)
        X_test = reshape(randn(10, 5), (10, 1, 5))
        
        # Create a model
        model, ps, st = create_lstm_model(1, 16, 1)
        
        # Create a conformal predictor
        icp = create_conformal_predictor(model, ps, st, X_calib, y_calib)
        
        # Generate prediction intervals
        y_pred, intervals = predict_intervals(icp, model, ps, st, X_test)
        
        @test size(y_pred) == (1, 5)  # Point predictions
        @test length(intervals) == 5   # Prediction intervals
        
        # Check that intervals have lower and upper bounds
        @test all(interval -> length(interval) == 2, intervals)
        
        # Check that lower bounds are less than upper bounds
        @test all(interval -> interval[1] < interval[2], intervals)
    end
    
    println("All tests passed!")
end