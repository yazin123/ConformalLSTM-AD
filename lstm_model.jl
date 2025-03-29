# Even more simplified lstm_model.jl

using Lux
using Random
using Optimisers
using Zygote
using MLUtils
using Statistics

# Set random seed for reproducibility
rng = Random.default_rng()
Random.seed!(rng, 123)

# Function to create a simple neural network model (no LSTM for now)
function create_lstm_model(input_size, hidden_size, output_size)
    model = Chain(
        Dense(input_size => hidden_size, tanh),
        Dense(hidden_size => hidden_size, tanh),
        Dense(hidden_size => output_size)
    )
    
    # Initialize parameters
    ps, st = Lux.setup(rng, model)
    
    return model, ps, st
end

# Loss function (Mean Squared Error)
function loss_function(model, ps, st, x, y)
    y_pred, st = model(x, ps, st)
    return mean((y_pred .- y).^2), st
end

# Function to train the model
function train_lstm(model, ps, st, X_train, y_train, epochs=100, learning_rate=0.01, batch_size=32)
    # Setup optimizer
    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, ps)
    
    # We'll use the data directly without reshaping
    data_loader = DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
    
    # Training loop
    losses = []
    for epoch in 1:epochs
        epoch_loss = 0.0
        batch_count = 0
        
        for (x_batch, y_batch) in data_loader
            # Compute gradients
            l, gs = Zygote.withgradient(p -> loss_function(model, p, st, x_batch, y_batch)[1], ps)
            
            # Update parameters
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
            
            # Accumulate loss
            epoch_loss += l
            batch_count += 1
        end
        
        # Average loss for the epoch
        avg_loss = epoch_loss / batch_count
        push!(losses, avg_loss)
        
        # Print progress
        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $avg_loss")
        end
    end
    
    return ps, st, losses
end

# Function to evaluate the model
function evaluate_lstm(model, ps, st, X_test, y_test)
    # Make predictions
    y_pred, _ = model(X_test, ps, st)
    
    # Calculate MSE
    mse = mean((y_pred .- y_test).^2)
    
    return y_pred, mse
end