# Import required packages
using Lux                  # Neural network definition and training
using LuxCUDA
using TaylorDiff           # Automatic differentiation
using Optimization         # Optimization problem solving
using Optimisers           # Optimizers such as Adam
using Random               # Random number generation
using Plots                # Plotting utilities
using Zygote               # Automatic differentiation
using ComponentArrays      # Array utilities for handling model parameters
using Printf               # Advanced formatted printing
using LinearAlgebra        # Linear algebra operations
using OptimizationOptimisers  # Optimization extensions for Optimisers
using OptimizationOptimJL     # Optimization extensions for LBFGS
using Statistics           # Statistical operations

# Initial program setup
# Seeding for reproducibility
rng = Random.default_rng()
Random.seed!(rng, 0)

# Device configuration (CPU/GPU)
const DEVICE_CPU = cpu_device()  # CPU device
const DEVICE_GPU = gpu_device()  # GPU device (if available)

# Data preparation
# Input range and target output
const INPUT_RANGE = range(-1f0, 1f0, length=100)
input_data = reshape(collect(INPUT_RANGE), 1, :)  |> DEVICE_GPU # Input data
true_output = reshape(Float32.(sin.(10 * input_data)), 1, :)  # Target output
data = (input_data, true_output) |> DEVICE_GPU  # Move data to the selected device

# Reference function definition
target_function(x) = sin.(10 * x)

# Neural network definition
# Architecture: Network with multiple dense layers and tanh activation
const HIDDEN_UNITS = 100
model = Chain(
    Dense(1 => HIDDEN_UNITS, tanh),
    Dense(HIDDEN_UNITS => HIDDEN_UNITS, tanh; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32),
    Dense(HIDDEN_UNITS => HIDDEN_UNITS, tanh; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32),
    Dense(HIDDEN_UNITS => HIDDEN_UNITS, tanh; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32),
    Dense(HIDDEN_UNITS => 1)
)

# Model initialization
params, states = Lux.setup(rng, model)  # Initial parameters and states
params = ComponentArray(params) |> DEVICE_GPU  # Move parameters to the device
states = states |> DEVICE_GPU  # Move states to the device
smodel = StatefulLuxLayer{true}(model, params, states)  # Stateful model layer

# Test the neural network (initial forward pass)
smodel(input_data, params)

# Callback function
# Monitors training progress
function callback(state, l)
    state.iter % 100 == 1 && @printf "Iteration: %5d, Loss: %.6e\n" state.iter l
    return l < 1e-8  # Stop if loss is sufficiently small
end

# Loss function definition
# Uses adjoint method for gradient computation
# Definir el valor de ε1
εmachine = eps(Float32)
ε2 = 1e-4

f(input_data) = smodel(input_data, params)
f(input_data)
MSELoss()(f(input_data), true_output)

f(input_data) = smodel(input_data, params)
ddu(input_data)=Float32.((f(input_data .+ ε2) .- 2f0 .* f(input_data) .+ f(input_data .- ε2)) / (ε2^2))#Float32.((f(input_data .+ ε2) .- 2f0 .* f(input_data) .+ f(input_data .- ε2)) / (ε2^2))
ddu(input_data)
ε2
res_eq = ddu(input_data)
loss_eq = mean((res_eq-true_output).^2)  

function loss_adjoint(params, (input_data, true_output))
    #pred = smodel(input_data, params)  # Network prediction
    f(input_data) = smodel(input_data, params)
    ddu(input_data)=Float32.((f(input_data .+ ε2) .- 2f0 .* f(input_data) .+ f(input_data .- ε2)) / (ε2^2))
    res_eq = ddu(input_data)
    loss_eq = mean((res_eq-true_output).^2)  
    return loss_eq#MSELoss()(pred, true_output)  # Mean squared error loss
end

# Define the optimization problem
opt_func = OptimizationFunction(loss_adjoint, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, params, data)
epochs = 500  # Maximum number of iterations

# Train using the Adam optimizer
res_adam = solve(opt_prob, Optimisers.Adam(0.001), callback=callback, maxiters=epochs)

# Redefine the optimization problem with updated parameters
opt_prob = OptimizationProblem(opt_func, res_adam.u, data)

# Further training using the LBFGS optimizer
res_lbfgs = solve(opt_prob, LBFGS(); callback, maxiters=epochs)

# Make predictions with the optimized model
pred = smodel(input_data, res_lbfgs.u)


# Compute the L2 error
error_ = LinearAlgebra.norm(Array(pred) .- Array(true_output), 2)
@printf "Error: %.5g\n" error_

# Visualization of results
plot(Array(input_data)', Array(true_output)', label="Target Function")
plot!(Array(input_data)', Array(pred)', label="Neural Network Output")
