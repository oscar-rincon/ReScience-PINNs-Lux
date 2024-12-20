# Import required packages
using Lux,LuxCUDA          # Neural network definition and training
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
using QuasiMonteCarlo, Distributions # Sampling utilities

# Initial program setup
# Seeding for reproducibility
rng = Random.default_rng()
Random.seed!(rng, 0)

# Device configuration (CPU/GPU)
const DEVICE_CPU = cpu_device()  # CPU device
const DEVICE_GPU = gpu_device()

# Sample points using Latin Hypercube Sampling
lb = [-1.0, -1.0]
ub = [1.0, 1.0]
xy_p = QuasiMonteCarlo.sample(1000, lb, ub, LatinHypercubeSample()) |> DEVICE_GPU

@printf "Generating sample points using Latin Hypercube Sampling\n"

bc_len = 20
x_bc = rand(Float32, bc_len) * 2 .- 1
y_bc = rand(Float32, bc_len) * 2 .- 1
xy_bc = hcat(
    stack((x_bc, zeros(Float32, bc_len).-1.0f0); dims=1),
    stack((x_bc, zeros(Float32, bc_len).+1.0f0); dims=1),
    stack((zeros(Float32, bc_len).-1.0f0, y_bc); dims=1),
    stack((zeros(Float32, bc_len).+1.0f0, y_bc); dims=1)
)

# Separar las coordenadas x e y
x_p = Array(xy_p[1, :])
y_p = Array(xy_p[2, :])

# Separar las coordenadas x e y de los puntos de contorno
x_bc = xy_bc[1, :]
y_bc = xy_bc[2, :]

# Graficar los puntos de muestreo
scatter(x_p, y_p, title="Latin Hypercube Sampling", xlabel="x", ylabel="y", legend=false)
scatter!(x_bc, y_bc, color=:red, legend=false)

# Define the analytic function
function analytical_solution(x, y)
    return @. sin(π * x) * cos(π * y)
end

analytical_solution(xy) = analytical_solution(xy[1, :], xy[2, :])

# Compute the true output
true_output_p = reshape(analytical_solution(xy_p), 1, :)
true_output_bc = reshape(analytical_solution(xy_bc), 1, :)
data = (xy_p, xy_bc, true_output_p, true_output_bc) |> DEVICE_GPU  # Move data to the selected device

# Neural network definition
# Architecture: Network with multiple dense layers and tanh activation
const HIDDEN_UNITS = 100
model = Chain(
    Dense(2 => HIDDEN_UNITS, tanh),
    Dense(HIDDEN_UNITS => HIDDEN_UNITS, tanh; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32),
    Dense(HIDDEN_UNITS => HIDDEN_UNITS, tanh; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32),
    Dense(HIDDEN_UNITS => HIDDEN_UNITS, tanh; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32),
    Dense(HIDDEN_UNITS => 1)
)

# Model initialization
parameters, states = Lux.setup(rng, model)  # Initial parameters and states
parameters = ComponentArray(parameters) |> DEVICE_GPU  # Move parameters to the device
states = states |> DEVICE_GPU  # Move states to the device
smodel = StatefulLuxLayer{true}(model, parameters, states)  # Stateful model layer

# Test the neural network (initial forward pass)
smodel(xy_p, parameters)

# Callback function
# Monitors training progress
function callback(state, l)
    state.iter % 100 == 1 && @printf "Iteration: %5d, Loss: %.6e\n" state.iter l
    return l < 1e-8  # Stop if loss is sufficiently small
end

# Loss function definition
# Uses adjoint method for gradient computation
function loss_adjoint(parameters, (xy_p, xy_bc, true_output_p, true_output_bc))
    pred_p = smodel(xy_p, parameters)  # Network prediction
    pred_bc = smodel(xy_bc, parameters)  # Network prediction
    loss_p = MSELoss()(pred_p, true_output_p) # Mean squared error loss
    loss_bc = MSELoss()(pred_bc, true_output_bc) # Mean squared error loss
    loss = loss_p + loss_bc
    return loss
end

# Define the optimization problem
opt_func = OptimizationFunction(loss_adjoint, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, parameters, data)
epochs = 5_000  # Maximum number of iterations

@printf "Training the neural network using the Adam optimizer\n"

# Train using the Adam optimizer
res_adam = solve(opt_prob, Optimisers.Adam(0.001), callback=callback, maxiters=epochs)

# Redefine the optimization problem with updated parameters
opt_prob = OptimizationProblem(opt_func, res_adam.u, data)

# Further training using the LBFGS optimizer
res_lbfgs = solve(opt_prob, LBFGS(); callback, maxiters=epochs)

# Make predictions with the optimized model
# Input range and target output
grid = range(-1.0f0, 1.0f0; length=100)
xy = stack([[elem...] for elem in vec(collect(Iterators.product(grid, grid)))])|> DEVICE_GPU
pred = smodel(xy, res_lbfgs.u)
true_output = Array(reshape(analytical_solution(xy), 1, :))
pred = Array(pred)

# Compute the L2 error
error_ = LinearAlgebra.norm(pred .- true_output, 2)
@printf "Error: %.5g\n" error_
heatmap(grid, grid, reshape(pred .-  Array(true_output), 100, 100), title="Predicted Solution", xlabel="x", ylabel="y", c=:viridis)
 