using Lux, Random, TaylorDiff, Plots, Optimisers, Zygote, ComponentArrays, Printf, LinearAlgebra
using Optimisers
using CUDA
gdev = CUDADevice()

# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

# Device setup
const DEVICE_CPU = cpu_device()
#const DEVICE_GPU = CUDADevice()

# Data preparation
const INPUT_RANGE = range(-1f0, 1f0, length=100)
input_data = reshape(collect(INPUT_RANGE), 1, :) |> gdev
true_output = reshape(Float32.(sin.(10*input_data)), 1, :)

# Reference function
target_function(x) = sin.(10*x)

# Neural network definition
const HIDDEN_UNITS = 100

model = Chain(
    Dense(1 => HIDDEN_UNITS, tanh),
    Dense(HIDDEN_UNITS => HIDDEN_UNITS, tanh; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32),
    Dense(HIDDEN_UNITS => HIDDEN_UNITS, tanh; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32),
    Dense(HIDDEN_UNITS => HIDDEN_UNITS, tanh; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32),
    Dense(HIDDEN_UNITS => 1)
)

# Model initialization
parameters, states = Lux.setup(rng, model) |> DEVICE_CPU
parameters = parameters |> ComponentArray
neural_network = StatefulLuxLayer{true}(model, parameters, states) |> gdev 

# Neural network predictions
predicted_output = neural_network(input_data|> DEVICE_CPU)

function neural_network_cpu(input_data)
    return Array(neural_network(input_data)) 
end

predicted_output =neural_network_cpu(Array(input_data))

# Plot the neural network output vs target function
plot(Array(input_data)', Array(true_output)', label="Target Function")
plot!(Array(input_data)', Array(predicted_output)', label="Neural Network Output")

# Gradient calculations
gradient_zygote = only(Zygote.gradient(sum ∘ neural_network, input_data))
gradient_taylor = TaylorDiff.derivative(neural_network|> DEVICE_CPU,input_data|> DEVICE_CPU, Float32.(ones(size(input_data|> DEVICE_CPU))), Val(1))
gradient_target = TaylorDiff.derivative(target_function, Array(input_data), Float32.(ones(size(input_data))), Val(1))

# Plot gradients
plot(input_data', gradient_zygote', label="Gradient (Zygote)")
plot!(input_data', gradient_taylor', label="Gradient (TaylorDiff)")
plot!(input_data', gradient_target', label="Gradient (Target Function)")

# Initialize optimizer
const LEARNING_RATE = 0.001f0
optimizer = Optimisers.Adam(LEARNING_RATE)
vjp_rule = AutoZygote()
train_state = Training.TrainState(model, parameters, states, optimizer)

# Training loop
function train_model!(state::Training.TrainState, vjp_rule, data, epochs)
    data = data .|> gdev
    for epoch in 1:epochs
        _, loss, _, state = Training.single_train_step!(vjp_rule, loss_function, data, state)
        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end
    return state
end

const loss_function = MSELoss()
train_state = train_model!(train_state, vjp_rule, (input_data, true_output), 5_000)

# Update the neural network with trained parameters
neural_network = StatefulLuxLayer{true}(train_state.model, train_state.parameters, train_state.states)
predicted_output = neural_network(input_data)

# Calcualte the L2 error
error = LinearAlgebra.norm(predicted_output.- true_output, 2)
@printf "Error: %.5g\n" error


# Plot the neural network output vs target function
plot(input_data', true_output', label="Target Function")
plot!(input_data', predicted_output', label="Neural Network Output")

# Recalculate gradients
gradient_zygote = only(Zygote.gradient(sum ∘ neural_network, input_data))
gradient_taylor = TaylorDiff.derivative(neural_network, input_data, Float32.(ones(size(input_data))), Val(1))
gradient_target = TaylorDiff.derivative(target_function, input_data, Float32.(ones(size(input_data))), Val(1))


# Calcualte the L2 error
error_zygote_1 = LinearAlgebra.norm(gradient_zygote.- gradient_target,2)
@printf "Error Zygote 1: %.5g\n" error_zygote_1
error_taylor_1 = LinearAlgebra.norm(gradient_taylor.- gradient_target,2)
@printf "Error Taylor 1: %.5g\n" error_taylor_1


# Plot gradients comparison
plot(input_data', gradient_target', label="Gradient (Target Function)")
plot!(input_data', gradient_zygote', label="Gradient (Zygote)")
plot!(input_data', gradient_taylor', label="Gradient (TaylorDiff)")

gradient_taylor_2 = TaylorDiff.derivative(neural_network, input_data, Float32.(ones(size(input_data))), Val(2))
gradient_target_2 = TaylorDiff.derivative(target_function, input_data, Float32.(ones(size(input_data))), Val(2))

# Calcualte the L2 error
error_taylor_2 = LinearAlgebra.norm(gradient_taylor_2.- gradient_target_2,2)
@printf "Error Taylor 2: %.5g\n" error_taylor_2

# Plot gradients comparison
plot(input_data', gradient_target', label="Gradient (Target Function)")
plot!(input_data', gradient_taylor', label="Gradient (TaylorDiff)")
