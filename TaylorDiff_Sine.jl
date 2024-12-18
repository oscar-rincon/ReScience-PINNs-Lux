using Random, Distributions, Plots, Optimisers, Zygote

# Constants
SEED = 42
HIDDEN_DEPTH = 100
LEARNING_RATE = 0.01
N_EPOCHS = 5000

# Initialize random number generator
rng = MersenneTwister(SEED)

# Initialize weights and biases
uniform_limit = sqrt(2 / (1 + HIDDEN_DEPTH))
W = rand(rng, Uniform(-uniform_limit, +uniform_limit), HIDDEN_DEPTH, 1)
V = rand(rng, Uniform(-uniform_limit, +uniform_limit), 1, HIDDEN_DEPTH)
b = zeros(HIDDEN_DEPTH)

# Bundle parameters
parameters = (; W, V, b)

# Define activation function
sigmoid(x) = 1.0 / (1.0 + exp(-x))

# Define network forward pass
network_forward(x, p) = p.V * sigmoid.(p.W * x .+ p.b)

# Generate input data
x = reshape(collect(range(0.0f0, stop=1.0f0, length=100)), (1, 100))

# Plot initial network output
#plot(network_forward(x, parameters)')

# Define loss function
function loss_forward(p)
    f(x) = network_forward(x, p)
    y_pred = f(x)
    y = sin.(x)
    return sum((y_pred .- y).^2) / size(y, 2)
end

# Initialize optimizer
opt = Adam(LEARNING_RATE)
opt_state = Optimisers.setup(opt, parameters)
loss_history = []

# Training loop
for i in 1:N_EPOCHS
    # Compute loss and gradients
    loss, back = Zygote.pullback(loss_forward, parameters)
    push!(loss_history, loss)
    grad, = back(1.0)
    
    # Update parameters
    opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
    
    # Print loss every 100 epochs
    if i % 100 == 0
        println("Epoch: $i, Loss: $loss")
    end
end

# Plot exact solution
plot(sin.(x)')

# Plot prediction
plot(network_forward(x, parameters)')
