using TaylorDiff, Plots 
using Lux,LuxCUDA
 
devices()

DEVICE_GPU = CUDADevice()
DEVICE_CPU = cpu_device()

x = collect(range(0f0, 5f0, length=200))
x1 = reshape(x, 1, :) |> DEVICE_GPU 

# Define the function
f(x) = sin.(x)  


derivative = TaylorDiff.derivative.(f, x1, CUDA.fill(1.0f0, size(x1)), Val(2))

# Input range and target output
grid = range(-1.0f0, 1.0f0; length=100)
xy = stack([[elem...] for elem in vec(collect(Iterators.product(grid, grid)))])

# Define the analytic function
function analytical_solution(x, y)
    return @. sin(π * x) * cos(π * x)
end

analytical_solution(xy) = analytical_solution(xy[1, :], xy[2, :])

xy
analytical_solution(xy)

true_output = reshape(analytical_solution(xy), 100, 100)
heatmap(grid, grid, true_output)



direction = Float32.(ones(size(xy)))
direction[:, 1] .= 0.0
direction
derivative_x = TaylorDiff.derivative(analytical_solution, xy, direction, Val(1))

derivative_output = reshape(derivative_x, 100, 100)
heatmap(grid, grid, derivative_output)
