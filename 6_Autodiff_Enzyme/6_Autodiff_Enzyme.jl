using Enzyme, Plots
 
using EnzymeCore

f(x) = x.*x
x1 = reshape(collect(range(-1f0, 1f0, length=100)), 1, :)
dx1 = ones(size(x1))
x = -5.0:0.1:5.0
y = f(reshape(collect(range(-5f0, 5f0, length=100)), 1, :)) 

x = Array([3.0, 4.0, 5.0])
dx = ones(size(x))
EnzymeCore.autodiff(ReverseSplitWithPrimal, f, Duplicated(x, dx))[1]
 

u =  zeros(length(x))
for i in eachindex(x)
    u[i] = Enzyme.autodiff(Enzyme.Reverse, f, Active(x[i]))[1][1]  
end

plot!(u)
plot!(y')