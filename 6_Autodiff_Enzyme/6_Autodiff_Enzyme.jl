using EnzymeCore
using Printf

f(x) = x*x
res, ∂f_∂x = autodiff(ForwardWithPrimal, f, Duplicated, Duplicated(3.14, 1.0))

# output

(6.28, 9.8596)