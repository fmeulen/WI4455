using Distributions
using Plots

# prior parameters
a = 0.1
b = 0.1
Πp = Pareto(a,b)

# data generating distribution
θₒ = 2.0
Pₒ = Uniform(0.0, θₒ)
n = 20

# sample data
x = rand(Pₒ,n)

# posterior distribution Π
b̄ =  b + maximum(x)
Π = Pareto(a+n, b̄)

p = plot(θ -> pdf(Πp,θ), 0, b̄+2,label="prior")
plot!(p, θ -> pdf(Π,θ), 0, b̄+2, label="posterior")

# Approximate the predictive distribution by sampling:
B = 100_000 # Monte-Carlo sample size
out = zeros(B)
for i ∈ 1:B
    θ = rand(Π)
    out[i] = rand(Uniform(0.0, θ))
end

histogram(out, label="predictive distribution")
