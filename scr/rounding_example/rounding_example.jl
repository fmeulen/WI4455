using Plots
using Distributions

θ = 4.5
n = 10000
x = rand(Uniform(0,θ), n)
y = map(x -> (x<1 ? x : ceil(x)), x)

histogram(y, label="observations"; normalize=true)

mean(y.<1)
mean(y.==2)

function expec(θ)
    out = 0.5θ
    if θ > 1.0    
        θc = ceil(θ)
        θf = floor(θ)
        out = (0.5+0.5*(θc-1.0)*θc-1.0+(θ-θf)*θc)/θ
    end
    out
end

θg = range(0.01, 3.5, length=100)
plot(θg, expec.(θg), label="")

# maximum likelihood estimation

""" 
    loglik(θ, y::Number)

    returns loglikelihood for observation y at θ
"""
function loglik(θ, y::Number)
    ll = 0.0
    if y < min(1.0, θ) || (y <= ceil(θ)-1.0)
        ll = -log(θ)
    elseif y == ceil(θ)
        ll = log(1.0 - floor(θ)/θ)
    else
        ll = -Inf64
    end
    ll
end

""" 
    loglik(θ, y::Vector)

    returns loglikelihood for vector of observations y at θ
"""
function loglik(θ, y::Vector)
    out = 0.0
    for i ∈ eachindex(y)
        out += loglik(θ, y[i])
    end
    out
end

ll(y) = θ -> loglik(θ,y)  # define loglik as function of θ


maxy = maximum(y)
loglik(4.0, y)
loglik(4.001, y)
loglik(4.99, y)

# note the irregularity of the loglik
loglik(maxy, y)
ϵ = 0.001
loglik(maxy + ϵ, y)
loglik(maxy - ϵ, y)

θs = range(2.0, 6.0, length = 10000)
out = ll(y).(θs)
plot(θs, out, label="" ) # can't see irregularity at θ = maxy from the fig, maximum should be attained in (maxy-1, maxy)
