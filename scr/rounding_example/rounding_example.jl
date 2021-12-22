cd("/Users/frankvandermeulen/.julia/dev/WI4455/")
#wd = @__DIR__

using Plots
using Distributions
using Random 

# generate data from model
h(x) =  x<1 ? x : ceil(x)

θtrue = 4.4
n = 100
Random.seed!(13) # fix RNG
x = rand(Uniform(0,θtrue), n)
y = h.(x)


histogram(y, label="observations"; normalize=true)

mean(y.<1)
mean(y.==2)

# plot expection of a draw versus θ
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



##################### maximum likelihood estimation #####################

function ψ(x)
    if isinteger(x)
        return(1.0)
    else 
        return(x - floor(x))
    end
end

""" 
    loglik(θ, y::Number)

    returns loglikelihood for observation y at θ
"""
function loglik(θ, y::Number)
    ll = 0.0
    if y < min(1.0, θ) || (y <= ceil(θ)-1.0)
        ll = -log(θ)
    elseif y == ceil(θ)
        ll = log(ψ(θ)) - log(θ) 
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
loglik(maxy-1.0, y)
loglik(maxy-0.999, y)
loglik(maxy, y)

θgrid = range(2.0, maxy+3, length = 10000)
out = ll(y).(θgrid)
plot(θgrid, out, label="" ) # maximum should be attained in (maxy-1, maxy)

θmle = θgrid[argmax(out)]
##################### Bayesian estimation #####################


update_x(y, θ) =   y<1 ? y :  rand(Uniform(y-1, min(y, θ)))

update_x(θ) = (y) -> update_x(y, θ)

data_augmentation = function(y, prior; ITER=20_000)
    α, θmin = params(prior)
    αp = α + length(y) 
    θ = ceil(maximum(y))
    θs = [θ]  # save θ values
    for i ∈ 1:ITER
        x_ = update_x(θ).(y)
        M = maximum(x_)
        θ = rand( Pareto(αp, max(M,θmin)) )
        push!(θs, θ)
    end
   θs
end


# prior specification 
θmin = 0.1
prior = Pareto(2.0, θmin)
print(mean(prior))
θs = data_augmentation(y, prior)
plot(θs)

# remove  burnin (assess visually)
BI = 1_000
histogram(θs[BI:end])

mean(θs[BI:end]) 