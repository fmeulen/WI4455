# simple stylized medical experiment as discussed in Berger (The case for Objective Bayesian Analysis)
using Random
using RCall
using Distributions
using DataFrames

Random.seed!(3)
cd(@__DIR__)

θtrue = [0.01, 0.9, 0.05] # true probs of 'having the disease', 'testing positive | having the disease', 'testing positive | not having the disease'
ψ(θ) = θ[1]*θ[2]/(θ[1]*θ[2] + (1-θ[1])* θ[3])
ψtrue = ψ(θtrue)
n = [17_000, 10, 100]

# simulate data
X = [rand(Binomial(n[i], θtrue[i])) for i in eachindex(n)]

"""
    randpost = function(n, X; a = 1.0, b = 1.0)

Simulate from posterior probability of θ = prob(disease | positive test)
The prior on each element of θ is Beta(a,b)
"""
function randpost(n, X; a = 1.0, b = 1.0)
    θpost = [rand(Beta(X[i]+a, n[i]-X[i]+b)) for i in eachindex(n)]
    push!(θpost, ψ(θpost))
end

ps = [randpost(n,X) for _ in 1:10_000]

# visualisation

ec(X,i) = map(x->x[i], X)
df = DataFrame(theta0=ec(ps,1),theta1=ec(ps,2),theta2=ec(ps,3),psi=ec(ps,4))
@rput df
R"""
library(tidyverse)
p1 <- df %>% gather(key='parameter', value=x) %>%
    ggplot(aes(x=x)) + geom_histogram(bins=75,fill="orange",colour="white",)+
    facet_wrap(~parameter,scales="free") + theme_light() + xlab("")
pdf("medicalout.pdf", width = 7, height = 4)
    show(p1)
dev.off()
"""
