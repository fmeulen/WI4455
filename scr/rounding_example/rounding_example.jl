using Plots
using Distributions

θ = 4.5
n = 10000
x = rand(Uniform(0,θ), n)
y = map(x -> (x<1 ? x : Int(ceil(x))), x)

histogram(y, label="observations"; normalize=true)

mean(y.<1)
mean(y.==2)

expec = function(θ)
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

