using Turing
using Distributions
using Random
using StatsPlots
using NamedArrays
using CSV
using MCMCChains
using DataFrames

@model function dmn(y)
  counts =  sum(y, dims=2)
  n, d = size(y)
  a1 ~ Gamma(45,1)
  a2 ~ Gamma(45,1)
  a3 ~ Gamma(10,1)
  α = [a1, a2, a3]
  θ = Matrix{Real}(undef, n, d)
  for i ∈ 1:n
      θ[i,:] ~ Dirichlet(α)
      y[i,:] ~ Multinomial(counts[i], θ[i,:])
  end
end

# y = [10 20 40; 3 40 10]  # simple testdata

wd = @__DIR__
cd(wd)
d = CSV.read("approvaldata.csv")
y = Matrix(d)

# use No_U-Turn-Sampler
chns = sample(dmn(y), HMC(0.01, 10), 1000)
iters = DataFrame(chns, [:parameters])


# default plot
#plot(chns)

# chns is of type Chains, type `?Chains` to see what is in and how to extract the info
# for example, the vector of parameters is extracted via
parnames = chns.name_map[:parameters]
# get iterates for par a[1], surprisingly difficult
a1iterates = chns[:a1].value.data[:,:,1][:,1]

# Summary stas can be obtained directly via
out = describe(chns)
#out[1] and out[2] contain summary stats
#println(out[1][:mean])

# get iterates for pars a[1] and a[2]
chns[[:a1,:a2]].value

chns |> display

"""
    extract_iters(chns)

Converts output of Turing sampler into a names array, each row containing iterates for one parameter in the model.
Warning: has only been tested in case of 1 chain.
"""
function extract_iters(chns)
  nr_iters =  size(chns.value.data)[1]
  parnames = chns.name_map[:parameters]
  out = NamedArray{Real}(length(parnames), nr_iters)
  setnames!(out, parnames, 1)
  for naam ∈ parnames
      out[naam,:] =  chns[naam].value.data[:,:,1][:,1]
  end
  out
end

iters = extract_iters(chns)
plot(iters["a2",:], label="a2")
plot(iters["a1",:], label="a1")
plot(iters[14,:])
