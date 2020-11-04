using Turing
using Distributions
using Random


model {
  a1 ~ gamma(45,1);
  a2 ~ gamma(45,1);
  a3 ~ gamma(10,1);

  for (n in 1:N) {
    theta[n] ~ dirichlet([a1, a2, a3]'); // prior
    y[n, ] ~ multinomial(theta[n]);      // likelihood
  }
}

@model gdemo(x, y) = begin
 # Assumptions
 σ ~ InverseGamma(2,3)
 μ ~ Normal(0,sqrt(σ))
 # Observations
 x ~ Normal(μ, sqrt(σ))
 y ~ Normal(μ, sqrt(σ))
end
