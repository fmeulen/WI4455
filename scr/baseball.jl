using DataFrames
using RCall
using Statistics
using Distributions
using LinearAlgebra

workdir = @__DIR__
println(workdir)
cd(workdir)

# data
players = ["McGwire", "Sosa", "Griffey", "Castilla", "Gonzalez", "Galaragga",  "Palmeiro",
"Vaughn"," Bonds", "Bagwell", "Piazza", "θome", "θomas","T. Martinez", "Walker", "Burks", "Buhner"]

Y = [7,9,4,7,3,6,2,10,2,2,4,3,2,5,3,2,6]
n = [58,59,74,84,69,63,60,54,53,60,66,66,72,64,42,38,58]
AB = [509,643,633,645,606,555,619,609,552,540,561,440,585,531,454,504,244]
HR = [70,66,56,46,45,44,43,40,37,34,32,30,29,28,23,21,15]

addfrank = true # in preseason frank did extremely well, though only 3 times at bat, each time this resulted in a homerun
# during the season he was 500 times at bat, with 5 homeruns

if addfrank
  push!(players,"Frank")
  push!(Y,3)
  push!(n,3)
  push!(AB, 500)
  push!(HR,5)
end
baseball = DataFrame(players=players,PS_AB=n,PS_HR=Y,S_AB=AB,S_HR=HR)

# functions for updating each element of μ
ψ(x) = 1.0/(1.0 + exp(-x))
#logtargetμ(μ, y, n, θ, τsq) = y*μ -  0.5*((μ-θ)^2)/τsq - n*log(1+exp(μ))
logtargetμ(μ, y, n, θ, τsq) = logpdf(Binomial(n,ψ(μ)),y) + logpdf(Normal(θ,sqrt(τsq)), μ)




function updateμ(y, n, μ, θ, τsq, tunePar)
  μᵒ = μ + tunePar * randn()
  logA = logtargetμ(μᵒ, y, n, θ, τsq) - logtargetμ(μ, y, n, θ, τsq)
  if log(rand()) < logA
    acc = 1
    μ = μᵒ
  else
    acc = 0
  end
  μ, acc
end

N = length(Y)
IT = 2000  # number of iterations
BI = div(IT,2)


tunePar = 1.0 # tuning par for MH step updating μ (sd of normal distr)

# prior hyperpars
α =  0.001
β = 0.001

# save iterates in matrix and vectors
μ = zeros(IT,N)
θ = zeros(IT)
τsq = zeros(IT)

# initialise
μ[1,:] = 5.0 * randn(N)
θ[1] = randn()
τsq[1] = 2.0
acc = Int[]
acc_ = 0

# Gibbs sampler:
for it in 2:IT
  for i in 1:N
    μ[it,i], acc_ = updateμ(Y[i],n[i],μ[it-1,i],θ[it-1],τsq[it-1],tunePar)
    push!(acc, acc_)
  end

  θ[it] = rand(Normal(mean(μ[it,:]),sqrt(τsq[it-1]/N)))

  shape = .5*N + α
  rate = β + .5 * norm(μ[it,:] .- θ[it])^2
  τsq[it] = 1/rand(Gamma(shape,1.0/rate))
end
println("acceptance percentange MH-steps: ", round(100*sum(acc)/((IT-1)*N); digits=2))

df = DataFrame(hcat(μ,θ,τsq))
names!(df, push!([Symbol("mu$i") for i in 1:N], :theta, :tausq))

@rput df
@rput IT
@rput N
R"""
library(tidyverse)
library(ggplot2)
dfs <- df %>% gather(key="parameter", value="value") %>% mutate(it=rep(1:IT,N+2))
p <- dfs %>% ggplot() + geom_path(aes(x=it, y=value)) + facet_wrap(~parameter, scales='free') + theme_light()
pdf("baseball_iterates.pdf",width=10, height=7)
show(p)
dev.off()
"""

# add p = ψ(μ)
for i in 1:N
  df[Symbol("p$i")]= ψ.(df[Symbol("mu$i")])
end
# computate posterior means
df_= df[BI:IT,:]

postmean = [mean(col) for col in eachcol(df_)]

# add postmean and mle to dataframe
baseball[:bayes] = postmean[end-N+1:end]
baseball[:mle] = baseball[:PS_HR]./baseball[:PS_AB] # mle equals empirical fraction)

# compute mean of predicted homeruns during season and add to dataframe
baseball[:S_HR_bayes] = baseball[:S_AB] .* baseball[:bayes]
baseball[:S_HR_mle] = baseball[:S_AB] .* baseball[:mle]

@rput baseball
R"""
library(gridExtra)
p1 <- ggplot(baseball, aes(x=players)) +
  geom_point(aes(y = bayes, shape="Bayes",colour='Bayes'),size=2.5) +
  geom_point(aes(y = mle, shape="mle",colour='mle'),size=2.5)+
  theme_light() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5),legend.position='none') +
    ylab('probability') +ggtitle("Preseason data")+xlab("")
p1

p2 = ggplot(baseball, aes(x=players)) +
     geom_point(aes(y = S_HR_bayes, shape ="Bayes", colour="Bayes"),size=2.5)+
  geom_point(aes(y = S_HR_mle, shape = "mle", colour="mle"),size=2.5)+
  geom_point(aes(y = S_HR),shape=8,size=2.5)+theme_light()+
    theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5),legend.position='bottom') +
    ylab('nr of homeruns') +ggtitle("Comparison predicted (using preseason data) and observed (season data)")
pdf('output-baseball_combined.pdf',width=8,height=8)
grid.arrange(p1,p2,ncol=1)
dev.off()
"""

# compare performance
performance_bayes = norm(baseball[:S_HR] - baseball[:S_HR_bayes])^2
performance_mle = norm(baseball[:S_HR] - baseball[:S_HR_mle])^2
println("sum-squared prediction error Bayes equals: ", round(performance_bayes,digits=0))
println("sum-squared prediction error MLE equals: ", round(performance_mle;digits=0))
