library(tidyverse)
library(GGally)
theme_set(theme_bw(base_size = 12))
#  Exercise 4.20

# generate some "true" thetas
n <- 100
theta0 <- runif(n,0,1000)

# generate data
x<- rep(0,n)
for (i in 1:n) x[i] <- runif(1,0,theta0[i])

# compute estiamtes
mle <- x
ebayes <- x+mean(x)
lambda_hyp <- 0.010 # hyperpar in prior
bayes <- x+1/lambda_hyp

ss <- function(x) { sum(x^2) }

mse.mle <- ss(mle-theta0)
mse.ebayes <- ss(ebayes-theta0)
mse.bayes <- ss(bayes-theta0)

mse.mle
mse.ebayes
mse.bayes

# visualise results
d <- data.frame(i=rep(1:n,3),value=c(mle-theta0,ebayes-theta0,bayes-theta0),type=rep(c('mle','ebayes','bayes'),each=n))
ggplot(data=d, aes(x=i,y=value,colour=type)) + geom_point()+facet_wrap(~type)+
  theme(legend.position="bottom")+ geom_hline(yintercept=0,size=1.1) + 
  ggtitle("estimate-theta0") + ylab("")

cat('1/(sample average) equals:', 1/mean(x))


###### fully bayesian with Gibbs sampling

# specify hyperpars of prior
alpha = 0.1
beta = 2

IT <- 10000
lambda <- rep(0,IT)
theta <- matrix(0,IT,n) # each row contains an MCMC iteration
theta[1,]  <- x  # initialise at mle
for (it in 2:IT)
{  lambda[it] = rgamma(1,shape=2*n+alpha, rate= beta+sum(theta[it-1,]))
   theta[it,] = x + rexp(1,rate=lambda[it])
}

# postprocessing
bayeshier <-   as_tibble(theta) %>% filter(row_number() > IT/2) %>% 
  summarise_all(mean)
bayeshier <- as.numeric(bayeshier[1,])
mse.bayeshier <- ss(bayeshier-theta0)

mse.mle
mse.ebayes
mse.bayes
mse.bayeshier


dd <- data.frame(i=rep(1:n,4),value=
                   c(mle-theta0,ebayes-theta0,bayes-theta0,bayeshier-theta0),
                 type=rep(c('mle','ebayes','bayes','hier.bayes'),each=n)) 
            

ggplot(data=dd, aes(x=i,y=value,colour=type)) + geom_point()+facet_wrap(~type,nrow=1)+
  theme(legend.position="none")+ geom_hline(yintercept=0,size=1) +   ggtitle("estimate-theta0") + ylab("")

# only show first 10 theta[i]
dd %>% filter(i<=10) %>% ggplot(aes(x=i,y=value,colour=type)) + geom_jitter(size=1.5,width=0.1, height=0)+
   geom_hline(yintercept=0,size=1) +   ggtitle("estimate-theta0 (only first 10 theta[i])") +
  ylab("")+ scale_x_continuous(breaks = seq(1, 10, by=1))


# verify chain by inspecting traceplots
diterates <- data.frame(iterate=1:IT,theta2=theta[,2],theta20=theta[,20],lambda=lambda)
diterates %>% filter(row_number() %in% seq(1,IT,by=10)) %>% gather(key=par,value=y,theta2,theta20,lambda) %>%
     ggplot() + geom_path(aes(x=iterate,y=y)) + facet_grid(par~.,scales='free')+ ggtitle("traceplots for a few parameters")
 

