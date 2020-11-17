######## Empirical Bayes example

N.MC <- 10^4        # Monte-Carlo sample size
theta=0.0
tau2 <- 10          # prior variance

empBayes <- function(x)
{
  s <- sum(x)
  n <- length(x)
  tau2 <- max(0,(s^2-n)/n)
  tau2*s/(1+n*tau2)
}

Bayes <- function(x,tau2)  # tau2 is prior variance
{
  s <- sum(x)
  n <- length(x)
  tau2*s/(1+n*tau2)
}

############# Monte-Carlo study (generate data, and compute all three estimators)

n <- 100            # sample size
ebayes <- numeric(N.MC)
bayes <- numeric(N.MC)
mle <- numeric(N.MC)
for (i in 1:N.MC)
{
  x <- rnorm(n,theta)
  ebayes[i] <- empBayes(x)
  bayes[i] <- Bayes(x,tau2)
  mle[i] <- mean(x)
}


 visualisation
library(ggplot2)
theme_set(theme_minimal())

d=data.frame(type=rep(c('Mle','Bayes','Emp. Bayes'),each=N.MC),
             estimate=c(mle,bayes,ebayes))
titel <- paste('n=',n,', tau^2=',tau2,', theta=',theta, sep="")
fname <- paste('comparison_boxplot','n=',n,'tau^2=',tau2,'theta=',theta,'.pdf',sep="_")
pdf(fname,width=6,height=3.5)
ggplot(data=d,aes(x=type,y=estimate,colour=type))+geom_boxplot()+
    theme(legend.position="bottom")
dev.off()

fname <- paste('comparison_histogram','n=',n,'tau^2=',tau2,'theta=',theta,'.pdf',sep="_")
pdf(fname,width=6,height=3.5)
ggplot(data=d,aes(x=estimate,colour=type))+
  geom_histogram(aes(fill=type))+facet_wrap(~type)+ #+ggtitle(titel)+
   theme(legend.position="none")
dev.off()

fname <- paste('comparison_histogram_free','n=',n,'tau^2=',tau2,'theta=',theta,'.pdf',sep="_")
pdf(fname,width=6,height=3.5)
ggplot(data=d,aes(x=estimate,colour=type))+
  geom_histogram(aes(fill=type))+facet_wrap(~type,scales = "free_y")+#+ggtitle(titel)+
  theme(legend.position="none")
dev.off()

