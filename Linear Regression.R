##...RISHUL GHOSH...##
##...N230...##
#_________________________________________________________#

##Implementation of simple linear regression
Height =c(151,174,138,186,136,179,163,152,131)
Weight =c(63,81,56,91,47,57,76,62,48)

library(e1071) # for skewness function
par(mfrow=c(1, 2)) # divide graph area in 2 columns
plot(density(Height), main="Density Plot: Height", ylab="Frequency", sub=paste("Skewness:", round(e1071::skewness(Height), 2)))
polygon(density(Height), col="red")
plot(density(Weight), main="Density Plot: Weight", ylab="Frequency", sub=paste("Skewness:", round(e1071::skewness(Weight), 2))) 
polygon(density(Weight), col="blue")