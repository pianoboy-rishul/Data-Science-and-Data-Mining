##...RISHUL GHOSH...##
##...N230...##

#______________________________________________________#
##STEP-1: Importing data set and factorization
dataset=read.csv("Social_Network_Ads.csv")
View(dataset)
dataset=dataset[3:5]
View(dataset)
dataset$Purchased=factor(dataset$Purchase,levels=c(0,1))
View(dataset)

#______________________________________________________#
##STEP-2: Splitting data set
library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased, SplitRatio=0.75)
training_set=subset(dataset, split==TRUE)
test_set=subset(dataset, split==FALSE)
View(training_set)
View(test_set)

#______________________________________________________#
##STEP-3: Feature Scaling
training_set[-3]=scale(training_set[-3])
test_set[-3]=scale(test_set[-3])
View(training_set)
View(test_set)

#______________________________________________________#
##STEP-4: Fitting Naive-Bayes to the training data set
library(e1071)
classifier=naiveBayes(x=training_set[-3], 
                      y=training_set$Purchased)
print(classifier)

#______________________________________________________#
##STEP-5: Predicting the test set results
y_pred=predict(classifier, newdata=test_set[-3])
print(y_pred)
print(test_set[3])

#______________________________________________________#
##STEP-6: Making confusion matrix
cm=table(test_set[,3],y_pred)
print(cm)
#CHECKING ITS MODEL ACCURACY
acc=(cm[1,1]+cm[2,2])/100
print(acc)