library(caret)
library(ggplot2)
library(MASS)
library(car)
library(mlogit)
library(sqldf)
library(Hmisc)

data<-train_1_
str(data)

# Creation of Dummy Variables
data$SibSp <- as.factor(data$SibSp)
data$Parch <- as.factor(data$Parch)
data$Embarked <- as.factor(data$Embarked)


# Summary of data
summary(data)


# Deleting column cabin [because column cabin consists of so many misiing values.]
data = data[-which(names(data)=="Cabin")]



# Treatment of missing values
sapply(data, function(x) sum(is.na(x)))
data <- na.omit(data)
sapply(data, function(x) sum(is.na(x)))




# Treatment of outliers
# Column Fare is having outliers.
boxplot(data$Fare)
quantile(data$Fare, c(0,0.05,0.1,0.25,0.5,0.75,0.90,0.95,0.97,0.98,0.985,0.99,0.995,1))
data1 = data[data$Fare<60,]
boxplot(data1$Fare)
data<-data1

# Column Age is having outliers.
boxplot(data$Age)
quantile(data$Age, c(0,0.05,0.1,0.25,0.5,0.75,0.90,0.95,0.97,0.98,0.985,0.99,0.995,1))
data1 = data[data$Age<70,]
boxplot(data1$Age)
data<-data1

names(data)



# Creating Logistic Model 
# Iteration 1 :
model <- glm(Survived~ PassengerId + Pclass  + Name + Sex + Age + SibSp +      
               Parch + Ticket + Fare + Embarked , data=data, family=binomial())
summary(model)

# Iteration 2 :
model <- glm(Survived~ PassengerId + Pclass  + Sex + Age + SibSp +      
               Parch + Ticket + Fare  + Embarked , data=data, family=binomial())
summary(model)

# Iteration 3 :
model <- glm(Survived~ PassengerId + Pclass  + Sex + Age + SibSp +      
               Parch  + Fare  + Embarked , data=data, family=binomial())
summary(model)

# Iteration 4 :
model <- glm(Survived~  Pclass  + Sex + Age + SibSp +      
               Parch + Embarked , data=data, family=binomial())
summary(model)

# Iteration 5 :
model <- glm(Survived~ Pclass  + Sex + Age + SibSp +      
               Parch  , data=data, family=binomial())
summary(model)

# Iteration 6 :
model <- glm(Survived~ Pclass  + Sex + Age + I(SibSp=='3') + I(SibSp=='4') +     
               Parch , data=data, family=binomial())
summary(model)

# Iteration 7 :
# FINAL MODEL
model <- glm(Survived~ Pclass  + Sex + Age + I(SibSp=='3') + I(SibSp=='4')     
             , data=data, family=binomial())
summary(model)




# Checking Multicollinearity 
vif(model)



# R square (nagelkarke)
modelChi <- model$null.deviance - model$deviance
#Finding the degree of freedom for Null model and model with variables
chidf <- model$df.null - model$df.residual
chisq.prob <- 1 - pchisq(modelChi, chidf)
R2.hl<-modelChi/model$null.deviance
R.cs <- 1 - exp ((model$deviance - model$null.deviance) /nrow(data))
R.n <- R.cs /(1-(exp(-(model$null.deviance/(nrow(data))))))
R.n 



# Predicted Probabilities
prediction <- predict(model,newdata = data,type="response")
library(pROC)
rocCurve   <- roc(response = data$Survived, predictor = prediction, 
                  levels = rev(levels(data$Survived)))
data$Survived <- as.factor(data$Survived)
#Metrics - Fit Statistics
predclass <-ifelse(prediction>coords(rocCurve,"best")[1],1,0)
Confusion <- table(Predicted = predclass,Actual = data$Survived)
AccuracyRate <- sum(diag(Confusion))/sum(Confusion)
Gini <-2*auc(rocCurve)-1
AUCmetric <- data.frame(c(coords(rocCurve,"best"),AUC=auc(rocCurve),AccuracyRate=AccuracyRate,Gini=Gini))
AUCmetric <- data.frame(rownames(AUCmetric),AUCmetric)
rownames(AUCmetric) <-NULL
names(AUCmetric) <- c("Metric","Values")
AUCmetric



# Confusion Matrix
Confusion 
plot(rocCurve)



#KS Statistics Calculation
data$m1.yhat <- predict(model, data, type = "response")
library(ROCR)
m1.scores <- prediction(data$m1.yhat, data$Survived)
plot(performance(m1.scores, "tpr", "fpr"), col = "red")
abline(0,1, lty = 8, col = "grey")
m1.perf <- performance(m1.scores, "tpr", "fpr")
ks1.logit <- max(attr(m1.perf, "y.values")[[1]] - (attr(m1.perf, "x.values")[[1]]))
ks1.logit 


library(xlsx)

View(data)
names(data)[13] <- "pred"
View(data)



# Imputing NA values under age column by mean = 30
test$Age[which(is.na(test$Age))]=30
View(test)


# Prediction 
x <- predict(model,test,"response")
x


y<- predict(model,train_1_,"response")
y

predict.glm(model)


getwd()
setwd("C:\\Users\\HP\\Desktop\\muitstudy\\R")
write.csv(x,"test.csv")






