library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)
library(ranger)
library(pROC)
library(pdp)
library(ggplot2)
library(gbm)
library(xgboost)
library(adabag)
library(ROCR)
library(lattice)
library(randomForest)

bank <- read.csv("Bank.csv")
bank$duration = NULL;

# Modifying the output column

bank$deposit <- ifelse(bank$deposit == 'yes', 1,0)
bank$deposit <- as.factor(bank$deposit)


# Partition the data into train/test sets
set.seed(1535)
index <- sample(nrow(bank),nrow(bank)*0.70)
train = bank[index,]
test = bank[-index,]


#Fitting the model
bank_rpart <- rpart(formula = deposit ~ ., data = train, method = "class")

#credit.rpart <- rpart(formula = deposit ~ . , data = train, 
                      method = "class", parms = list(loss=matrix(c(0,10,1,0), nrow = 2)))

#Make the best model

bank_rpart <- rpart(
  deposit ~ ., data = train, 
  control = list(cp = 0.001, minbucket = 1, minsplit = 1)
)

plotcp(bank_rpart)

bank_rpart <- rpart(
  deposit ~ ., data = train, 
  control = list(cp = 0.003, minbucket = 1, minsplit = 1)
)


#Plotting the tree
bank_rpart
prp(bank_rpart, extra = 1)

#Prediction on training data

pred.tree1 <- predict(bank_rpart, train, type="class")
table(train$deposit, pred.tree1, dnn=c("Truth","Predicted"))

#Prediction on test data
pred.test.tree <- predict(bank_rpart, test, type="class")
table(test$deposit, pred.test.tree, dnn=c("Truth","Predicted"))

#ROC Curve ,Gain Chart and K-S Chart for test data and AUC values 0.722

bank.test.prob = predict(bank_rpart,test, type="prob")
pred = prediction(bank.test.prob[,2], test$deposit)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]

gain <- performance(pred, "tpr", "rpp")
plot(gain, main = "Gain Chart")

ks=max(attr(perf,'y.values')[[1]]-attr(perf,'x.values')[[1]])
plot(perf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
print(ks); #0.35




#ROC and Gain chart curve and K-S Chart for training data 0.733
bank.train.prob = predict(bank_rpart,train, type="prob")
pred1 = prediction(bank.train.prob[,2], train$deposit)
perf1 = performance(pred1, "tpr", "fpr")
plot(perf1, colorize=TRUE)

slot(performance(pred1, "auc"), "y.values")[[1]]

gain1 <- performance(pred1, "tpr", "rpp")
plot(gain1, main = "Gain Chart")

ks=max(attr(perf1,'y.values')[[1]]-attr(perf1,'x.values')[[1]])
plot(perf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
print(ks);

# Extract tibble of variable importance scores
vip::vi(bank_rpart)

# Construct ggplot2-based variable importance plot
vip::vip(bank_rpart, num_features = 10)



##Random Forest

#Calculate the number of trees required- Optimal of 300 trees chosen

oob_error <- function(trees) {
  fit <- ranger(
    formula    = deposit ~ ., 
    data       = trn, 
    num.trees  = trees,
    mtry       = floor(16 / 3),
    respect.unordered.factors = 'order',
    verbose    = FALSE,
    seed       = 1535
  )
  sqrt(fit$prediction.error)
}
# tuning grid
trees <- seq(10, 1000, by = 20)
(rmse <- trees %>% purrr::map_dbl(oob_error))

ggplot(data.frame(trees, rmse), aes(trees, rmse)) +
  geom_line(size = 1)



#Tuning Hyper Parameters

hyper_grid <- expand.grid(
  mtry            = floor(16 * c(.05, .15, .25, .333, .4)),
  min.node.size   = c(1, 3, 5),
  replace         = c(TRUE, FALSE),
  sample.fraction = c(.5, .63, .8),
  rmse            = NA
)
# number of hyperparameter combinations
nrow(hyper_grid)
## [1] 90
head(hyper_grid)


for(i in seq_len(nrow(hyper_grid))) {
  # fit model for ith hyperparameter combination
  fit <- ranger(
    formula         = deposit ~ ., 
    data            = trn, 
    num.trees       = 300,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    replace         = hyper_grid$replace[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose         = FALSE,
    seed            = 1535,
    respect.unordered.factors = 'order',
  )
  # export OOB error 
  hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
}

hyper_grid %>%
  arrange(rmse) %>%
  head(10)

#Fitting the accurate RF model usinf Random Forest

fit_final <- randomForest(
  formula         = deposit ~ ., 
  data            = train, 
  ntree       = 300,
  mtry            = 4,
  nodesize   = 5,
  replace         = TRUE
)



#Prediction on training data

pred.random_train <- predict(fit_final, data = train, type="class")
table(train$deposit, pred.random_train, dnn=c("Truth","Predicted"))

#Prediction on test data
pred.random_test <- predict(fit_final, test, type="class")
table(test$deposit, pred.random_test, dnn=c("Truth","Predicted"))

#ROC Curve ,Gain Chart and K-S Chart for test data and AUC values 0.722

bank.test.random = predict(fit_final,test, type="prob")
pred = prediction(bank.test.random[,2], test$deposit)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]  #0.79

gain <- performance(pred, "tpr", "rpp")
plot(gain, main = "Gain Chart")

ks=max(attr(perf,'y.values')[[1]]-attr(perf,'x.values')[[1]])
plot(perf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
print(ks); #0.47

#ROC and Gain chart curve and K-S Chart for training data 
bank.train.random = predict(fit_final,train, type="prob")
pred1 = prediction(bank.train.random[,2], train$deposit)
perf1 = performance(pred1, "tpr", "fpr")
plot(perf1, colorize=TRUE)

slot(performance(pred1, "auc"), "y.values")[[1]]  #0.99

gain1 <- performance(pred1, "tpr", "rpp")
plot(gain1, main = "Gain Chart")

ks=max(attr(perf1,'y.values')[[1]]-attr(perf1,'x.values')[[1]])
plot(perf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
print(ks);    #0.93

# Extract tibble of variable importance scores
vip::vi(fit_final)

# Construct ggplot2-based variable importance plot
vip::vip(fit_final, num_features = 10)


###Boosting
bank.boost = boosting(deposit~., data = train, boos = TRUE, mfinal = 100,
                      control = rpart.control(cp = 0.01, maxdepth = 8))
save(bank.boost, file = "bank.boost.Rdata")


# Training AUC, ROC, Gain Chart and KS-plot
pred.bank.boost= predict(bank.boost, newdata = train, type = "class")
table(train$deposit, pred.bank.boost[,3], dnn=c("Truth","Predicted"))
pred <- prediction(pred.bank.boost$prob[,2], train$deposit)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))   #0.82

gain <- performance(pred, "tpr", "rpp")
plot(gain, main = "Gain Chart")

ks=max(attr(perf,'y.values')[[1]]-attr(perf,'x.values')[[1]])
plot(perf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
print(ks); #0.49


# Testing AUC, ROC, Gain Chart and KS-plot

pred.bank.boost= predict(bank.boost, newdata = test)
table(test$deposit, pred.random_test, dnn=c("Truth","Predicted"))
pred <- prediction(pred.bank.boost$prob[,2], test$deposit)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values")) #0.79

gain <- performance(pred, "tpr", "rpp")
plot(gain, main = "Gain Chart")

ks=max(attr(perf,'y.values')[[1]]-attr(perf,'x.values')[[1]])
plot(perf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
print(ks); #0.45


# Construct ggplot2-based variable importance plot
summary(bank.boost)








