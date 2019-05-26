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

#QNs to be answered

bank <- read.csv("Bank.csv")
bank$duration = NULL;

# Modifying the output column

bank$deposit <- ifelse(bank$deposit == 'yes', 1,0)
bank$deposit <- as.factor(bank$deposit)


# Partition the data into train/test sets
set.seed(1535)
trn_id <- createDataPartition(
  y = bank$deposit, p = 0.7, list = FALSE
)
trn <- bank[trn_id, ]   # training data
tst <- bank[-trn_id, ]  # test data


# Function to calculate accuracy
accuracy <- function(pred, obs) {
  sum(diag(table(pred, obs))) / length(obs)
}

##CART
# Optimal tree (test error = 0%):
cart2 <- rpart(
  deposit ~ ., data = trn, 
  control = list(cp = 0, minbucket = 1, minsplit = 1)
)


# Get test set predictions
pred2 <- predict(
  cart2, newdata = tst, 
  type = "class"
)


# Compute test set accuracy
accuracy(
  pred = pred2, 
  obs = tst$deposit
)  #0.63


# Plot train and test ROC curves
roc_trn <- roc(  
  predictor = predict(cart2, newdata = trn, type = "prob")[, 1L], 
  response = trn$deposit,
  levels = rev(levels(trn$deposit))
)

roc_tst <- roc( 
  predictor = predict(cart2, newdata = tst, type = "prob")[, 1L], 
  response = tst$deposit,
  levels = rev(levels(tst$deposit))
)
plot(roc_trn)
lines(roc_tst, col = "dodgerblue2", lty = 2)
legend("bottomright", legend = c("Train", "Test"), bty = "n", 
       cex = 1.5, col = c("black", "dodgerblue2"), inset = 0.01, 
       lwd = 2, lty = c(1, 2))

# Extract tibble of variable importance scores
vip::vi(cart2)

# Construct ggplot2-based variable importance plot
vip::vip(cart2, num_features = 10)

# Gain Chart
lift1 <- lift(deposit ~ balance+age+day, data = tst)
xyplot(lift1)

#Lift plot
gain <- performance(pred, "tpr", "rpp")
plot(gain, main = "Gain Chart")


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

#Fitting the accurate RF model

fit_final <- ranger(
  formula         = deposit ~ ., 
  data            = trn, 
  num.trees       = 300,
  mtry            = 4,
  min.node.size   = 5,
  sample.fraction = .5,
  replace         = TRUE,
  importance      = 'permutation',
  respect.unordered.factors = 'order',
  verbose         = FALSE,
  seed            = 1535
)

vip::vip(fit_final, num_features = 15)


#Partial dependency plots for RF  #Not working...need to check

fit_final %>%
  partial(pred.var = "education", type = "auto") %>%
  autoplot()

fit_final %>%
  partial(pred.var = "age", train = as.data.frame(trn)) %>%
  autoplot()

fit_final %>%
  partial(pred.var = "age", train = as.data.frame(trn), ice = TRUE) %>%
  autoplot(alpha = 0.05, center = TRUE)

# Predictions from the ranger model

#get the prediction for the ranger model
pred.data <- predict(fit_final, data = tst)
table(pred.data$predictions)

###GBM
bank.boost = boosting(deposit~., data = trn, boos = T)
save(credit.boost, file = "credit.boost.Rdata")

bank.gbm <- gbm(deposit~., data = trn, distribution = "bernoulli", 
                   n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)


#No of features
vip::vip(bank.gbm, num_features = 10)

# Training AUC
pred.bank.gbm = predict(bank.gbm, newdata = tst, n.trees =  5000)
pred <- prediction(pred.bank.gbm$prob[,2], trn$deposit)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

pred.credit.boost= predict(credit.boost, newdata = credit.test)
# Testing AUC
pred <- prediction(pred.credit.boost$prob[,2], credit.test$default.payment.next.month)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

