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

##Gradient Boosting

set.seed(1535)
bank_gbm <- gbm(
  formula = deposit ~ .,
  data = trn,
  distribution = "bernoulli",
  n.trees = 100, 
  shrinkage = 0.1, 
  interaction.depth = 1, 
  n.minobsinnode = 10, 
  cv.folds = 5 
)  

# find index for n trees with minimum CV error
min_MSE <- which.min(bank_gbm$cv.error)   #No of trees 1012
# get MSE and compute RMSE
sqrt(bank_gbm$cv.error[min_MSE])   #0.44

gbm.perf(bank_gbm, method = "cv") # or "OOB"


#Hyper Parameter tuning

# search grid
hyper_grid <- expand.grid(
  n.trees = 2000,
  shrinkage = .01,
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 10, 15)
)
model_fit <- function(n.trees, shrinkage, interaction.depth, n.minobsinnode) {
  set.seed(1535)
  m <- gbm(
    formula = deposit ~ .,
    data = trn,
    distribution = "bernoulli",
    n.trees = n.trees,
    shrinkage = shrinkage,
    interaction.depth = interaction.depth,
    n.minobsinnode = n.minobsinnode,
    cv.folds = 10
  )
  # compute RMSE
  sqrt(min(m$cv.error))
}
hyper_grid$rmse <- pmap_dbl(
  hyper_grid,
  ~ model_fit(
    n.trees = ..1,
    shrinkage = ..2,
    interaction.depth = ..3,
    n.minobsinnode = ..4
  )
)
arrange(hyper_grid, rmse)



#Get important features from the model
vip::vip(bank_gbm, num_features = 15)

#Partial dependency plots for GB  

bank_gbm %>%
  partial(pred.var = "month", train = as.data.frame(trn), ice = TRUE) %>%
  autoplot(alpha = 0.05, center = TRUE)

bank_gbm %>%
  partial(pred.var = "job", train = as.data.frame(trn), ice = TRUE) %>%
  autoplot(alpha = 0.05, center = TRUE)

bank_gbm %>%
  partial(pred.var = "balance", train = as.data.frame(trn), n.trees = 2000) %>%
  autoplot()


##XG Boost
library(recipes)
xgb_prep <- recipe(deposit ~ ., data = trn) %>%
  step_other(all_nominal(), threshold = .05) %>%
  step_integer(all_nominal()) %>%
  prep(training = trn, retain = TRUE) %>%
  juice()
X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "deposit")])
Y1 <- trn$deposit

set.seed(1535)
ames_xgb <- xgb.cv(
  data = X,
  label = Y,
  nrounds = 2000,
  objective = "reg:logistic",
  early_stopping_rounds = 50, 
  nfold = 10,
  verbose = 0,
)  
ames_xgb$evaluation_log %>% tail()
