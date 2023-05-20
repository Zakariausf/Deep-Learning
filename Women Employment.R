library(caret)
library(car)
library(MASS)
library(ROCR)
library(ranger)
library(xgboost)
library(e1071)
library(effects)
library(vip)
library(gam)
library(boot)
library(ggplot2)
library(tidyr)



### Logistic Regression Model ###########################################
modelG <- glm(work~poly(ykids, 2)
              +educ
              +poly(hhours, 2)
              +hage
              +hhours*hwage
              +poly(hwage, 2)
              +tax,
              family=binomial,
              data=femtrainT)

summary(modelG)
Anova(modelG)
vif(modelG)

### LDA ###########################################
LDAmodelB <- lda(work~ykids 
                 + educ
                 + hhours  
                 + hage 
                 + hwage 
                 + tax, 
                 data=femtrainT)

### QDA ###########################################
QDAmodelB <- qda(work~ykids 
                 + educ
                 + hhours  
                 + hage 
                 + hwage 
                 + tax,
                 data=femtrainT)


### Bagging (out-of-bag) ###########################################

hypergridA <- expand.grid(
  trees = c(100,250,500,1000),
  nodesize = c(1,3,5,10),
  OOBinaccuracy = NA,
  CVinaccuracy = NA)

for (i in seq_len(nrow(hypergridA))) {
  fit <- ranger(
    formula = work ~ .,
    data=femtrainT,
    mtry = 13,
    num.trees = hypergridA$trees[i],
    min.node.size = hypergridA$nodesize[i],
    verbose=FALSE,
    respect.unordered.factors = "order")
  hypergridA$OOBinaccuracy[i] <- fit$prediction.error
}

bestbagoob <- ranger(
  formula = work ~ .,
  data=femtrainT,
  num.trees = 500,
  mtry = 13,
  min.node.size = 5,
  verbose=FALSE,
  respect.unordered.factors = "order",
  probability=T,
  importance='permutation')


### Bagging (cross validation) ###########################################
repeatcv <- 10
nfolds <- 5
nfeatures <- 13
m1 <- matrix(NA,nrow(hypergridA),repeatcv*nfolds)
for(r in 1:repeatcv) {
  c <- (r-1)*nfolds
  folds <- createFolds(femtrainT$work,nfolds)
  for (i in seq_len(nrow(hypergridA))) {
    j <- c+1
    for (fold in folds) {
      cvtrain <- femtrainT[-fold,]
      cvtest <- femtrainT[fold,]
      fit <- ranger(
        formula = work ~ .,
        data=cvtrain,
        mtry=nfeatures,
        num.trees = hypergridA$trees[i],
        min.node.size = hypergridA$nodesize[i],
        verbose=FALSE,
        respect.unordered.factors = "order")
      cvtestfit <- predict(fit,data=cvtest)
      cm <- table(cvtest$work,cvtestfit$predictions)
      m1[i,j] <- 1-(cm[1,1]+cm[2,2])/sum(cm)
      j <- j+1
    }
  }
}
hypergridA$CVinaccuracy <- apply(m1,1,mean)

## Fit best CV-tuned bagging model to full training data

bestbagCV <- ranger(
  formula = work ~ .,
  data=femtrainT,
  num.trees = 100,
  mtry = nfeatures,
  min.node.size = 3,
  verbose=FALSE,
  respect.unordered.factors = "order",
  probability=T,
  importance='permutation')

### RANDOM FOREST (out-of-bag) ###########################################

hypergridRF <- expand.grid(
  trees = c(100,200,400),
  mtry = c(2,5,8,10),
  min.node.size = c(3,5,10),
  replace = c(TRUE,FALSE),
  sample.fraction = c(.5,.75),
  oobinaccuracy = NA,
  cvinaccuracy = NA)

for (i in seq_len(nrow(hypergridRF))) {
  fit <- ranger(
    formula = work ~ .,
    data=femtrainT,
    num.trees = hypergridRF$trees[i],
    mtry = hypergridRF$mtry[i],
    min.node.size = hypergridRF$min.node.size[i],
    replace = hypergridRF$replace[i],
    sample.fraction = hypergridRF$sample.fraction[i],
    verbose=FALSE,
    respect.unordered.factors = "order")
  hypergridRF$oobinaccuracy[i] <- sqrt(fit$prediction.error)
}

bestbagRF <- ranger(
  formula = work ~ .,
  data=femtrainT,
  num.trees = 200,
  mtry = 8,
  min.node.size = 5,
  replace = TRUE,
  sample.fraction = 0.50,
  verbose=FALSE,
  respect.unordered.factors = "order",
  probability=T,
  importance='permutation')


### RANDOM FOREST (cross validation) ###########################################
repeatcv <- 5
nfolds <- 5
nfeatures <- 13
m1 <- matrix(NA,nrow(hypergridRF),repeatcv*nfolds)
for(r in 1:repeatcv) {
  c <- (r-1)*nfolds
  folds <- createFolds(femtrainT$work,nfolds)
  for (i in seq_len(nrow(hypergridRF))) {
    j <- c+1
    for (fold in folds) {
      cvtrain <- femtrainT[-fold,]
      cvtest <- femtrainT[fold,]
      fit <- ranger(
        formula = work ~ .,
        data=cvtrain,
        num.trees = hypergridRF$trees[i],
        mtry = hypergridRF$mtry[i],
        min.node.size = hypergridRF$min.node.size[i],
        replace = hypergridRF$replace[i],
        sample.fraction = hypergridRF$sample.fraction[i],
        verbose=FALSE,
        respect.unordered.factors = "order")
      cvtestfit <- predict(fit,data=cvtest)
      cm <- table(cvtest$work,cvtestfit$predictions)
      m1[i,j] <- 1-(cm[1,1]+cm[2,2])/sum(cm)
      j <- j+1
    }
  }
}
hypergridRF$cvinaccuracy <- apply(m1,1,mean)

bestCVRF <- ranger(
  formula = work ~ .,
  data=femtrainT,
  num.trees = 400,
  mtry = 10,
  min.node.size = 3,
  replace = TRUE,
  sample.fraction = 0.50,
  verbose=FALSE,
  respect.unordered.factors = "order",
  probability=T,
  importance='permutation')


### Gradient Boosting Models ###########################################

ytrain <- femtrainTgbm$work-1
xtrain <- as.matrix(femtrainTgbm[,1:13])
ytest <- femtestTgbm$work-1
xtest <- as.matrix(femtestTgbm[,1:13])

hypergridgbm <- expand.grid(
  eta=c(0.1,0.05,0.01,0.005),
  max_depth = c(3,5), 
  min_child_weight = c(5,10), 
  subsample = c(0.5,0.75),
  colsample_bytree = c(1.0),
  colsample_bynode = c(1.0),
  loss = NA,
  trees = NA
)

for (i in seq_len(nrow(hypergridgbm))) {
  xgb1cv <- xgb.cv(
    data=xtrain, label=ytrain,
    objective="binary:logistic",
    nrounds = 2000,  
    nfold=10,
    early_stopping_rounds = 5,
    verbose=0,
    params = list(
      eta = hypergridgbm$eta[i], 
      max_depth=hypergridgbm$max_depth[i],
      min_child_weight=hypergridgbm$min_child_weight[i],
      subsample=hypergridgbm$subsample[i],
      colsample_bytree=hypergridgbm$colsample_bytree[i],
      colsample_bynode=hypergridgbm$colsample_bynode[i])
  )
  print(i)
  hypergridgbm$loss[i] <- min(xgb1cv$evaluation_log$test_logloss_mean)
  hypergridgbm$trees[i] <- xgb1cv$best_iteration
}

xgbfit <- xgboost(
  params = list(
    eta = 0.1, 
    max_depth=5,
    min_child_weight=10,
    subsample=0.75,
    colsample_bytree=1.0,
    colsample_bynode=1.0),
  data=xtrain,
  label=ytrain,
  objective="binary:logistic",
  nrounds=68)


### Linear Support Vector Machine ###########################################

all <- vector("list",20)
for (i in 1:20) {
  test <- tune(svm,work~.,data=femtrainT,
               kernel="linear",
               ranges=list(cost=2^(-4:2))) 
  all[[i]] <- c(test$best.parameters[1,1])
}
table(unlist(lapply(all, paste, collapse = " ")))

svmlinear <- svm(work~.,data=femtrainT,kernel="linear",
                 cost=0.125,probability=TRUE)


### Polynomial Support Vector Machine ###########################################

all <- vector("list",10)
for (i in 1:10) {
  test <- tune(svm,work~.,data=femtrainT,
               kernel="polynomial",
               ranges=list(
                 degree=c(2,3,4),
                 cost=2^(-4:2),
                 gamma=2^(-4:1)))
  all[[i]] <- c(test$best.parameters[1,1],
                test$best.parameters[1,2],
                test$best.parameters[1,3])
}
table(unlist(lapply(all, paste, collapse = " ")))

svmpoly <- svm(work~.,data=femtrainT,kernel="polynomial",
               cost=2,degree=2,gamma=0.25, 
               probability=TRUE)


### Radial Support Vector Machine ###########################################

all <- vector("list",10)
tsvmradial <- system.time(
  for (i in 1:10) {
    test <- tune(svm,work~.,data=femtrainT,
                 kernel="radial",
                 ranges=list(cost=2^(-4:2),gamma=2^(-4:1)))
    all[[i]] <- c(test$best.parameters[1,1],test$best.parameters[1,2])
  })
table(unlist(lapply(all, paste, collapse = " ")))

svmradial <- svm(work~.,data=femtrainT,kernel="radial",
                 cost=1,gamma=0.0625,probability=TRUE)







