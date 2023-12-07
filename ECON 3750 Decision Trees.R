#ECON 3750 Decision Trees


###### Fitting Classification Trees #####

#To create classification and regression trees, we will use the
#tree library

library(tree)


#We will use classification trees to analyze the Carseats data to
#predict whether Sales are high

library(ISLR2)
Carseats <- Carseats
Carseats$High <- factor(ifelse(Carseats$Sales <= 8, "No", "Yes"))

#We will use the tree() function to fit a classification tree to
#predict High using all variables other than Sales.

tree.carseats <- tree(High ~ . - Sales, data = Carseats)

#Looking at the summary
summary(tree.carseats)

#This tells us which variables were used as internal nodes, the number
#of terminal nodes, and the training error rate

#The training error rate here is 9%

#The deviance is given by -2 \sum_m \sum_k n_{mk} log \hat p_{mk}

#We want a small deviance

#The residual mean deviance is the deviance divided by
#n - |T_0|, which is 400 - 27 = 373

#We can graphically display the tree
plot(tree.carseats)
text(tree.carseats, pretty = 0)

#The argument pretty = 0 tells R to include category names
#for any qualitative predictors, instead of simply
#displaying a letter for each category

#Shelving location appears to be the most important criterion
#for sales

#We can look at the split criterion and the number of observations
#in a branch by typing in the name of the tree object. Branches
#that lead to terminal nodes are marked with an asterisk.

tree.carseats

#To evaluate how well the classification tree does on this data,
#we will estimate the test error

#Splitting the data
set.seed(2)
train <- sample(1:nrow(Carseats), 200)
Carseats.test <- Carseats[-train,]
High.test <- Carseats$High[-train]

#Fitting the model on the training data
tree.carseats <- tree(High ~ . - Sales, data = Carseats,
                      subset = train)

#To get class predictions, we will set type = "class"
tree.pred <- predict(tree.carseats, Carseats.test,
                     type = "class")

#Creating the confusion matrix
table(tree.pred, High.test)

#Overall rate of correct predictions
(104 + 50) / 200

#Rerunning predict() may give slightly different results
#due to ties

#Now, we will prune the tree back.

#We will run cv.tree() to perform cross-validation to pick the
#optimal level of tree complexity using cost complexity pruning

#By setting FUN = prune.misclass, R will use the classification
#error rate to make decisions on the CV and pruning process

set.seed(7)
cv.carseats <- cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)
cv.carseats

#size tells us the number of terminal nodes of each tree
#considered. k tells us the value of the cost-complexity
#parameter used (it corresponds to the alpha parameter)

#Despite the name, dev tells us the number of CV errors

#The tree with 9 terminal nodes has the fewest errors

#Plotting the error rate as a function of size and k
par(mfrow=c(1,2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")

#Now, we will apply prune.misclass() to prune our tree to have
#9 nodes

prune.carseats <- prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats, pretty = 0)

#We will use predict() again to see how well this tree performs
tree.pred <- predict(prune.carseats, Carseats.test,
                     type = "class")
table(tree.pred, High.test)

(97 + 58) / 200

#We now have a tree that is more interpretable and slightly
#more accurate


###### Fitting Regression Trees #####

#We will use the Boston data
Boston <- Boston

#Split the data into training and test sets
set.seed(1)
train <- sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston <- tree(medv ~ ., data=Boston, subset = train)
summary(tree.boston)


#Only 4 variables were used to construct the tree.

#The deviance is equal to the sum of the squared errors

#Plotting the tree
plot(tree.boston)
text(tree.boston, pretty = 0)

#lstat — percentage of people with lower socioeconomic status
#rm — average number of rooms
#Larger values of rm and lower values of lstat correspond to higher
#prices

#We could fit a larger tree by passing
#control = tree.control(nobs = length(train), mindev = 0)

tree.boston1 <- tree(medv ~ ., data=Boston, subset = train,
                     control = tree.control(nobs = length(train), mindev = 0))
tree.boston1


#Pruning our tree
cv.boston <- cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type = "b")


#The most complex tree is selected by CV. However, if we wanted to select
#a simpler model, we could use the prune.tree() function.

prune.boston <- prune.tree(tree.boston, best = 5)
plot(prune.boston)
text(prune.boston, pretty = 0)

#If we stick with the CV results
yhat <- predict(tree.boston, newdata = Boston[-train,])
boston.test <- Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0, 1)
mean((yhat - boston.test)^2)


#The test MSE is 35.29. The RMSE is then 5.941, implying our
#predictions are within around $5,941 of the true median home
#value for the census tract

####### Bagging and Random Forests #####


#We will use the randomForest package

#Because bagging is a special case of random forests in which
#m = p, we can use the randomForest() function to fit random
#forests and bagging.

#NOTE: Results might differ based on the version of R and version of
#randomForest that is intalled

library(randomForest)
set.seed(1)
bag.boston <- randomForest(medv ~ ., data = Boston, subset = train,
                           mtry = 12, importance = TRUE)

#mtry = 12 tells R to use all of the predictors, which means we will use
#bagging

bag.boston

#To test performance on the test set
yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0, 1)
mean((yhat.bag - boston.test)^2)

#This is smaller than what we got from optimally pruning a single tree

#To change the number of trees grown by randomForest(), we can use
#the ntree argument. ntree defaults to 500.

bag.boston <- randomForest(medv ~ ., data = Boston, subset = train,
                           mtry = 12, ntree=25, importance = TRUE)
yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])
mean((yhat.bag - boston.test)^2)


#To grow a random forest, we simply select a smaller value for mtry.
#mtry defaults to p/3 for regression trees and sqrt(p) for classification
#trees.

set.seed(1)
rf.boston <- randomForest(medv ~ ., data = Boston,
                          subset = train, mtry = 6, importance = TRUE)
yhat.rf <- predict(rf.boston, newdata = Boston[-train, ])
mean((yhat.rf - boston.test)^2)


#To see how important each variable is
importance(rf.boston)

#%IncMSE gives the mean decrease of accuracy in predictions on the out
#of bag samples when a given variable is permuted

#IncNodePurity gives a measure of the total decrease in node impurity
#that results from splits over that variable

#For regression trees, node impurity is measured by the training RSS. For
#classification trees, it is measured by the deviance.

#To plot these importance measures
varImpPlot(rf.boston)

#Wealth of the community (lstat) and house size are the two
#most important variables


####### Boosting #####

#We will use the gbm package and gbm() function to fit boosted
#regression trees to the Boston dataset. We will use the argument
#distribution = "gaussian" because this is a regression problem. If this
#was a classification problem, we would set distribution = "bernoulli".

library(gbm)

#We set n.trees = 5000 to tell R to make 5000 trees

#interaction.depth = 4 limits the depth of each tree

set.seed(1)
boost.boston <- gbm(medv ~ ., data = Boston[train, ],
                    distribution = "gaussian", n.trees = 5000,
                    interaction.depth = 4)

#Getting the summary
summary(boost.boston)

#summary() gives us a relative influence plot and the relative influence
#statistics.

#For gaussian distribution, this tells us the reduction of squared error
#attributed to each variable

#Again, lstat and rm are the most important variables.


#To make predictions using our boosted model
yhat.boost <- predict(boost.boston,
                      newdata = Boston[-train, ],
                      n.trees = 5000)
mean((yhat.boost - boston.test)^2)

#Our boosted model outperforms random forests and bagging

#If we want to perform boosting with a different shrinkage
#parameter lambda, we can use the shrinkage argument. It
#defaults to 0.001

boost.boston <- gbm(medv ~ ., data = Boston[train, ],
                    distribution = "gaussian", n.trees = 5000,
                    interaction.depth = 4, shrinkage = 0.2)

yhat.boost <- predict(boost.boston,
                      newdata = Boston[-train, ],
                      n.trees = 5000)

mean((yhat.boost - boston.test)^2)

#The model now performs slightly better


####### Bayesian Additive Regression Trees #####

#We will use the BART package and the gbart() function to fit a BART model
#to the Boston housing data set

library(BART)

#gbart() is used for quantitative outcome variables. If you are working with
#binary outcomes, you can use lbart() and pbart()

#To use gbart(), we need to create matrices of predictors for the training
#and test data.

X <- Boston[, 1:12]
Y <- Boston[, "medv"]
X_train <- X[train, ]
Y_train <- Y[train]
X_test <- X[-train, ]
Y_test <- Y[-train]

#Fitting the model
set.seed(1)
bartfit <- gbart(X_train, Y_train, x.test= X_test)

#Computing the test error
yhat.bart <- bartfit$yhat.test.mean
mean((Y_test - yhat.bart)^2)

#BART outperformed random forests and boosting

#Finally, we can see how many times each variable appeared in our collection
#of trees
ord <- order(bartfit$varcount.mean, decreasing = T)
bartfit$varcount.mean[ord]

?gbart

#varcount — a matrix with ndpost rows and nrow(x.train) columns.
#Each row is for a draw. For each variable (corresponding to the columns), the
#total count of the number of times that variable is used in a tree decision
#rule (over all trees) is given.




