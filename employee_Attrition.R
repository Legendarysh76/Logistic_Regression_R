# Case II
# Oluwaseyi Henry Orioye

##################################################################################
#.......................STEP ZERO - PRE-PROCESSING...............................
##################################################################################
# Pre-Process the original Data (employee_Retention.csv) file in excel.
# The purpose of this step is to transform all variables into
# numerical variables by creating Dummy Variables.
# See the formulas in the excel file (employee_Retention.xlsx) for how to do this
# This step is assumed completed prior to proceeding to Step one.
# The rest of the analysis is done on the pre-processed
# dataset (employee_Retention_Processed completed.csv).

##########################################################################################
#.......................STEP ONE - INSTALL-PACKAGES and LOAD DATA.....................
##########################################################################################
options("install.lock" = FALSE) # this is to prevent package installation error

# load necessary packages
install.packages("mice") # for multiple imputation (step two)
library(mice)

install.packages("caTools") # for train/test split (step three)
library(caTools)

install.packages("usdm") # to check for multicollinearity (step four)
library(usdm)

install.packages("ROCR") # for prediction model tuning (step six)
library(ROCR)

install.packages("caret") # to build a cross-validated model (step eight)
library(caret)

# read in the data
Employee_Retention <- read.csv(file.choose(), header = TRUE) # employee_Retention_Processed completed.csv

# Convert the data to a data frame (R converts .csv files to a data frame automatically
# but it is a good idea to do so formally anyway (in case you're dealing with other file types)
Employee_Retention<- data.frame(Employee_Retention)

# inspect the data
View(Employee_Retention)

# View structure of dataframe 
str(Employee_Retention) 

# summary tells you number of missing values, among other things
summary(Employee_Retention) # lots of missing data
# Logistic Regression assumes that the response (dependent) variable is a binary (two-value) variable

#########################################################################################################
#.......................STEP TWO - DEAL WITH OUTLIER AND MISSING DATA.....................
#########################################################################################################
# Checking for outliers 
for (i in c(2,14))  
{
  
  q1 <- quantile(Employee_Retention[,i], 0.25, na.rm = TRUE)
  q3 <- quantile(Employee_Retention[,i], 0.75, na.rm = TRUE)
  iqr <- IQR(Employee_Retention[,i],na.rm = TRUE)
  outlier_rows <- subset(Employee_Retention, (Employee_Retention[,i] < q1-1.5*iqr)|(Employee_Retention[,i] > q3+1.5*iqr))
  if (nrow(outlier_rows) > 0)
  {
    print(colnames(Employee_Retention)[i])
    print(outlier_rows)
  }
}

# make a copy of the dataframe in case you decide to delete some of the rows with outliers
Employee_Retention_copy <- Employee_Retention 

#View rows number
nrow(Employee_Retention)

#Remove a specific outlier
Employee_Retention<- Employee_Retention[-c(199), ]

#View row number again. 
nrow(Employee_Retention)

summary(Employee_Retention) # original NAs from missing values
summary(Employee_Retention_copy)

Employee_Retention <- complete(mice(Employee_Retention)) # fill in the missing data using multiple imputation

summary(Employee_Retention) # note: no missing values

str(Employee_Retention) # note that the categorical column (Left) is of datatype character (chr)  

# We want category variables to be cast to factor variables
# it is always a good idea to do this manually instead of relying on R
# to automatically convert them (to factor variables)
Employee_Retention$Left <- factor(Employee_Retention$Left)

class(Employee_Retention$Left) # the column is now of type factor

###################################################################################################################
#.......................STEP THREE - SPLIT THE SAMPLE INTO TRAINING AND TESTING SETS...............
###################################################################################################################
# set a common seed (for replicability of results) 
set.seed(123)

split <- sample.split(Employee_Retention$Left, SplitRatio = 0.7) # note: you need to specify the response 
# variable (Left) in the first parameter

# create the training set
train <- subset(Employee_Retention, split == TRUE)

# create the testing set
test <- subset(Employee_Retention, split == FALSE)

nrow(train)/nrow(Employee_Retention) # ~ 70% of 614
nrow(test)/nrow(Employee_Retention) # ~ 30% of 614

###################################################################################################################
#........................STEP FOUR - MODEL INTERNAL VALIDATION............................ 
###################################################################################################################
# Internal validation means CHECKING the ASSUMPTIONS OF LOGISTIC REGRESSION are met
# begin by building the model on the training set

############################################ MODEL 1 #############################################################
model_Employee_Retention1 <- glm(Left ~., data = train, family = "binomial") # note the function is glm, and the family "binomial"
summary(model_Employee_Retention1)

# assumptions of Logistic Regression

# 1. The response variable is binary (takes only two values) - fulfilled

# 2. The observations are independent - i.e. Left for one
# applicant is independent of Left for another - assumed fulfilled

# 3. There are no outliers - dealt with in step two

# 4. There is no multicollinearity (mc) among the (numerical) predictor variables

# explicitly checking this last assumption

indep_vars <- subset(Employee_Retention, select = -1) # column 1 is not a predictor variable so remove

str(indep_vars) # making sure that the first column has been excluded

usdm::vif(indep_vars) # no values are 5 or more (10 would be worrisome) - so mc not an issue

# assumptions have been fulfilled and the model is internally validated - so we can proceed
# to build the model with the significant predictors: Survey_Score + Projects + AVG_Hours + Tenure + Work_Injury + medium + high

############################################ MODEL 1 #############################################################
model_Employee_Retention2 <- glm(Left ~ Survey_Score + Projects + AVG_Hours + Tenure + Work_Injury + medium + high, data = train, family = "binomial")
summary(model_Employee_Retention2)

# assumptions of Logistic Regression

# 1. The response variable is binary (takes only two values) - fulfilled

# 2. The observations are independent - i.e. Left for one
# applicant is independent of Left for another - assumed fulfilled

# 3. There are no outliers - dealt with in step two

# 4. There is no multicollinearity (mc) among the (numerical) predictor variables

# explicitly checking this last assumption

indep_vars <- subset(Employee_Retention, select = -1) # column 1 is not a predictor variable so remove

str(indep_vars) # making sure that the first column has been excluded

usdm::vif(indep_vars) # no values are 5 or more (10 would be worrisome) - so mc not an issue

# assumptions have been fulfilled and the model is internally validated - so we can proceed

# interpretation of predictor coefficients
# recall that in the logistic regression model, the response is the log-odds of the positive response
# i.e. the probability of a Y (positive response) versus a N (negative response) -
# we first need to convert the coefficients to relate to an odds response instead of a log-odds
# response - we do this using the inverse of log, exp.
odds_coefficients <- exp(coef(model_Employee_Retention2))
odds_coefficients

# (Intercept) Survey_Score     Projects    AVG_Hours       Tenure  Work_Injury       medium         high 
# 0.92410617   0.01499085   0.69845950   1.00212838   2.50897760   0.19778732   0.56334720   0.15894387

# we interpret the Tenure coefficient as: all other things being equal,
# the odds of an Employee with longer Tenure leaving is 2.5 times (Great odd!) than those with shorter Tenures

# On the other hand, the odds of an employee having a higher Survey_Score NOT Leaving
# are 1/0.01499085 ~ 66.7 times than those employees with a lowers survey scores leaving, with all other things being equal

###################################################################################################################
#........................STEP FIVE - MODEL EXTERNAL VALIDATION............................ 
###################################################################################################################
# compute probabilities of "Y" (positive response) on the test set
pred <- predict(model_Employee_Retention2, newdata = test, type = "response")

# translate the probabilities into predictions using the default threshold of 0.5
predictions <- vector()

for (i in length(pred))
{
  predictions <- c(predictions, ifelse(pred > 0.5, "Y", "N"))
}

# construct the confusion matrix
table(predictions, test$Left)

# predictions   no  yes
#             N 1643  320
#             Y  286  751

# (a)	What was the predictive accuracy of the model with a threshold of 0.5? 
#  79.8% or 0.798

# (b) What is the corresponding sensitivity of this model? 
#      70.1% or 0.7012

# (c)	What is the corresponding specificity of this model? 
#    85.17% or 0.8517


# we can increase the specificity by raising the threshold 
# (currently at the default value of 0.5)
predictions <- vector()

for (i in length(pred))
{
  predictions <- c(predictions, ifelse(pred > 0.8, "Y", "N"))
}

# construct the confusion matrix
table(predictions, test$Left)

# predictions   no  yes
#             N 1821  932
#             Y  108  139

#(e)	What was the predictive accuracy of the model with a threshold of 0.8? 
#  65.33% or 0.6533

#(f)	What is the corresponding sensitivity of the model? 
#  12.97% or 0.1297

#(g)	What is the corresponding specificity of the model? 
#  94.40% or 0.9440


# note: if we wished to increase the sensitivity, we'd lower the threshold

###################################################################################################################
#........................STEP SIX - PREDICTION MODEL TUNING............................ 
###################################################################################################################
# In general, the performance of the model can be tuned to a specific
# sensitivity and specifity by computing the model performance cutoffs
pred <- predict(model_Employee_Retention2, newdata = test, type = "response")

pred_obj <- prediction(pred, test$Left)

perf_obj <- performance(pred_obj, "tpr", "tnr")


cutoffs <- data.frame(cut=perf_obj@alpha.values[[1]], tnr=perf_obj@x.values[[1]], 
                      tpr=perf_obj@y.values[[1]])
cutoffs

new_cutt<-subset(cutoffs[(cutoffs$tnr > 0.7) &  (cutoffs$tpr >= 0.90),])
new_cutt

# note: the cutoffs dataframe also has the corresponding tpr and tnr 
# values in its second and third columns

     # cut       tnr         tpr
# 1   Inf       1.0000000 0.000000000
# 2   0.9844038 0.9994816 0.000000000
# 3   0.9808888 0.9989632 0.000000000
# 4   0.9804862 0.9984448 0.000000000
# 5   0.9775262 0.9979264 0.000000000
# 6   0.9730444 0.9974080 0.000000000
# 7   0.9720077 0.9968896 0.000000000
# 8   0.9710379 0.9963712 0.000000000

predictions <- vector()

for (i in length(pred))
{
  predictions <- c(predictions, ifelse(pred > 0.31, "Y", "N"))
}

# construct the confusion matrix
table(predictions, test$Left)

#.   predictions    no  yes
#.               N 1433  103
#.               Y  496  968

# model_sensitivity = 65.4%

# model_specificity = 75.9%

# note: model_accuracy = 68.6%

###################################################################################################################
#........................STEP SEVEN - MAKE ACTUAL PREDICTIONS............................ 
###################################################################################################################
# suppose we wished to predict the Left ("N" or "Y") of an application with the following data:
# Using this model, predict whether a two-year, mid-tier-salary, “other” department employee with no work injuries or promotions, 
# working an average of 160 hours per month on three projects, and with survey and evaluation scores of 0.6 and 0.8, respectively, 
# will stay or leave, using both the default threshold and the threshold after tuning the model. 

# (note: the data needs to be made into a list using the list() function)
pred_data <- list(2,1,0,0,0,0,0,0,0,160,3,0.6,0.8)

# and then converted to a data frame
pred_data <- data.frame(pred_data)

# we also need to assign names to the dataframe columns so they match with the column names in the model
colnames(pred_data) <- c('Tenure','medium','high','sales','accounting','IT','marketing','Work_Injury','Promotions','AVG_Hours','Projects','Survey_Score','Eval_Score')

pred <- predict(model_Employee_Retention2, newdata = pred_data, type = "response")

prediction <- ifelse(pred > 0.5, "Y", "N") # using the default cutoff
prediction # N


new_prediction <- ifelse(pred > 0.31, "Y", "N") # using the new cutoff
new_prediction # N

###################################################################################################################
#........................STEP EIGHT - BUILDING A CROSS-VALIDATED MODEL............................ 
###################################################################################################################
# in Linear Regression, we used the difference between R-squared and adjusted R-squared 
# as one indicator of overfitting - (if the difference is less than 0.05, there is no
# overfitting (due to too many variables in the model)
# There are no easily obtainable quantities in Logistic Regression that we can use
# for this purpose. But one way of constructing a model that does not overfit is to
# build one - using cross-validatedation

# set the cross-validation parameters
cv_params <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

# these settings are standard for cross-validation (the 10 stands for: do a 10-fold cross-validation)
cv_model <- train(Left ~ Survey_Score + Projects + AVG_Hours + Tenure + Work_Injury + medium + high, data = Employee_Retention,
                  family = "binomial", method = "glm", trControl = cv_params)

cv_model <- train(Left ~., data = Employee_Retention,
                  family = "binomial", method = "glm", trControl = cv_params)

# note that we're using the full dataset, since the cross-validation will take care
# of splitting it into training and testing subsets
summary(cv_model)

cv_model <- train(Left ~ Survey_Score + Projects + AVG_Hours + Tenure + Work_Injury + medium + high, data = Employee_Retention,
                  family = "binomial", method = "glm", trControl = cv_params)

summary(cv_model)
# note: AIC is not a reliable criterion of model fit in the context of cross-validation - 
# because of this, it is always advisable to use a non-cross-validated model if
# one can (a cross-validated model is also highly variable performances depending
# on the data splitting (which is a black box) - and as such inherently has low
# performance reliability (consistency)) - so, only use cross-validation if overfitting
# is of particular concern

cv_model$pred # the predictions of the model are in the pred column of cv_model

table(cv_model$pred$pred, cv_model$pred$obs) # the confusion matrix of our model

#      no  yes
# no  5544 1035
# yes  885 2536

# tuning the model
pred <- predict(cv_model, newdata = Employee_Retention, type = "prob") # note: the type is "prob" and not "response"

pred_obj <- prediction(pred[2], Employee_Retention$Left)

perf_obj <- performance(pred_obj, "tpr", "tnr")


cutoffs <- data.frame(cut=perf_obj@alpha.values[[1]], tnr=perf_obj@x.values[[1]], 
                      tpr=perf_obj@y.values[[1]])
cutoffs

new_cutt2<-subset(cutoffs[abs(cutoffs$tnr - cutoffs$tpr) < 0.1, ])
new_cutt2


# making predictions using the cutoff threshold - for instance for a more
predictions <- vector()

for (i in length(pred[2]))
{
  predictions <- c(predictions, ifelse(pred[2] > 0.418, "Y", "N"))
}

table(predictions, Employee_Retention$Left)

# predictions   no  yes
#.          N 5292  628
#.          Y 1137 2942

# (b)	What was the predictive accuracy of this model? 
#  82.3% or 0.8234

# (c)	What is the corresponding sensitivity of the model? 
#  0.8232493

# (d)	What is the corresponding specificity of the model? 
#  0.8233007      

# as before, we can make predictions using this model as well (say for pred_data used earlier)
pred_data <- list(2,1,0,0,0,0,0,0,0,160,3,0.6,0.8)

# and then converted to a data frame
pred_data <- data.frame(pred_data)

# we also need to assign names to the dataframe columns so they match with the column names in the model
colnames(pred_data) <- c('Tenure','medium','high','sales','accounting','IT','marketing','Work_Injury','Promotions','AVG_Hours','Projects','Survey_Score','Eval_Score')

pred <- predict(cv_model, newdata = pred_data, type = "prob")

prediction <- ifelse(pred[2] > 0.418, "Y", "N") # using the cutoff threshold from step seven
prediction # N




