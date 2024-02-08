# Library imports
library(caret)
library(tidyverse)    
library(mltools) 
library(pROC)
library(randomForest)
library(rpart)
library(MLeval) 
library(e1071)

# Make figure and chart sizes bigger
options(repr.plot.width=12, repr.plot.height=8)

# Set seed for reproduceability 
set.seed(123)

#Importing the dataset
heart_disease_modified <- 
  read_csv("~/Desktop/PMIM402J ASSIGNMENT/CLASSIFICATION/heart_disease_modified.csv")

#The attribute idx was obviously used as an index for this data set at some point,
#however for classification this is meaningless information so we'll remove it. 
#Similarly, we'll remove the categorical attribute has_pace_maker as every single 
#sample has the same value:

heart_disease_modified <- heart_disease_modified %>% select(-c(...1, Patient_ID, pace_maker))

# We have three types of data in the data set: continuous, categorical/discrete, 
# and ordinal. We can represent continuous and ordinal data in the real coordinate 
# space (as integers and/or floats) because that data naturally has an ordering and 
#subsequently we can compare distance between data to distinguish different classes. 
#For our categorical data, there is no such ordering and so we must convert this data 
#into some other sensible representation

#First we need to change the drug column into categorical variable by creating seperate columns
# for aspirin and clopidogrel AND change fam_hist from yes, no to 0,1

heart_disease_modified <- heart_disease_modified %>%
  mutate(taken_aspirin = if_else(drug == 'Aspirin' | drug == 'Both', 0, 1)) %>%
  mutate(taken_clopidogrel = if_else(drug == 'Clopidogrel' | drug == 'Both', 0, 1)) %>%
  mutate(family_hist = if_else(fam_hist == 'yes', 0, 1)) %>%
  select(-c(fam_hist,drug))

#Exploring the datasets and checking for outliers
summary(heart_disease_modified)

#checking the datatypes of each column in the data
str(heart_disease_modified)

# CHecking for NA's
na_check <- anyNA(heart_disease_modified)
na_check

# Making the class a factor 
heart_disease_modified$class <- as.factor(heart_disease_modified$class)

summary(heart_disease_modified)

# Perform a balanced 70%/30% split of samples in heart data set to give the train data more 
# data points to train and also a reasonable 30% of the dataset to train and get accurate 
# values

train_partition <- createDataPartition(
  heart_disease_modified$class,
  p = 0.7,
  list = FALSE
)

# Extract training set as balanced 70% of heart data set
train_data = heart_disease_modified[train_partition,]

# Extract test set as balanced 30% of heart data set
test_data = heart_disease_modified[-train_partition,]

# Check sizes of new subsets
paste(cat(
  "  Training set:", nrow(train_data), "samples\n",
  "  Test set:", nrow(test_data), "samples"
))

table(train_data$class)
table(test_data$class)

# Train datasets
train_data$class <- make.names(train_data$class)
x_train = subset(train_data, select = -c(class))
y_train = train_data$class

#Test data sets
test_data$class <- make.names(test_data$class)
x_test = subset(test_data, select = -c(class))
y_test = test_data$class


# Create a training control object for K-fold cross-validation
train_control <- trainControl(
  method = "cv",      # Use K-fold cross-validation
  number = 5,        # Number of folds (change as needed)
  verboseIter = FALSE,  # Show progress during cross-validation
  savePredictions = "all",    # Save predictions at each resampling
  classProbs = TRUE           # Additionally, compute class probabilities at each resampling
)

#heart_disease_modified$class <- make.names(heart_disease_modified$class)

# Train the Random Forest model using K-fold cross-validation
rf_model <- train(
  x_train, y_train, # Formula specifying the target and predictor variables
  method = "rf",            # Use Random Forest as the modeling method
  trControl = train_control, # Specify the training control object
  tuneGrid = expand.grid(
    mtry = floor(sqrt(length(train_data))) # keep the hyperparameter 'mtry' 
    #fixed to a default value of sqrt(a) where a is the number of attributes in the data
  )
)

# Print the cross-validated results
print(rf_model)

# Generate confusion matrix
confusionMatrix(rf_model, norm = "none")

# Plot ROC curve for the default RF classifier
roc <- evalm(
  rf_model,
  silent = TRUE,    # Silence bloody spam output
  title = "ROC Curve: Random Forest Classifier (default configuration)",
  plots = "r"    # Select ROC curve
)

############# Support Vector Model #######################


# train a svm model
control <- caret::trainControl(method = "cv", number = 5,
                               savePredictions = TRUE, classProbs = TRUE)
svmFit <- caret::train(x_train, y_train, data = train_data, method = "svmLinear",
)

svmFit

svmGrid <- expand.grid(C=c(0.1:5))

svmFitGrid <- caret::train(x_train, y_train, data = train_data, method = "svmLinear",
                           tuneGrid = svmGrid, trControl = control)

svmFitGrid

# Confusion Matrix
confusionMatrix(svmFitGrid, norm='none') # show confusion matrix aggregated across all folds


# Plot ROC curve for the default RF classifier
roc <- evalm(
  svmFitGrid,
  silent = TRUE,    # Silence bloody spam output
  title = "ROC Curve: SVM Classifier (default configuration)",
  plots = "r"    # Select ROC curve
)


########################## TESTING THE MODEL #########################
y_test <- as.factor(y_test)
levels(predict(rf_model, x_test))
levels(y_test)

# Testing the model for random forest 
confusionMatrix(predict(rf_model, x_test), y_test)# use random forrest model to make 
#predictions on the test set and plot confusion matrix of predictions and actual values

#Testing the model for svm 
confusionMatrix(predict(svmFitGrid, x_test), y_test) # use svm model to make predictions 
#on the test set and plot confusion matrix of predictions and actual values
###############################################################


######  QUESTION 2 #######


# The model chosen to optimize is the random forest model because of a few reasons. Random forest 
# model gives a variable important score to indicate the relevance of each feature in the model's 
# decision-making process. It has the ability to handle non-linear relationships without 
# any complex transformations and also does not require feature scaling. Random forest is 
# also an ensemble learning method and it combines multiple decision trees to make predictions.

# FEATURE IMPORTANCE 
# In order to optimize the data to give the best result with greater accuracy, our models 
# needs to be pruned thereby removing features with little or no importance to our model
# which is also one of the reasons i chose random forest for optimization.

plot(varImp(rf_model))

# From the plot above, we can see that cp (chest pain) is the feature with the highest importance
# amongst all the features followed by the maximum heart rate achieved "thalach". The least 
# features which are of little or no importance to the model includes fbs, smoker, family_hist,
# taken_clopidogrel, taken_aspirin. 

# Having the result of the importance of the features in mind, we will now retrain the model and \
# see how it affects the accuracy


# List of the variable names with descending level of importance

importance_list <- rownames(
  arrange(
    varImp(rf_model)$importance,    # importance values
    desc(Overall)                     # put in descending order
  )
)

# creating a function to return the results for a model trained 
# on the most important n attributes
pruned_rfmodel<- function(n, importance = importance_list, base_train = x_train, 
                          labels = y_train, train_ctrl = train_control) {
  rf_train <- train(
    base_train[importance[1:n]], # Select top n attributes
    labels,
    method = "rf",
    trControl = train_ctrl,
    # Fixing the hyper parameter 'mtry' at default value of sqrt(n)
    tuneGrid = expand.grid(
      mtry = floor(sqrt(n))
    )
  )
  
  return (cbind(rf_train$results, data.frame("AttributesSelected" = n)))
}

# After creating the fucntion above, we can now evaluate the performance of the RF model
# for n values and compare them as well (recall n being the value of the features according to 
# their importance in the model).

# for n == 1 

pruned_rfmodel_1 <- pruned_rfmodel(1)
pruned_rfmodel_1

# for n == 4 
pruned_rfmodel_4 <- pruned_rfmodel(4)
pruned_rfmodel_4

# for n == 7 
pruned_rfmodel_7 <- pruned_rfmodel(7)
pruned_rfmodel_7

# for n == 9 
pruned_rfmodel_9 <- pruned_rfmodel(9)
pruned_rfmodel_9

# for n == 11
pruned_rfmodel_11 <- pruned_rfmodel(11)
pruned_rfmodel_11

# # for n == 14
pruned_rfmodel_14 <- pruned_rfmodel(14)
pruned_rfmodel_14

# for n == n
pruned_rfmodel_max <- pruned_rfmodel(length(x_train))
pruned_rfmodel_max

# combining all results into a data frame 

pruned_attribute_results <- rbind.data.frame(
  pruned_rfmodel_1, pruned_rfmodel_4, pruned_rfmodel_7, pruned_rfmodel_9,
  pruned_rfmodel_11, pruned_rfmodel_14, pruned_rfmodel_max
)

pruned_attribute_results

# From the results so far, the accuracy of the model increased with increase in

################## SAMPLING #############################

# Since the training and test sets were obtained using stratified sampling, 
# the proportions of positive and negative samples are maintained in both 
# subsets. This is in contrast to random sampling, where the sets are selected 
# randomly, leading to an imbalanced distribution of positive and negative 
# samples between the two sets. Stratified sampling is preferred because it 
# provides a more accurate representation of the source data and ensures a 
# more generalizable model.

# To examine the impact of sampling on accuracy, the model will be retrained 
# using both random sampling to create the training and test sets and 
# bootstrapping for model validation. This comparison will help evaluate 
# the effectiveness of different sampling techniques on the overall performance
# of the model.

# Generate randomly sampled training set
train_random_sample <- slice_sample(heart_disease_modified, prop = 0.7)

# Train rf on randomly sampled training set with bootstrapping
train_random_sample$class <- make.names(train_random_sample$class)

bootstrap_rf <- train(x_train, y_train,
                 data = train_random_sample,
                 method = "rf",
                 trControl = trainControl(
                   method = "boot",            # Resample using bootstrapping
                   savePredictions = TRUE, 
                   classProbs = TRUE 
                 ),
                 tuneGrid = expand.grid(
                   mtry = floor(sqrt(length(train_random_sample)))
                 )
)

print(bootstrap_rf)

confusionMatrix(bootstrap_rf, norm = "none")

########## HYPERPARAMETER OPTIMIZATION ###########

# Hyperparameter optimization refers to the procedure of meticulously adjusting
#the hyperparameters of a machine learning model to discover the most favorable
# combination that yields optimal performance. Unlike model parameters, 
# hyperparameters are set prior to the commencement of the training process 
# and are not learned from the data. These hyperparameters govern different 
# facets of the learning process and exert a direct influence on how the model 
# acquires knowledge and generalizes to new, unseen data.

# Define the hyperparameter grid to search
#param_grid <- expand.grid(mtry = floor(sqrt(length(train_data))),  # Number of variables randomly sampled at each split
               #ntree = 5000) # Number of trees in the forest

# Create the training control object for cross-validation
ctrl <- trainControl(method = "cv",  # Cross-validation method ("cv" for k-fold cross-validation)
                     number = 5,
                     search = "random",  # Random parameter search
                     savePredictions = "all",
                     classProbs = TRUE)      # Number of folds for cross-validation

# Train the Random Forest model using cross-validation and hyperparameter tuning
rf_model_optimized <- train(x_train, y_train,        # Formula specifying the target and features
                  data = train_data,      # Training data
                  method = "rf",    # Random Forest method
                  trControl = ctrl, # Training control object for cross-validation
                  tuneLength = 15,
                  ntree = 5000) # Hyperparameter grid


# Print the best hyperparameters and model performance
print(rf_model_optimized$bestTune)
print(rf_model_optimized$results)


# Generate confusion matrix
confusionMatrix(rf_model_optimized, norm = "none")


# Testing the model for random forest 
confusionMatrix(predict(rf_model_optimized, x_test), y_test)# use random forrest model to make 
#predictions on the test set and plot confusion matrix of predictions and actual values

# Plot ROC curve for the default RF classifier
roc <- evalm(
  rf_model_optimized,
  silent = TRUE,    # Silence bloody spam output
  title = "ROC Curve: Classifier (default configuration)",
  plots = "r"    # Select ROC curve
)


