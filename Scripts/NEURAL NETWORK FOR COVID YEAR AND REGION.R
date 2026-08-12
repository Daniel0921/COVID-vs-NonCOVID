# ============================================================
# COVID YEAR AND REGION NEURAL NETWORK ANALYSIS
# ============================================================
#
# MODEL 1:
# Predict whether an observation belongs to a COVID year
#
# MODEL 2:
# Predict geographic region
#
# This revised version intentionally removes unnecessary
# packages and avoids dependencies such as:
#
#   mice
#   reformulas
#   caret
#   missForest
#   randomForest
#
# Only the nnet package is required for modeling.
#
# ============================================================


# ============================================================
# 1. LOAD REQUIRED PACKAGE
# ============================================================

library(nnet)


# ============================================================
# 2. LOAD DATA
# ============================================================
#
# Recommended:
# Keep the CSV in the same project folder as this R script.
#
# This avoids machine-specific paths such as:
#
# C:/Users/Name/OneDrive/...
#
# ============================================================

data <- read.csv("")


# ============================================================
# 3. INSPECT DATA
# ============================================================

head(data)

str(data)

summary(data)

print(names(data))


# ============================================================
# 4. REMOVE INCOMPLETE OBSERVATIONS
# ============================================================
#
# Neural networks require complete predictor values.
#
# This project uses complete-case filtering rather than
# multiple imputation, so mice is no longer required.
#
# ============================================================

data <- data[
  complete.cases(data),
]


# ============================================================
# ============================================================
#
# MODEL 1
#
# COVID YEAR BINARY CLASSIFICATION
#
# ============================================================
# ============================================================


# ============================================================
# 5. PREPARE COVID YEAR DATA
# ============================================================
#
# Remove variables that should not be used as predictors:
#
#   Region
#   Country
#   Happiness.score
#
# Target:
#
#   COVID.Year
#
# ============================================================

covid_data <- subset(
  data,
  select = -c(
    Region,
    Country,
    Happiness.score
  )
)


# ============================================================
# 6. PREPARE TARGET VARIABLE
# ============================================================

covid_target <- covid_data$COVID.Year


# Make sure the target is numeric 0/1.
#
# If COVID.Year is already coded as 0 and 1,
# this leaves it unchanged.

covid_target <- as.numeric(
  as.character(covid_target)
)


# ============================================================
# 7. PREPARE PREDICTOR MATRIX
# ============================================================

covid_predictors <- covid_data[
  ,
  names(covid_data) != "COVID.Year"
]


# Convert categorical predictors, if any, into dummy variables.

covid_matrix <- model.matrix(
  ~ . - 1,
  data = covid_predictors
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================
#
# 80% Training
# 20% Testing
#
# ============================================================

set.seed(42)

covid_train_indices <- sample(
  seq_len(
    nrow(covid_matrix)
  ),
  size = floor(
    0.80 * nrow(covid_matrix)
  )
)


covid_x_train <- covid_matrix[
  covid_train_indices,
  ,
  drop = FALSE
]

covid_x_test <- covid_matrix[
  -covid_train_indices,
  ,
  drop = FALSE
]

covid_y_train <- covid_target[
  covid_train_indices
]

covid_y_test <- covid_target[
  -covid_train_indices
]


# ============================================================
# 9. NORMALIZE USING TRAINING DATA
# ============================================================
#
# IMPORTANT:
#
# Scaling parameters are learned ONLY from the training set.
#
# Using the entire dataset before splitting can cause
# information leakage.
#
# ============================================================

covid_means <- apply(
  covid_x_train,
  2,
  mean
)

covid_sds <- apply(
  covid_x_train,
  2,
  sd
)


# Prevent division by zero for constant columns.

covid_sds[
  covid_sds == 0
] <- 1


covid_x_train_scaled <- scale(
  covid_x_train,
  center = covid_means,
  scale = covid_sds
)


covid_x_test_scaled <- scale(
  covid_x_test,
  center = covid_means,
  scale = covid_sds
)


# ============================================================
# 10. TRAIN COVID YEAR NEURAL NETWORK
# ============================================================
#
# size = 10
#     10 neurons in the hidden layer
#
# entropy = TRUE
#     Uses cross-entropy loss for classification
#
# maxit = 1000
#     Maximum training iterations
#
# decay
#     Small regularization penalty to reduce overfitting
#
# ============================================================

set.seed(42)

covid_nn <- nnet(
  x = covid_x_train_scaled,
  y = covid_y_train,
  size = 10,
  entropy = TRUE,
  decay = 0.001,
  maxit = 1000,
  trace = FALSE
)


# ============================================================
# 11. COVID YEAR PREDICTIONS
# ============================================================

covid_probability <- predict(
  covid_nn,
  covid_x_test_scaled,
  type = "raw"
)


covid_prediction <- ifelse(
  covid_probability >= 0.50,
  1,
  0
)


# ============================================================
# 12. COVID YEAR MODEL EVALUATION
# ============================================================

covid_accuracy <- mean(
  covid_prediction ==
    covid_y_test
)


covid_mse <- mean(
  (
    covid_probability -
      covid_y_test
  )^2
)


covid_confusion_matrix <- table(
  Predicted = covid_prediction,
  Actual = covid_y_test
)


cat(
  "\n",
  "============================================================\n",
  "COVID YEAR NEURAL NETWORK RESULTS\n",
  "============================================================\n",
  "\n",
  "Accuracy:",
  round(
    covid_accuracy,
    4
  ),
  "\n",
  "MSE:",
  round(
    covid_mse,
    4
  ),
  "\n\n"
)


print(
  covid_confusion_matrix
)


# ============================================================
# 13. COVID YEAR PREDICTION TABLE
# ============================================================

covid_results <- data.frame(
  
  Actual = covid_y_test,
  
  Predicted = covid_prediction,
  
  Probability_COVID_Year =
    as.numeric(
      covid_probability
    )
  
)


print(
  covid_results
)


# ============================================================
# ============================================================
#
# MODEL 2
#
# REGION MULTICLASS CLASSIFICATION
#
# ============================================================
# ============================================================


# ============================================================
# 14. PREPARE REGION DATA
# ============================================================
#
# Remove:
#
#   COVID.Year
#   Country
#   Happiness.score
#
# Target:
#
#   Region
#
# ============================================================

region_data <- subset(
  data,
  select = -c(
    COVID.Year,
    Country,
    Happiness.score
  )
)


# ============================================================
# 15. PREPARE REGION TARGET
# ============================================================

region_target <- factor(
  region_data$Region
)


# Save original region names.

region_levels <- levels(
  region_target
)


cat(
  "\nRegion Classes:\n"
)

print(
  region_levels
)


# ============================================================
# 16. PREPARE REGION PREDICTORS
# ============================================================

region_predictors <- region_data[
  ,
  names(region_data) != "Region"
]


region_matrix <- model.matrix(
  ~ . - 1,
  data = region_predictors
)


# ============================================================
# 17. REGION TRAIN / TEST SPLIT
# ============================================================

set.seed(42)

region_train_indices <- sample(
  seq_len(
    nrow(region_matrix)
  ),
  size = floor(
    0.80 * nrow(region_matrix)
  )
)


region_x_train <- region_matrix[
  region_train_indices,
  ,
  drop = FALSE
]

region_x_test <- region_matrix[
  -region_train_indices,
  ,
  drop = FALSE
]

region_y_train <- region_target[
  region_train_indices
]

region_y_test <- region_target[
  -region_train_indices
]


# ============================================================
# 18. NORMALIZE REGION DATA
# ============================================================

region_means <- apply(
  region_x_train,
  2,
  mean
)

region_sds <- apply(
  region_x_train,
  2,
  sd
)


region_sds[
  region_sds == 0
] <- 1


region_x_train_scaled <- scale(
  region_x_train,
  center = region_means,
  scale = region_sds
)


region_x_test_scaled <- scale(
  region_x_test,
  center = region_means,
  scale = region_sds
)


# ============================================================
# 19. ONE-HOT ENCODE REGION TARGET
# ============================================================
#
# Region is NOT binary.
#
# Instead of converting Region into 0/1, each region gets its
# own output neuron.
#
# Example:
#
# Region A -> 1 0 0 0
# Region B -> 0 1 0 0
# Region C -> 0 0 1 0
# Region D -> 0 0 0 1
#
# ============================================================

region_y_train_matrix <- class.ind(
  region_y_train
)


# ============================================================
# 20. TRAIN REGION NEURAL NETWORK
# ============================================================
#
# softmax = TRUE
#
# The output layer produces probabilities across all
# available Region classes.
#
# ============================================================

set.seed(42)

region_nn <- nnet(
  x = region_x_train_scaled,
  y = region_y_train_matrix,
  size = 10,
  softmax = TRUE,
  decay = 0.001,
  maxit = 1500,
  trace = FALSE
)


# ============================================================
# 21. REGION PROBABILITY PREDICTIONS
# ============================================================

region_probabilities <- predict(
  region_nn,
  region_x_test_scaled,
  type = "raw"
)


# ============================================================
# 22. CONVERT PROBABILITIES TO REGION CLASSES
# ============================================================
#
# Select the region with the highest predicted probability.
#
# ============================================================

region_predicted_index <- apply(
  region_probabilities,
  1,
  which.max
)


region_prediction <- factor(
  colnames(
    region_probabilities
  )[
    region_predicted_index
  ],
  levels = levels(
    region_y_test
  )
)


# ============================================================
# 23. REGION MODEL EVALUATION
# ============================================================

region_accuracy <- mean(
  region_prediction ==
    region_y_test
)


region_confusion_matrix <- table(
  Predicted = region_prediction,
  Actual = region_y_test
)


cat(
  "\n",
  "============================================================\n",
  "REGION NEURAL NETWORK RESULTS\n",
  "============================================================\n",
  "\n",
  "Accuracy:",
  round(
    region_accuracy,
    4
  ),
  "\n\n"
)


print(
  region_confusion_matrix
)


# ============================================================
# 24. REGION PREDICTION TABLE
# ============================================================

region_results <- data.frame(
  
  Actual_Region =
    region_y_test,
  
  Predicted_Region =
    region_prediction,
  
  Correct =
    region_y_test ==
    region_prediction
  
)


print(
  region_results
)


# ============================================================
# 25. EXPORT RESULTS
# ============================================================

write.csv(
  covid_results,
  "COVID_Year_Neural_Network_Predictions.csv",
  row.names = FALSE
)


write.csv(
  region_results,
  "Region_Neural_Network_Predictions.csv",
  row.names = FALSE
)


# ============================================================
# 26. FINAL MODEL SUMMARY
# ============================================================

cat(
  "\n",
  "============================================================\n",
  "NEURAL NETWORK ANALYSIS COMPLETE\n",
  "============================================================\n",
  "\n",
  "MODEL 1 - COVID YEAR CLASSIFICATION\n",
  "Accuracy:",
  round(
    covid_accuracy * 100,
    2
  ),
  "%\n",
  "\n",
  "MODEL 2 - REGION CLASSIFICATION\n",
  "Accuracy:",
  round(
    region_accuracy * 100,
    2
  ),
  "%\n",
  "\n",
  "============================================================\n"
)
