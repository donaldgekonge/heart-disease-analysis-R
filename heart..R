install.packages("tidyverse")  # For data manipulation and plotting
install.packages("caret")      # For machine learning and evaluation
install.packages("corrplot")   # For the correlation heatmap


# Load libraries
library(tidyverse)

# URL for the UCI Heart Disease dataset (processed Cleveland version)
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# The data doesn't have headers, so we define them based on the UCI documentation
column_names <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                  "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target")

# Read the data
heart_data <- read_csv(url, col_names = column_names, na = "?")

# Preview the data
head(heart_data)


# Using the tidyverse to change the target column
clean_data <- heart_data %>%
  mutate(target = ifelse(target > 0, 1, 0))

# Check the count
table(clean_data$target)


# 1. Check for missing values (NAs)
colSums(is.na(heart_data))

# 2. Fill NAs in 'ca' and 'thal' using the median
clean_data <- heart_data %>%
  mutate(ca = ifelse(is.na(ca), median(ca, na.rm = TRUE), ca),
         thal = ifelse(is.na(thal), median(thal, na.rm = TRUE), thal))

# 3. Verify
colSums(is.na(clean_data))


library(corrplot)

# 1. Calculate correlations
cor_matrix <- cor(clean_data)

# 2. Plot the matrix
corrplot(cor_matrix, method = "color", addCoef.col = "black", 
         tl.col = "black", number.cex = 0.7, 
         title = "Heart Disease Correlation Matrix")


# In R, we can use the 'scale' function directly
# We scale everything EXCEPT the target column (the 14th column)
scaled_features <- scale(clean_data[, -14])

# Combine the scaled features back with the target
final_data <- as.data.frame(cbind(scaled_features, target = clean_data$target))

head(final_data)


# Using the caret library
set.seed(42) 

# Create index for 80% training data
trainIndex <- createDataPartition(final_data$target, p = .8, list = FALSE)

train_set <- final_data[trainIndex,]
test_set  <- final_data[-trainIndex,]

print(paste("Training set:", nrow(train_set)))
print(paste("Testing set:", nrow(test_set)))


# 1. Train the model (family='binomial' makes it Logistic Regression)
log_model <- glm(target ~ ., data = train_set, family = "binomial")

# 2. Predict on the test set (gives probabilities)
prob <- predict(log_model, test_set, type = "response")

# 3. Convert probabilities to 0 or 1
pred <- ifelse(prob > 0.5, 1, 0)

# 4. Check accuracy
mean(pred == test_set$target)

# Using the caret library's powerful confusionMatrix function
results <- confusionMatrix(as.factor(pred), as.factor(test_set$target))

# Print the full table and statistics
print(results)

# To see just the table:
print(results$table)

# 1. Extract the table from the results
cm_table <- as.data.frame(results$table)

# 2. Plot using ggplot2 for a professional look
ggplot(data = cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", size = 8) +
  scale_fill_gradient(low = "white", high = "#357ABD") +
  labs(title = "Confusion Matrix: Heart Disease Prediction (R)",
       subtitle = paste("Accuracy:", round(results$overall['Accuracy'], 4)*100, "%"),
       x = "Actual Health Status (0=Healthy, 1=Sick)",
       y = "Model Prediction") +
  theme_minimal()
