!pip3 install -U ucimlrepo  pandas matplotlib seaborn scikit-learn

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# metadata 
print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables) 

# Convert any value > 0 into 1 (Heart Disease)
# If it's 0, keep it 0 (Healthy)
y_binary = (y > 0).astype(int)

# Check the new count
print("Heart Disease (1) vs Healthy (0):")
print(y_binary.value_counts())

# 1. Check for missing values in X
print("Missing values before cleaning:")
print(X.isnull().sum())

# 2. Fill the missing values with the median of the column
X_clean = X.fillna(X.median())

# 3. Verify they are gone
print("Missing values after cleaning:")
print(X_clean.isnull().sum())


import seaborn as sns
import matplotlib.pyplot as plt

# 1. Combine features and target into one table for the chart
df_plot = X_clean.copy()
df_plot['target'] = y_binary

# 2. Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_plot.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: Clinical Features vs Heart Disease")
plt.show()


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 1. Initialize the Scaler
scaler = StandardScaler()

# 2. Scale the features (X_clean)
# We don't scale 'y' because it's just 0 and 1
X_scaled = scaler.fit_transform(X_clean)

# 3. Convert back to a DataFrame so it's easy to read
X_scaled_df =pd.DataFrame(X_scaled, columns=X.columns)

print("First row of scaled data:")
print(X_scaled_df.iloc[0])

from sklearn.model_selection import train_test_split

# We split the Scaled Features (X) and the Binary Target (y)
# test_size=0.2 means 20% goes to testing
# random_state=42 ensures the split is the same every time you run it
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)} patients")
print(f"Testing set size: {len(X_test)} patients")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Initialize the Model
model = LogisticRegression()

# 2. Train the model using the Training Data
model.fit(X_train, y_train.values.ravel())

# 3. Make predictions on the Test Data (The Exam)
y_pred = model.predict(X_test)

# 4. Check the score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report

# 1. Create the matrix
cm = confusion_matrix(y_test, y_pred)

# 2. Print a detailed report (Precision, Recall, F1-Score)
print("--- Detailed Performance Report ---")
print(classification_report(y_test, y_pred))

# 3. Plot the matrix visually
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Heart Disease Prediction')
plt.show()
