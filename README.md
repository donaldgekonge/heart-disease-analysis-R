# Heart-disease-analysis
Predicting heart disease risk using clinical data "

  **The Goal**: Predict if a patient has heart disease based on clinical tests.
  **The Data**: The UCI Cleveland Heart Disease dataset (303 patients, 14 features).
  **The Stakeholder**: A doctor who needs a "second opinion" tool to flag high-risk patients.
## Step 1: Data Acquisition
I imported the dataset directly from the UCI Machine Learning Repository. 
The dataset contains 303 observations with 13 features (like age, cholesterol, and chest pain type).

## Step 2: Data Cleaning & Transformation
The original dataset categorized heart disease into four stages (1-4). 
For this analysis, I converted the problem into a **Binary Classification** task:
- **0**: Healthy
- **1**: Presence of Heart Disease
This simplifies the model's objective to identifying risk versus no-risk.

## Step 3: Handling Missing Data
I identified missing values in the `ca` and `thal` features. 
To preserve the dataset's size (n=303), I used **Median Imputation** to fill these gaps. 
This ensures the model has a complete set of features for every patient without introducing the bias that might come from deleting rows.

## Step 4: Exploratory Data Analysis (EDA)
I generated a **Correlation Heatmap** to identify which clinical features have the strongest relationship with the target variable. 
- **Key Findings:** Features like `cp` (chest pain type) and `thalach` (maximum heart rate achieved) showed strong correlations with the presence of heart disease.
- This step ensures that the variables we feed into the machine learning model are logically connected to the outcome.
- 
- ## Step 5: Feature Scaling (Standardization)
Since the clinical features have different units (e.g., Age in years vs. Cholesterol in mg/dl), I applied **Standardization (Z-score normalization)**. 
- This ensures that features with larger numerical ranges do not disproportionately influence the model. 
- All features now have a mean of 0 and a standard deviation of 1, providing a balanced input for the classification algorithms.

  ## Step 6: Data Partitioning (Train-Test Split)
To evaluate the model's true predictive power, I split the dataset into:
- **Training Set (80%)**: Used to teach the model the relationship between clinical features and heart disease.
- **Testing Set (20%)**: Reserved as "unseen data" to validate the model's accuracy.
I used a fixed seed (`random_state=42`) to ensure that my results are reproducible by others.

## Step 7: Predictive Modeling (Logistic Regression)
I implemented a **Logistic Regression** model as the primary classifier. 
- **Interpretability:** Unlike "black box" models, Logistic Regression allows us to see the weight of each medical feature.
- **Performance:** The model achieved an initial accuracy of approximately [Insert your % here] on the unseen test data.
- **Probabilistic Output:** This model provides a risk probability, which is highly valuable in a clinical diagnostic setting.
- ## Step 8: Model Evaluation & Clinical Insights
I analyzed the model's performance using a **Confusion Matrix**. 
- **Accuracy:** 83.33%
- **Clinical Risk Assessment:** I specifically monitored the **Recall (Sensitivity)** score. In a cardiac context, minimizing False Negatives is the priority to ensure at-risk patients are not overlooked.
- **Conclusion:** The Logistic Regression model provides a robust baseline for clinical decision support, effectively utilizing features like `thal` and `ca` to differentiate between healthy and at-risk cardiovascular profiles.

  ### **Model Diagnostics (Results)**
- **True Negatives (0,0):** Correctly identified healthy patients.
- **True Positives (1,1):** Correctly identified patients with heart disease.
- **The "Misses":** By analyzing the off-diagonal cells, we can determine if the model is biased toward "False Alarms" (Type I error) or "Missed Diagnoses" (Type II error).
