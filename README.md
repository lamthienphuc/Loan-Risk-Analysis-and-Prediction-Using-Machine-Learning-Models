# Loan-Risk-Analysis-and-Prediction-Using-Machine-Learning-Models

# Data Collection and Preprocessing
Dataset was taken from Kaggle website, where we can explore, analyze, and share quality data. Dataset name: “Loan_data”, publicly available data from LendingClub.com. Lending Club connects
people who need money (borrowers) with people who have money (investors). Hopefully, as an
investor you would want to invest in people who showed a profile of having a high probability of
paying you back. We will use lending data from 2007-2010 and be trying to classify and predict whether or not the
borrower paid back their loan in full.
![image](https://github.com/user-attachments/assets/f57c9cc2-3a80-4f9e-b1e3-8a1bec14fb5e)

Number of Entries: 9578

Number of Features: 14

Feature Details
![image](https://github.com/user-attachments/assets/ca4c9c1d-0d71-4139-9973-cb30ccf50ef7)

# Data Cleaning
There are no missing values in the dataset
Inspect the 'purpose' Column
['debt_consolidation' 'credit_card' 'all_other' 'home_improvement'
'small_business' 'major_purchase' 'educational']
This line prints the unique values in the 'purpose' column to ensure it contains valid values. This
helps in understanding the different categories present in the column. Clean the 'purpose' Column
If the 'purpose' column has concatenated values (e.g., "debt consolidation"), this line splits the
values and keeps only the first part. This is done using the apply function with a lambda
function that splits the string by spaces and takes the first element.
One-Hot Encoding for the 'purpose' Column
![image](https://github.com/user-attachments/assets/0526e2c2-c4e0-4155-9d0d-df361ff8a17d)
Concatenate Encoded Features with the Original Dataset
![image](https://github.com/user-attachments/assets/9d886f4b-d8b5-4443-bfb4-0a0589c2f6b8)
# Feature Egineering
Income to Debt Ratio
Formula:
Income-to-Debt Ratio (ID Ratio) = log(Annual Income) / (Debt-to-Income Ratio (DTI) + 1)
Creates a new feature called income_to_debt_ratio by dividing the logarithm of the annual
income by the debt-to-income ratio (plus 1 to avoid division by zero). This ratio helps in
understanding the borrower's ability to manage debt relative to their income, which is crucial
for risk assessment.

FICO Score Binning
Bins the FICO scores into categorical ranges: 'Poor', 'Fair', 'Good', and 'Excellent'. Binning
helps in simplifying the continuous FICO score into discrete categories, making it easier to
interpret and analyze the risk associated with different credit score ranges.

# Exploratory Data Analysis (EDA)
These descriptive statistics provide a comprehensive overview of the dataset, highlighting the
central tendencies and variations in the key numerical features.
![image](https://github.com/user-attachments/assets/db47ed57-291f-4d20-a406-2b4b86e05de4)
![image](https://github.com/user-attachments/assets/cfa46622-e528-4912-b703-89e9904c3c52)
# Data Visualizations
![image](https://github.com/user-attachments/assets/f6162a7f-a805-4e89-8698-bc30ae17ad33)
Insights
Debt Consolidation Dominance: The high count for debt consolidation loans suggests that
many borrowers are looking to manage their existing debts more effectively. 
Credit Card Debt: The significant number of loans for credit card debt indicates that credit
card debt is a common financial issue among borrowers.
Diverse Purposes: The presence of the "all_other" category with a substantial count shows that
there are various other reasons borrowers take loans, which are not specifically categorized.

![image](https://github.com/user-attachments/assets/d81890b2-242e-41cc-b11f-9db27ed51534)

Common Interest Rates: The most frequent interest rates are around 12%, which could be
considered the average or typical interest rate for loans in this dataset.
Loan Affordability: The concentration of loans with interest rates between 10% and 14%
suggests that these rates are common and possibly more affordable for borrowers.
High-Interest Loans: There are fewer loans with very high interest rates (above 16%), indicating that such loans are less common or less preferred by borrowers. 
Understanding this distribution is crucial for financial institutions to set competitive interest
rates and for borrowers to understand the typical rates they might encounter.

![image](https://github.com/user-attachments/assets/391e94b0-2d0e-4220-a51b-80e1f9ca10bf)
This bar chart provides a clear overview of the distribution of FICO scores among borrowers. It
highlights that most borrowers have FICO scores in the "Good" and "Fair" categories, indicating relatively strong credit histories.

![image](https://github.com/user-attachments/assets/fa44194f-1d1e-4afc-8a9a-965266178567)

Positive Correlations: Features with a positive correlation to "not.fully.paid" might indicate factors
associated with an increased risk of loan default. For example, a high credit utilization ratio
(revol.util) could be positively correlated with "not.fully.paid," suggesting borrowers with higher
credit card balances are more likely to default. 

Negative Correlations: Features with a negative correlation to "not.fully.paid" might indicate
factors that mitigate default risk. For instance, a high FICO score (fico) could be negatively
correlated with "not.fully.paid," implying borrowers with strong creditworthiness are less likely to
default.
# Model Development
Machine Learning Algorithms
- Random Forest:
Concept: This is an ensemble learning method that combines multiple decision trees to create a
more robust and accurate predictor. How it Works: Random forests train individual decision trees on random subsets of data and
features. Each tree predicts an outcome, and the final prediction is based on the majority vote
(classification) or average (regression) of all the individual trees.
- K-Nearest Neighbors (KNN):
Concept: This is a non-parametric, lazy learning algorithm. It classifies new data points based
on the labels of their k nearest neighbors in the training data. How it Works: KNN identifies the k data points in the training set that are closest to the new
data point based on a chosen distance metric (e.g., Euclidean distance). The new data point is
assigned the majority class label (classification) or average value (regression) of its k neighbors.
- Gradient Boosting:
Concept: This is an ensemble learning method that combines multiple weak learners (typically
decision trees) in a sequential way. Each subsequent learner focuses on improving the errors
made by the previous ones.
![image](https://github.com/user-attachments/assets/53bd1114-1524-4f18-b134-62fb9c5774f9)

How it Works: Gradient boosting trains a sequence of models, where each model learns from
the residuals (errors) of the previous model. This sequential approach allows the model to
progressively improve its predictions.
- Logistic Regression:
Concept: This is a statistical method used for binary classification problems. It estimates the
probability of a data point belonging to a specific class (e.g., loan default or not default) by
fitting a sigmoid function to the data.
![image](https://github.com/user-attachments/assets/b479be84-bdda-40ea-aa7c-9546dfcc8aba)

How it Works: Logistic regression analyzes the relationship between independent variables
(features) and the dependent variable (binary outcome) using a logistic function. The model
outputs a probability value between 0 (not likely) and 1 (highly likely) for the data point
belonging to the positive class.
- Naive Bayes:
Concept: This is a probabilistic classification method based on Bayes' theorem. It assumes
independence between features (features are not correlated) and predicts the class with the
highest probability based on this assumption.
![image](https://github.com/user-attachments/assets/265f0198-6887-4264-8206-6c3af095a400)

How it Works: Naive Bayes calculates the probability of each class given the features of a new
data point. It then uses Bayes' theorem to determine the most probable class for the data point. Strengths: Simple to implement and computationally efficient, works well for certain types of
problems with categorical features.
# Model Selection
Model Rankings based on ROC-AUC:
1. Random Forest: 0.9978935799981683
2. K-Nearest Neighbors: 0.9732413428173112
3. Gradient Boosting: 0.816063742100925
4. Logistic Regression: 0.6874703625688149
5. Naive Bayes: 0.6323147215353461
   
Model Rankings based on accuracy:
1. Random Forest: 0.9796450939457203
2. Gradient Boosting: 0.8538622129436325
3. K-Nearest Neighbors: 0.7693110647181628
4. Logistic Regression: 0.6450939457202505
5. Naive Bayes: 0.5715031315240083

Based on these models comparison, top three of the board will be select to analyze and forecast is
Random Forest, Gradient Boosting, K-Nearest Neighbors
# Handling class imbalance
![image](https://github.com/user-attachments/assets/aadc50e0-22b7-40a9-8827-97991af5e11f)

Class Weight Adjustment

Class weight adjustment is a technique used to handle class imbalance by assigning different
weights to classes during the training of a machine learning model. This method helps the model
pay more attention to the minority class by penalizing misclassifications of minority class instances
more heavily than those of the majority class.
![image](https://github.com/user-attachments/assets/2bee0744-eaa8-4dc5-b757-e7d55b3f1e81)
![image](https://github.com/user-attachments/assets/87be4ccd-6558-4a2f-bfe0-2172cff726fd)
![image](https://github.com/user-attachments/assets/f71d827a-38a9-47e1-a233-7ac2497785aa)

# Hyperparameter tuning

What are Hyperparameters?

Hyperparameters are the parameters of a machine learning model that are not learned from the data
but are set before the training process begins. They control the behavior of the training algorithm
and the structure of the model.
Why Tune Hyperparameters?
Hyperparameter tuning is crucial because the performance of a machine learning model can be
significantly affected by the choice of hyperparameters. Proper tuning can lead to:
Improved Accuracy: Better generalization to unseen data.
Reduced Overfitting: Prevents the model from learning noise in the training data. 
Optimized Training Time: Efficient use of computational resources. 

K-Nearest Neighbors Performance:
precision recall f1-score support
0 1.00 1.00 1.00 1611
1 1.00 1.00 1.00 305
accuracy 1.00 1916
macro avg 1.00 1.00 1.00 1916
weighted avg 1.00 1.00 1.00 1916
ROC-AUC: 1.0

# Model Evaluation

Confusion Matrix
A confusion matrix is a table used to describe the performance of a classification model. It shows
the counts of true positive, true negative, false positive, and false negative predictions. 
        Predicted Positive          Predicted Negative
Actual Positive True Positive (TP) False Negative (FN)
Actual Negative False Positive (FP) True Negative (TN)

![image](https://github.com/user-attachments/assets/6166010e-13cf-40c1-9a55-6c8ecb0334f1)

![image](https://github.com/user-attachments/assets/4c006665-7046-48da-934a-5fa49d380b6e)

# ROC Curves
![image](https://github.com/user-attachments/assets/6aa86052-c30b-469b-a600-17c91b5aef06)

Key Components of the ROC Curve:
True Positive Rate (TPR):
1. Also known as Sensitivity or Recall.
2. It is plotted on the Y-axis.
3. TPR = TP / (TP + FN), where TP is True Positives and FN is False Negatives.
4. It represents the proportion of actual positives that are correctly identified by the
model.
False Positive Rate (FPR):
1. It is plotted on the X-axis.
2. FPR = FP / (FP + TN), where FP is False Positives and TN is True Negatives.
3. It represents the proportion of actual negatives that are incorrectly identified as
positives by the model.
Diagonal Line (Random Classifier):
1. The diagonal dashed line represents the performance of a random classifier.
2. It has an Area Under the Curve (AUC) of 0.5, indicating no discrimination ability
between positive and negative classes.
Area Under the Curve (AUC):
The Area Under the ROC Curve (AUC) is a single scalar value that summarizes the
performance of the classifier. AUC ranges from 0 to 1.

 AUC = 0.5: The model has no discrimination ability (equivalent to random guessing).
 
 AUC < 0.5: The model performs worse than random guessing.
 
 AUC > 0.5: The model has some discrimination ability, with higher values indicating
better performance. 

Summary:

The ROC curve provides a visual comparison of the performance of different classifiers.
The KNN model has the highest AUC (0.59), indicating it performs the best among the three
models in distinguishing between positive and negative classes.
The Gradient Boosting model has an AUC of 0.58, indicating it performs slightly worse than
KNN but better than Random Forest.
The Random Forest model has the lowest AUC (0.54), indicating it has the least discrimination
ability among the three models.
# Model Interpretation
Feature importance refers to techniques that assign a score to input features based on how useful
they are at predicting a target variable. Understanding feature importance can help in:
Model Interpretation: Understanding which features are most influential in making predictions. 
Feature Selection: Reducing the number of input features to improve model performance and
reduce overfitting. 
Insight Generation: Gaining insights into the underlying data and the relationships between
features and the target variable.
![image](https://github.com/user-attachments/assets/4e6c00c5-580f-4b52-b857-bf7f8b4f746f)
![image](https://github.com/user-attachments/assets/5d9ba6e2-0c86-4026-9e3c-3353a627f1b3)
Interpretation:
FICO Score: The FICO score is the most influential feature, suggesting that it is a strong
predictor of the target variable. This makes sense as the FICO score is a widely used
measure of creditworthiness.
Interest Rate and Installment: These financial metrics are also highly important, indicating that they play a significant role in the model's predictions.  Credit History and Income: Features related to credit history (e.g., days with a credit line)
and income (e.g., log of annual income) are also important, reflecting their relevance in
financial decision-making.
Loan Purpose: The purpose of the loan (e.g., debt consolidation, credit card) has varying
levels of importance, with debt consolidation being more important than others like
educational purposes.
Less Important Features: Features like the number of public records and delinquencies in
the past two years have lower importance, suggesting they are less influential in the model's
predictions.

Summary of Predicted Results (Random Forest):

Actual Predicted_RF Predicted_RF_Prob

8558     0 1         0.67

4629     0 0         0.17

1383     1 0         0.14

8142     0 0         0.18

1768     0 0         0.06

Summary of Predicted Results (K-Nearest Neighbors):

Actual Predicted_KNN Predicted_KNN_Prob

8558     0 1             1.0

4629     0 0             0.0

1383     1 0             0.0

8142     0 0             0.0

1768     0 0             0.0

Summary of Predicted Results (Gradient Boosting):

Actual Predicted_GB Predicted_GB_Prob

8558     0 1           0.687005

4629     0 0           0.455550

1383     1 0           0.276111

8142     0 0           0.441451

1768     0 0            0.250791

The Random Forest model has some misclassifications. For instance, it incorrectly predicts that
instance 8558 will fully pay the loan with a relatively high probability (0.67). 
It correctly predicts most of the instances with lower probabilities for not fully paid loans. 

The KNN model also has misclassifications. For instance, it incorrectly predicts that instance 8558
will fully pay the loan with a probability of 1.0. It tends to give extreme probabilities (0.0 or 1.0), which might indicate overconfidence in its
predictions.

The Gradient Boosting model has some misclassifications as well. For instance, it incorrectly
predicts that instance 8558 will fully pay the loan with a probability of 0.687005. It provides more nuanced probabilities compared to KNN, indicating a more calibrated prediction.

# Predictions and Future work

![image](https://github.com/user-attachments/assets/b67157ed-066f-4cfe-b9eb-73c0b8a38148)

Because FICO score and Interest rate is a most importance features and the predictive models—
KNN, Gradient Boosting, and Random Forest—provide varying outcomes based on different
borrower profiles, showcasing their distinct decision-making mechanisms.Here's a breakdown of
their predictions and some potential conclusions:

Predictions Summary

FICO Score: 700, Interest Rate: 0.125

KNN: Will fully pay the loan. 

Gradient Boosting: Will not fully pay the loan. 

Random Forest: Will fully pay the loan.

FICO Score: 600, Interest Rate: 0.2

KNN: Will fully pay the loan. 

Gradient Boosting: Will not fully pay the loan. 

Random Forest: Will not fully pay the loan.

FICO Score: 800, Interest Rate: 0.3

KNN: Will fully pay the loan. 

Gradient Boosting: Will not fully pay the loan. 

Random Forest: Will fully pay the loan.

Model Differences:
KNN consistently predicts that borrowers will fully pay the loan across all profiles. Gradient Boosting consistently predicts that borrowers will not fully pay the loan across all profiles. Random Forest predictions are similar to KNN for profiles with higher FICO scores but align with
Gradient Boosting for lower FICO scores and higher interest rates. 

FICO Score Impact:
Higher FICO scores generally indicate a better credit history and a higher likelihood of loan
repayment. Gradient Boosting seems more conservative, predicting non-repayment even for higher FICO
scores, possibly due to a higher sensitivity to interest rates or other factors. 

Interest Rate Impact:
Higher interest rates are generally associated with higher risk, which can affect the likelihood of
loan repayment. KNN and Random Forest might weigh FICO scores more heavily than interest rates, leading to
more optimistic predictions for high-FICO-score borrowers despite high interest rates. Gradient Boosting might consider the interest rate more heavily, leading to a pessimistic prediction
even for high-FICO-score borrowers with high interest rates. 

In summary, while KNN and Random Forest are more optimistic about loan repayment, Gradient
Boosting is more conservative. The differences highlight the importance of model selection and the
need for potentially using ensemble methods or further model refinement for consistent and reliable
predictions.
