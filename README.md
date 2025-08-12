# Berkeley_ML_PA17_1_Assignment
# Practical Application III: Comparing Classifiers

## Link to the Jupyter Notebook: https://github.com/mithran-gheewiz/Berkeley_ML_PA17_1_Assignment/blob/main/M11_Mithran_Menon_Bank_Data.ipynb

## Introduction 
In this practical application, my goal is to compare the performance of the classifiers, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. I utilized a dataset related to marketing bank products over the telephone. The dataset comes from the UCI Machine Learning repository link. The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. 

## 1: Understanding the Data
The dataset contains: 
bank-additional-full.csv → the main dataset (41,188 rows, 20 features + target y)
bank-additional.csv → a 10% sample of the above
bank-additional-names.txt → the description file with attribute details

The dataset collected is related to 17 marketing campaigns that occurred between May 2008 and November 2010, corresponding to a total of 79354 contacts. During these phone campaigns, an attractive long-term deposit application, with good interest rates, was offered. 

## 2: Read in the Data and 3: Understanding the Features
Using pandas, the data from bank-additional-full.csv was read in. There are 21 columns including the target column y. The description of the columns are shown below:  
### standard attributes
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
### related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
### other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
### social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

### Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

An initial check showed that there appears to be no missing values. Upon further investigation of the data, it was determined that there are placeholders in the data set for missing values called "unknown".
I counted all the unknown values and are shown below for the columns: 
Number of 'unknown' entries per column:
job           330
marital        80
education    1731
default      8597
housing       990
loan          990

## 4: Understanding the Task
Based on what I have explored in the data, the business objective is to predict which customers are most likely to subscribe to a long-term term deposit, so the bank can focus its marketing calls on those most likely to say “yes.” The goal of the model is to predict "yes" so that the bank gets a new or returning customer. 

## 5: Engineering Features and EDA 
I did exploratory data analysis (EDA) and generated some plots to understand the data better for feature selection. 
Below are a few of the figures that are relavent. 

<img width="736" height="504" alt="image" src="https://github.com/user-attachments/assets/6fdb902f-8c7d-47e3-8152-bd273cdc81ae" />
Fig. 1. Class balance

<img width="665" height="488" alt="image" src="https://github.com/user-attachments/assets/4f73b00e-7b7a-4861-85f8-96ba921f8c27" />
Fig. 2. Histogram: age

<img width="709" height="510" alt="image" src="https://github.com/user-attachments/assets/0ae9f35f-a529-446f-bee8-41a391902883" />
Fig. 3. Top 10 Categories - Categorical data

<img width="600" height="533" alt="image" src="https://github.com/user-attachments/assets/c23241df-2604-4f82-a668-00935526191e" />
Fig. 4. Numeric feature Correlation heatmap

The indicators (emp.var.rate, euribor3m, nr.employed, cons.price.idx) are highly correlated with each other — this is expected because they often move together over time.
Previous and Pdays have a moderate positive correlation (0.5), since they both relate to previous marketing contact history. I restricted the dataset to relevant bank info features, handled "unknown", and set up categorical encoding. I separated X and y, identified feature types, and created a ColumnTransformer that one-hot encodes categoricals while keeping numeric features as-is.

## 6: Train/Test split
With your data prepared, split it into a train and test set. I obtained the following: 
Training set shape: (32950, 7)
Test set shape: (8238, 7)

## 7: A Baseline Model
For the baseline model, I used a simple majority class baseline to predict the most frequent class for all cases which is "no". Results are shown below: 
Majority Regression Accuracy: 0.887
Precision_score: Of all the customers predicted as "yes", what fraction actually said "yes".  Precision score: 0
Recall_score: Of all the customers who actually said "yes", what fraction did the model correctly identify. Recall score: 0
Majority class accuracy is high (88.7%) because the dataset is imbalanced, but it had 0 recall for the minority class (yes responses). We need a better model. 

## 8: A Simple Model and 9: Score the Model
I continued with logistic regression. This provides a meaningful starting point that accounts for the features. 
The model also had a high accuracy of 0.887. 
The model predicted all samples as “No”, so it never correctly identified a positive case. This happens because:
The dataset is highly imbalanced (~89% “No”, 11% “Yes”) as shown in the Fig. 1. 
I re-ran the model with class_weight='balanced' on the Logistic Regression to improve recall and precision. The overall accuracy decreased to 0.584. This is not all bad. 
Of all the customers who actually said "yes", now 62.3% the model correctly identify compared to zero from before. This has improved recall. However, precision is still low at ~16 %.

## 10: Model Comparisons
Next I aimed to compare the performance of the Logistic Regression model to the KNN algorithm, Decision Tree, and SVM models. Using the default settings for each of the models, I fit and score each. Also, I compared the fit time of each of the models. The results are shown below:

Table 1. Model comparisons
<img width="461" height="180" alt="image" src="https://github.com/user-attachments/assets/90204daa-22fa-46cb-b56e-e24e581b9344" />

## 11: Improving the Model
I performed hyperparameter tuning & Grid Search CV. The best-performing KNN model used 3 nearest neighbors. Instead of all neighbors voting equally, closer neighbors have more influence on the prediction. The mean F1-Score is 0.135. 

Next, I set up GridSearchCV for all four models, optimize them for F1-score, and compare them using accuracy, precision, recall, and F1.

Table 2: Model comparisons after improving it
<img width="980" height="245" alt="image" src="https://github.com/user-attachments/assets/16a430c2-5364-49f3-92e4-3a4e0275c12a" />

## Model Interpretation
SVM had the highest raw accuracy but classified very low positives. Therefore, it is not useful for this business problem.
KNN performed well in accuracy but extremely poorly in recall (missed ~90% of subscribers).
Decision Tree gave balanced precision/recall but still underperformed logistic regression on recall.
Logistic Regression with class_weight='balanced' produced the highest recall (0.578) and the best F1-score (0.246) among the models. While accuracy was lower, recall is the more important measure here.

## Justification for Final Model
Logistic Regression (balanced) is the preferred final model because:
High Recall. Therefore, it catches more potential subscribers, reducing missed opportunities.
Best F1-score among tuned models → balances precision and recall better than others.
Interpretability: coefficients show which customer attributes influence subscription likelihood, aiding in marketing strategy design.
Simplicity: fast to train, easy to deploy, and less prone to overfitting than deeper trees or complex kernels.









