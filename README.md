# Berkeley_ML_PA17_1_Assignment
# Practical Application III: Comparing Classifiers

# Link to the Jupyter Notebook: 

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
I did exploratory data analysis (EDA) and generated some plots to understand the data better for feature selection. Below are a few of the figures that are relavent. 
<img width="736" height="504" alt="image" src="https://github.com/user-attachments/assets/6fdb902f-8c7d-47e3-8152-bd273cdc81ae" />
Fig. 1. Class balance

<img width="665" height="488" alt="image" src="https://github.com/user-attachments/assets/4f73b00e-7b7a-4861-85f8-96ba921f8c27" />
Fig. 2. Histogram: age
