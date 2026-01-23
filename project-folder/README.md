This is a submission for program AIMLCZG565 assignment 2

##### 1. Problem Statement

Given the degree of education and  years of experience, whether the individual given dataset will earn over 50k USD in annual salary. This will be a binary classification problem. I will use multipe classifiers and compare against metrics to see which one works best for this. 

#### 2. Dataset description
This dataset is the adult income census dataset from https://www.kaggle.com/datasets/uciml/adult-census-income 

There are 14 features with 32,561 instances. 

*Here is a description of the features* 
| Feature | datatype | description |
| ---- | ---- | ---- |
| age | int | age of person |
| workingclass | string | 8 values and some unknowns to indicate employement status |
| fnlwgt | int | final weight column |
| education | string | different levels of education | 
| education.num | int | numeric value assigned to each of these education status | 
| marital.status | string | marital status selected out of 7 different values | 
| occupation | string | brief description of occupation level |
| relationship | string | brief description of relationship status | 
| race | string | ethnicity |
| sex | string | gender as of 1994 census data source |
| capital.gain | float | currency field | 
| capital.loss | float | currency field |
| hours.per.week | int | number of hours worked per week | 
| native.country | string | whether native to United States or from other country |

Target variable is *income*

#### 3. Model Metric Comparision

**Model comparision metrics**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Logistic Regression | 85.31% | 90.24% | 84.67% | 85.30% | 84.76% | 57.58% |
| Decision Tree | 81.53% | - | 81.70% | 81.53% | 81.61% | 49.95% | 
| kNN | 83.14% | - | 82.58% | 83.14% | 82.78% | 52.18% |
| Naive Bayes | 53.66% | - | 81.00% | 53.66% | 55.38% | 32.2% | 
| Random Forest (Ensemble) | 85.12% | - | 84.53% | 85.12% | 84.68% | 57.36% | 
| XGBoost | 86.67% | 92.23% | 86.21% | 86.67% | 82.27% | 61.93% |

**Commentary on model performance**

*Here's what all these metrics mean and how they are to be used:*

**Accuracy:** 
The number of true positives to the total data in the validation data. Most useful when data is well balanced. Imbalance can lead to inaccuracies. 

**Precision:**
Ratio of true positives predicted to the total data. This is the metric to pick when impact of false positives is high. 

**Recall:**
Tracks how many of postiives detected are true positives. Important when true positive when predicted as negative can have huge impact. 

**F1-Score:**
Harmonic mean of precision and recall. Useful when there is imbalance in classification and useful to balance out impact of false positive and missed true positive. 

**Area under ROC curve AOC:**
Measure of model's ability to rank positives higher than negatives. 

**Matthew's correlation coefficient MCC:**
Consolidated consideration of true positives, true negatives, false positives, and false negatives. Balanced assessment over all predicted metrics for imbalanced class. 

*Here's my commentary on the model performance*

| ML Model Name | Observation about model performance | 
| ---- | ---- |
| Logistic Regression | Good overall levels of accuracy and recall. MCC score shows resilient handling of imbalances | 
| Decision Tree | High on accruacy, precision, recall but lower overall on MCC score means sensitive to class imbalance and maybe over-fitting | 
| kNN | close to logistic regression, but additional features created by one hot encoding will degreade results | 
| Naive Bayes | Poorly performing, indicating that naive bayes assumption of independence of features does not hold true | 
| Random Forest | Being an ensemble version of decision tree, it performs much better showing resilience to class imbalance |
| XGBoost | Best overall model shining across all metrics, able to identify true positives accurately, reducing false positives, while handling imbalances |

