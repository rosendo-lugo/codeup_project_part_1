# Standard imports
import pandas as pd
import numpy as np

# Graph imports
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import plot_tree

# Import to ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Stats import
import scipy.stats as stats

# My imports
import wrangle as w
import prepare as p
import acquire as a


# ----------------Question 1-------------------------------
# This shows the distribution between the customers that churn vs the ones that don't. 
# In the following countplot it show us how imbalance in customers that churn.
def get_q_one(train):
    sns.countplot(x=train.churn, data=train)
    plt.title('Distribution of Customer Churn')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    return plt.show()
    
# ----------------Question 2-------------------------------
def get_q_two(train):
    # This histogram shows the relationship between the tenure and customer churn. 
    plt.figure(figsize=(10,5))
    sns.histplot(x=train.tenure, data=train, hue=train.churn, element='step', bins=30, kde=True)
    plt.title('Distribution of Tenure by Churn')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Count')
    return plt.show()

def get_q_two_chi2(train):
    # A t-test is used to evaluate the significant difference in the mean tenure
    # between customers who churn vs those who don't churn.

    # separate the data into two groups: customers who churned and customers who did not churn
    churn_yes = train[train.churn==1]['tenure']
    churn_no = train[train.churn==0]['tenure']

    # Perform the t-test
    t, p = stats.ttest_ind(churn_yes, churn_no)
    a = .05

    # # Print the results
    # print('T-test Results:')
    # print(f'T-statistic: {t:.2f}')
    # print(f'P-value: {p:.5f}')

    # Evaluate p-value
    if p < a:
        print(f'We reject the null hypothesis\nThere is a significant difference in tenure between customers who churned and customers who did not churn.')
    else:
        print(f'We fail to reject the null hypothesis\nThere is no significant difference in tenure between customers who churned and customers who did not churn.')
    
# ----------------Question 3-------------------------------
def get_q_three_figure(train):
    # A countplot illustrates the relationship between the payment method and churn.
    plt.figure(figsize=(10,5))
    sns.countplot(x=train.payment_type, hue=train.churn, data=train)
    plt.title('Distribution of Churn by Payment Method')
    plt.xlabel('Payment Method')
    plt.ylabel('Count')
    return plt.show()

def get_q_three_chi2(train):
    # Perform the chi-squared test
    payment_churn = pd.crosstab(train['payment_type'], train['churn'])
    chi2, p, dof, expected  = stats.chi2_contingency(payment_churn)
    a = .05

    # Print the results
    # Evaluate p-value
    if p < a:
        print(f'We reject the null hypothesis\nThere is a significant relationship between payment method and churn.')
    else:
        print(f'We fail to reject the null hypothesis\nThere is no significant relationship between payment method and churn.')
# ----------------Question 4-------------------------------
def get_q_four_countplot(train):
    # The relationship between tech support and churn
    sns.countplot(data=train, x=train.tech_support_yes, hue=train.churn)
    plt.title('Customers with Tech Support')
    plt.xlabel('Has tech support')
    plt.ylabel('Count')
    return plt.show()