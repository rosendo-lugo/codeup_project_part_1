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
from scipy.stats import chi2_contingency

# My imports
import wrangle as w
import prepare as p
import acquire as a


# ----------------Question 1-------------------------------
# This shows the imbalance between the customers that churn vs the ones that don't. 
# The following countplot it show us how imbalance in customers that churn.
def get_q_one(train):
    # Calculate the percentage of customers who churned and round it to the nearest integer
    percentage = round(train.churn.astype(int).mean()*100)

    # Copy the train data
    df_train = train.copy()
    
    # Replace 0s with 'No' and 1s with 'Yes' in the 'churn' column
    df_train['churn'] = df_train['churn'].replace({0: 'No', 1: 'Yes'})

    # Create a count plot
    sns.countplot(x=df_train.churn, data=df_train)

    # Set title, x-label and y-label of the plot
    plt.title('Difference between Customer Churn vs Not churn')
    plt.xlabel('Churn')
    plt.ylabel('Count')

    # Add a horizontal line at the median count of customers who churned
    plt.axhline(y=df_train.shape[0] / 2, c='red', linestyle='--', label='Median')

    # Add text labels to show the percentage of customers who churned and who didn't churn
    plt.text(1, df_train.churn.value_counts()[1] / 2, f"{percentage}%", ha="center", va="center")
    plt.text(0, df_train.churn.value_counts()[0] / 2, f"{(100-(percentage))}%", ha="center", va="center")

    # Add a legend to the plot
    plt.legend()

    # Show the plot
    return plt.show()

def get_q_one_countplot2(train):
    # The relationship between gender and churn

    
    # # Replace 0s with 'No' and 1s with 'Yes' in the 'churn' column
    # train['churn'] = train['churn'].replace({0: 'No', 1: 'Yes'})
    sns.countplot(data=train, x=train.gender_male, hue=train.churn)
    plt.title('Does gender affect churn')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    return plt.show()

    
# ----------------Question 2-------------------------------
def get_q_two(train):
    # # Create a figure with two subplots: histogram and box plot
    # fig, axs = plt.subplots(ncols=2, figsize=(15,5))

    # # Histogram subplot
    # sns.histplot(x=train.tenure, data=train, hue=train.churn, element='step', bins=30, kde=True, ax=axs[0])
    # axs[0].set_title('Distribution of Tenure by Churn')
    # axs[0].set_xlabel('Tenure (months)')
    # axs[0].set_ylabel('Count')
    # Copy the train data
    df_train = train.copy()
    
    # Replace 0s with 'No' and 1s with 'Yes' in the 'churn' column
    df_train['churn'] = df_train['churn'].replace({0: 'No', 1: 'Yes'})
    
    # Box plot subplot
    # sns.boxplot(x='churn', y='tenure', data=train, ax=axs[1])
    sns.boxplot(x=df_train.churn, y=df_train.tenure, data=df_train)
    plt.title('Box Plot of Tenure by Churn')
    plt.xlabel('Churn')
    plt.ylabel('Tenure (months)')

    # Show the plot
    return plt.show()
    

def get_q_two_t_test(train):
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
        print(f'p-value {p}')
        print(f"We reject the null hypothesis\n.")
    else:
        print(f'p-value {p}')
        print(f"We fail to reject the null hypothesis\n.")
    
# ----------------Question 3-------------------------------
def get_q_three_figure(train):
    # A countplot illustrates the relationship between the payment method and churn.
    plt.figure(figsize=(10,5))
    sns.countplot(x=train.payment_type, hue=train.churn, data=train)
    plt.title('Distribution of Churn by Payment Method')
    plt.xlabel('Payment Method')
    plt.ylabel('Count')
    return plt.show()

def get_q_three_churn_rates(train):
    # Calculate churn rates by payment type
    payment_churn = train.groupby('payment_type')['churn'].mean().reset_index()

    # Separate electronic check from other payment types
    electronic_check = payment_churn[payment_churn['payment_type'] == 'Electronic check']['churn'].values[0]
    other_payment = payment_churn[payment_churn['payment_type'] != 'Electronic check']['churn'].mean()

    # Calculate the difference in churn rates
    diff = electronic_check - other_payment

    # Plot the results
    plt.figure(figsize=(10,5))
    plt.bar(payment_churn['payment_type'], payment_churn['churn'], color=np.where(payment_churn['payment_type'] == 'Electronic check', 'orange', 'blue'))
    plt.axhline(y=other_payment, color='blue', linestyle='--')
    plt.axhline(y=electronic_check, color='orange', linestyle='--')
    plt.annotate(f'Churn rate: {other_payment:.2%}', xy=(0, other_payment), xytext=(0.1, other_payment + 0.02), fontsize=12)
    plt.annotate(f'Churn rate: {electronic_check:.2%}', xy=(1, electronic_check), xytext=(1.1, electronic_check + 0.02), fontsize=12)
    plt.annotate(f'Difference: {diff:.2%}', xy=(0.5, (electronic_check + other_payment)/2), xytext=(0.4, (electronic_check + other_payment)/2 + 0.03), fontsize=12)
    plt.title('Churn Rates by Payment Type')
    plt.xlabel('Payment Type')
    plt.ylabel('Churn Rate')
    plt.ylim(0, 0.5)
    return plt.show()


def get_q_three_demographic_group(train):
    # Subset the data to customers who pay by electronic check and have churned
    elec_check_churned = train[(train['payment_type'] == 'Electronic check') & (train['churn'] == 1)]

    # Total number of customers that churned while paying with electronic check
    total_churned = elec_check_churned['churn'].sum()

    # Demographic analysis with churn ratio
    demographic_analysis = elec_check_churned.groupby(['gender_male', 'senior_citizen', 'partner_yes', 'dependents_yes']).agg({'churn': ['count', 'sum']})
    demographic_analysis.columns = ['total_customers', 'churned_customers']
    demographic_analysis['churn_ratio'] = demographic_analysis['churned_customers'] / total_churned * 100
    demographic_analysis['churn_ratio'] = demographic_analysis['churn_ratio'].apply(lambda x: "{:.1f}%".format(x))
    demographic_analysis = demographic_analysis.reset_index()
    demographic_analysis = demographic_analysis[['gender_male', 'senior_citizen', 'partner_yes', 'dependents_yes', 'total_customers', 'churn_ratio']]

    da_df = pd.DataFrame(demographic_analysis)
    da_df = da_df.sort_values(by='total_customers', ascending=False)
    return da_df

def get_q_three_dg_plot(train):
    # Subset the data to customers who pay by electronic check and have churned
    elec_check_churned = train[(train['payment_type'] == 'Electronic check') & (train['churn'] == 1)]

    # Total number of customers that churned while paying with electronic check
    total_churned = elec_check_churned['churn'].sum()

    # Demographic analysis with churn ratio
    demographic_analysis = elec_check_churned.groupby(['gender_male', 'senior_citizen', 'partner_yes', 'dependents_yes']).agg({'churn': ['count', 'sum']})
    demographic_analysis.columns = ['total_customers', 'churned_customers']
    demographic_analysis['churn_ratio'] = demographic_analysis['churned_customers'] / total_churned * 100
    demographic_analysis['churn_ratio'] = demographic_analysis['churn_ratio'].apply(lambda x: "{:.1f}%".format(x))
    d_a_df = pd.DataFrame(demographic_analysis)


    # Sort the bars by churn ratio
    d_a_df = d_a_df.sort_values('churn_ratio', ascending=False)

    # Convert churn_ratio to a numeric data type
    d_a_df['churn_ratio'] = d_a_df['churn_ratio'].apply(lambda x: float(x.replace('%', '')))

    # Plot the chart
    # Create a horizontal bar chart
    d_a_df['churn_ratio'].plot(kind='barh', figsize=(8,6))
    plt.title('Churn Ratio by Demographic Group')
    plt.xlabel('Churn Ratio')
    plt.ylabel('Demographic Group')

    # Show the plot
    return plt.show()

def get_q_three_dg_payment_amount(elec_check_churned):
    # Payment amount analysis
    plt.figure(figsize=(10,5))
    sns.kdeplot(elec_check_churned['monthly_charges'], shade=True, color='red')
    plt.title('Distribution of Monthly Charges for Electronic Check Customers Who Churned')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Density')
    return plt.show()

def get_q_three_dg_payment_frequency(elec_check_churned):
    # Payment frequency analysis
    plt.figure(figsize=(10,5))
    sns.countplot(x='paperless_billing_yes', hue='churn', data=elec_check_churned)
    plt.title('Distribution of Churn by Paperless Billing for Electronic Check Customers')
    plt.xlabel('Paperless Billing')
    plt.ylabel('Count')
    return plt.show()

def get_q_three_dg_payment_user(elec_check_churned):
    sns.countplot(x='gender_male', hue='senior_citizen', data=elec_check_churned)
    plt.title('Demographic Distribution of Electronic Check Users Who Churned')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    return plt.show()

def get_q_three_dg_payment_bill(elec_check_churned):
    sns.scatterplot(x='monthly_charges', y='tenure', hue='paperless_billing_yes', data=elec_check_churned)
    plt.title('Payment Amount and Frequency Analysis for Electronic Check Users Who Churned')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Tenure')
    return plt.show()
    
def get_q_three_chi2(train):
    # Perform the chi-squared test
    payment_churn = pd.crosstab(train['payment_type'], train['churn'])
    chi2, p, dof, expected  = stats.chi2_contingency(payment_churn)
    a = .05

    # Print the results
    # Evaluate p-value
    if p < a:
        print(f'p-value {p}')
        print(f'We reject the null hypothesis\n')
    else:
        print(f'p-value {p}')
        print(f'We fail to reject the null hypothesis\n')
        

# ----------------Question 4-------------------------------
def get_q_four_countplot(train):
    # The relationship between tech support and churn
    df_train = train.copy()
    
    # Replace 0s with 'No' and 1s with 'Yes' in the 'churn' column
    df_train['churn'] = df_train['churn'].replace({0: 'No', 1: 'Yes'})
    sns.countplot(data=df_train, x=train.tech_support_yes, hue=train.churn)
    plt.title('Customers with Tech Support')
    plt.xlabel('Has tech support')
    plt.ylabel('Count')
    return plt.show()

def get_q_four_chi2(train):
    # Create a contingency table
    contingency_table = pd.crosstab(train.tech_support_yes, train.churn)

    # Run chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    a = .05

    # Print the results
    # Evaluate p-value
    if p < a:
        print(f'p-value {p}')
        print(f'We reject the null hypothesis\n.')
    else:
        print(f'p-value {p}')
        print(f'We fail to reject the null hypothesis\n.')