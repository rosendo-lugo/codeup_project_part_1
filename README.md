# Codeup Project Part I


## Project Description
> To find drivers for customer churn at Telco.

## Project Goal
>- To identify the factors that contribute to customer churn in the Telco dataset.
>- To develop strategies for retaining customers who are at high risk of churning, based on their tenure and other characteristics.
>- To determine whether there are specific payment methods that are associated with higher rates of churn and to develop recommendations for improving retention for customers who use those payment methods.
>- To investigate whether there is a relationship between the number of services a customer has and their likelihood of churning, and to develop strategies for retaining customers who are at high risk of churning based on their service usage.


## Initial hypotheses and/or questions about the data.
>- What is the distribution of customer churn in the dataset?
>- How does tenure relate to customer churn?
>- Is there a relationship between the payment method and churn?
>- Do customers with tech support tend to churn less often than those without tech support?

## THE Big Plan
> Gather ideas for the project and develop the hypothesis based on the variables. 
> Acquire "Get the data", identify were the data is coming from and set functions to bypass security constrains. 
> Prepare the data by filtering out missing values, nulls, duplicates, irrelavent dtypes. Also, renaming columns and adding new features by encode "get_dummies".
> Spliting the data into train, validate and test samples. Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset.
> Explore our takeaways and insights by learning about the context of our data. Understanding our data by using visualizations like histograms, barplots and follow with statistical test and let's not forget about our hypothesize. 
> Modele stage test your data using different models like Decission Tree, Random Forest, KNN and logistic Regression
> Conclusion your takeaway from the whole project



## Data Dictionary

| Features | Definition |
| --- | --- |
| payment_type_id | Payment type ID |
| internet_service_type_id | Internet service type ID |
| contract_type_id | Contract type ID |
| customer_id | Customer ID |
| gender | Whether the customer is a male or a female |
| senior_citizen | Whether the customer is a senior citizen or not |
| partner | Whether the customer has a partner or not |
| dependents | Whether the customer has dependents or not |
| tenure | Number of months the customer has stayed with the company |
| phone_service | Whether the customer has a phone service or not |
| multiple_lines | Whether the customer has multiple lines or not |
| online_security | Whether the customer has online security or not |
| online_backup | Whether the customer has online backup or not |
| device_protection | Whether the customer has device protection or not |
| tech_support | Whether the customer has tech support or not |
| streaming_tv | Whether the customer has streaming TV or not |
| streaming_movies | Whether the customer has streaming movies or not |
| paperless_billing | Whether the customer has paperless billing or not |
| monthly_charges | The amount charged to the customer monthly |
| total_charges | The total amount charged to the customer |
| churn | Whether the customer churned or not |
| contract_type | The contract term of the customer (Month-to-month, One year, Two year) |
| internet_service_type | Customer’s internet service provider (DSL, Fiber optic, No) |
| payment_type | The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)) |


## Instructions on how someone else can reproduce this project and findings (What would someone need to be able to recreate your project on their own?)
> 1. Clone this entire repository.
> 2. Acquire the telco_df data from MySQL or Kaggle. If data is coming form MySQL you need to have access to the data. Request user and password from Codeup instructors. 
> 3. Run project_1.ipynb to extract telco.csv file.

## Recommendations
> Build recommendations based on the findings in the data.

## Takeaways
> What was learned at the end of the project. 