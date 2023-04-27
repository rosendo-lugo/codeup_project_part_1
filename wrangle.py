# WRANGLE FILE

# My Imports
from acquire import get_telco_data
from prepare import prep_telco

import numpy as np
import pandas as pd


# ----------------------------------------------
def data_dictionary(telco_df):
    # This will show the Data Dictionary
    columns = telco_df.columns
    features = []
    definition = ['Payment type ID','Internet service type ID','Contract type ID','Customer ID','Whether the customer is a male or a female','Whether the customer is a senior citizen or not','Whether the customer has a partner or not','Whether the customer has dependents or not','Number of months the customer has stayed with the company','Whether the customer has a phone service or not','Whether the customer has multiple lines or not','Whether the customer has online security or not',
                 'Whether the customer has online backup or not','Whether the customer has device protection or not','Whether the customer has tech support or not','Whether the customer has streaming TV or not','Whether the customer has streaming movies or not','Whether the customer has paperless billing or not','The amount charged to the customer monthly','The total amount charged to the customer','Whether the customer churned or not',
                 'The contract term of the customer (Month-to-month, One year, Two year)','Customer’s internet service provider (DSL, Fiber optic, No)','The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))']
    for i in columns[0:24]:
        features.append(i)
    data_dictionary = pd.DataFrame({'features':features,
                                   'definition':definition})

    # Set the column width to unlimited
    pd.set_option('display.max_colwidth', None)

    # Left align the columns
    left_aligned_df = data_dictionary.style.set_properties(**{'text-align': 'left'}).hide_index()
    left_aligned_df = left_aligned_df.set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])

    return display(left_aligned_df)


# ----------------------------------------------
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import acquire as a
import prepare as p

def generate_predictions_df():
    # Get your telco data
    telco_df = a.get_telco_data()

    # Clean the new dataset using the new function called prep_telco
    telco_df = p.prep_telco(telco_df)

    # Split your data into train, validate and test
    train, validate, test = p.split_function(telco_df, 'churn')

    X_train = train.drop(columns=['customer_id', 'payment_type', 'churn'])
    X_validate = validate.drop(columns=['customer_id', 'payment_type', 'churn'])
    X_test = test.drop(columns=['customer_id', 'payment_type', 'churn'])

    # Set target
    target = 'churn'

    # 'y' variables are series
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    # load the trained random forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # make predictions on the test data
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    # get the customer IDs from the test data
    customer_ids = test['customer_id']

    # create a dataframe with the customer IDs, probabilities, and predictions
    results_df = pd.DataFrame({
        'customer_id': customer_ids,
        'probability_of_churn': y_proba,
        'prediction_of_churn': y_pred
    })

    # write the results to a CSV file
    results_df.to_csv('predictions.csv', index=False)

    # read the CSV file into a Pandas dataframe
    predictions_df = pd.read_csv('predictions.csv')

    return predictions_df




