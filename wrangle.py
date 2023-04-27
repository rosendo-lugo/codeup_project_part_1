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
def generate_predictions(model, X_test, output_path):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Create a DataFrame with the customer IDs, predicted probabilities, and predicted classes
    customer_id = X_test["customer_id"]    
    predictions_df = pd.DataFrame({'customer_id': customer_id, 
                                'probability_of_churn': y_proba, 
                                'prediction_of_churn': y_pred})

    # Convert the predicted classes to binary values (1 = churn, 0 = not churn)
    predictions_df['prediction_of_churn'] = predictions_df['prediction_of_churn'].map({'Yes': 1, 'No': 0})

    # Write the predictions dataframe to a CSV file
    predictions_df.to_csv(output_path, index=False)
    
    return predictions_df
