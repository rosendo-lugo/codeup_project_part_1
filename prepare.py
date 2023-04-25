# Standard ds libraries
import pandas as pd

# Import splitting function
from sklearn.model_selection import train_test_split


#-------------------------
# This function splits the data in an 80/20%
def split_function(df, target_varible, seed=123):
    train_validate, test = train_test_split(df,
                                   random_state=seed,
                                   test_size=0.2,
                                   stratify= df[target_varible])
    
    train, validate = train_test_split(train_validate,
                                   random_state=seed,
                                   test_size=0.25, 
                                   stratify= train_validate[target_varible])
    return train, validate, test



#-------------------------
# This will clean the data. 
def prep_telco(telco_df):
    '''
    This function will clean the the telco dataset
    '''
    telco_df = telco_df.drop(columns =['contract_type_id', 'internet_service_type_id', 'payment_type_id'])
    
    dummy_telco = pd.get_dummies(telco_df[['gender',
                                             'partner',
                                             'dependents',
                                             'phone_service',
                                             'multiple_lines',
                                             'online_security',
                                             'online_backup',
                                             'device_protection',
                                             'tech_support',
                                             'streaming_tv',
                                             'streaming_movies',
                                             'paperless_billing',
                                             'churn',
                                             'contract_type',
                                             'internet_service_type',
                                             'payment_type']], dummy_na=False, drop_first=[True, True])
    telco_df = pd.concat([telco_df, dummy_telco], axis=1)
    telco_df.total_charges = telco_df.total_charges.str.replace(' ', '0').astype(float)
    return telco_df









