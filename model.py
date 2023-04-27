import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix


# -------------Random Forest--------------------------------

def get_rf(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf

# -----------------------------------------------
def get_rf_score(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_train, y_train)
    return rf_score

# -----------------------------------------------
def get_rf_cm(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    plot = plot_confusion_matrix(rf, X_train, y_train)
    return plot

# -----------------------------------------------

def get_rf_cr(rf, X_train, y_train):
    # The y_predict for train
    y_pred_rf = rf.predict(X_train)

    #see probability of predictions 
    y_proba_rf = rf.predict_proba(X_train)

    # print the classification report
    return print(classification_report(y_train, y_pred_rf))

# ------------------------------------------------

def get_rf_accuracy(rf, X_train, y_train):
    # The y_predict for train
    y_pred_rf = rf.predict(X_train)

    #see probability of predictions 
    y_proba_rf = rf.predict_proba(X_train)
    
    # CONFUSION MATRIX
    rf_cm = confusion_matrix(y_train, y_pred_rf)

    TN, FP, FN, TP = rf_cm.ravel()
    TN, FP, FN, TP

    rf_all_ = (TP + TN + FP + FN)

    rf_accuracy = (TP + TN) / rf_all_
    print(f"Accuracy: {rf_accuracy}\n")

    rf_TPR = rf_recall = TP / (TP + FN)
    rf_FPR = FP / (FP + TN)
    print(f"True Positive Rate/Sensitivity/Recall/Power: {rf_TPR}")
    print(f"False Positive Rate/False Alarm Ratio/Fall-out: {rf_FPR}")

    rf_TNR = TN / (FP + TN)
    rf_FNR = FN / (FN + TP)
    print(f"True Negative Rate/Specificity/Selectivity: {rf_TNR}")
    print(f"False Negative Rate/Miss Rate: {rf_FNR}\n")

    rf_precision =  TP / (TP + FP)
    rf_f1 =  2 * ((rf_precision * rf_recall) / ( rf_precision + rf_recall))
    print(f"Precision/PPV: {rf_precision}")
    print(f"F1 Score: {rf_f1}\n")

    rf_support_pos = TP + FN
    rf_support_neg = FP + TN
    print(f"Support (0): {rf_support_pos}")
    print(f"Support (1): {rf_support_neg}")

# -------------------------------------------------------------------------

def get_rf_train_val_acc(rf, X_train, y_train, X_validate, y_validate, scores_df):
    # Run everything in one simple code. 
    scores_all = []

    for x in range(1,11):
        # make the object
        rf = RandomForestClassifier(random_state=123, min_samples_leaf=x, max_depth=11-x)

        # fit the object
        rf.fit(X_train, y_train)

        # transform the object
        train_acc = rf.score(X_train, y_train)

        # evaluate on my validate data
        val_acc = rf.score(X_validate, y_validate)

        # store results in a DataFrame
        result = pd.DataFrame({'min_samples_leaf': [x], 'max_depth': [11-x], 'train_acc': [train_acc], 'val_acc': [val_acc]})

        # append to scores_all list
        scores_all.append(result)

    # combine all results into a single DataFrame
    scores_df = pd.concat(scores_all, ignore_index=True)

    scores_df['difference'] = scores_df.train_acc - scores_df.val_acc

    scores_df = scores_df.style.hide_index()
    
    return scores_df


# -------------------------------------------------------
def get_rf_plot(scores_df):
    plt.figure(figsize=(12,6))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.val_acc, label='unseen val', marker='o')
    plt.xlabel('max depth and min leaf sample')
    plt.ylabel('accuracy')
    plt.title('how does the accuracy change with max depth on train and validate?')
    plt.legend()
    plt.xticks(np.arange(0,10,1))
    plt.grid()
    return plt.show()