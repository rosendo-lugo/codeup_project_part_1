import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Graph imports
import matplotlib.pyplot as plt
import seaborn as sns

# SKLearn imports
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

#import this for the decision tree!
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.tree import plot_tree



# ----------------BASELINE ACCURACY-------------------------------
def get_baseline_acc(train):
    baseline_accuracy = (train.churn == 0).mean()
    return baseline_accuracy

# ------------------------------------------------------------------
def get_Xs_ys(train, validate, test):
    # Lets drop columns that are objects and don't add any value to the data. Also, we need to remove the 'churn_Yes' column because is our TARGET.
    # Also, lets convert train to X_train.
    X_train = train.drop(columns = ['customer_id','payment_type','churn'])
    X_validate = validate.drop(columns = ['customer_id','payment_type','churn'])
    X_test = test.drop(columns = ['customer_id','payment_type','churn'])

    # Set target
    target = 'churn'

    # 'y' variables are series
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]
    return X_train, X_validate, X_test, y_train, y_validate, y_test

# -------------Decision Treee-------------------------------------
# Setting the decision tree classifier
def get_dt(X_train, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    return dt

# -----------------------------------------------
# Getting the score
def get_dt_score2(X_train, y_train):
    dt_score = DecisionTreeClassifier()
    dt_score.fit(X_train, y_train)
    dt_score = dt_score.score(X_train, y_train)
    return dt_score

# -----------------------------------------------
# Getting the plot confusion matrix
def get_dt_cm(X_train, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    plot = plot_confusion_matrix(dt, X_train, y_train)
    return plot

# -----------------------------------------------
# Getting the classification report
def get_dt_cr(dt, X_train, y_train):
    # The y_predict for train
    y_pred = dt.predict(X_train)

    #see probability of predictions 
    y_proba = dt.predict_proba(X_train)

    # print the classification report
    return print(classification_report(y_train, y_pred))

# -----------------------------------------------
def get_decision_tree(dt, X_train):
    #see the tree that was built
    #plot_tree
    plt.figure(figsize=(6,6))
    plot_tree(dt, #our fitted object
             feature_names=X_train.columns, #puts in our features for the questions
              class_names=dt.classes_.astype(str), #enters the selected class
              filled=True #colors our leaves and branches based on the class
             )
    return plt.show()

# -----------------------------------------------
def get_dt_score(train, X_train, y_train, X_validate, y_validate, X_test, y_test):
    scores_all = []

    for x in range(1,10):

        dt = DecisionTreeClassifier(max_depth=x)
        dt.fit(X_train, y_train)
        # evaluate on train
        train_acc = dt.score(X_train, y_train)

        #evaluate on validate
        val_acc = dt.score(X_validate, y_validate)

        # view the difference between train_acc and val_acc
        diff = train_acc - val_acc

        scores_all.append([x, train_acc, val_acc, diff])

    scores_df = pd.DataFrame(scores_all, columns=['max_depth','train_acc','val_acc','diff'])
    test_acc = dt.score(X_test, y_test)
    dt_scores_df = scores_df.sort_values('diff')
    baseline = (train.churn == 0).mean()
    print(f'Baseline: {baseline}')
    print(f'train data prediction: {dt_scores_df.iloc[6].train_acc}')
    print(f'validation data prediction: {dt_scores_df.iloc[6].val_acc}')
    print(f'unseen data prediction: {test_acc}')



# -----------------------------------------------
def get_df_plot(dt_scores_df, max_depth):
    train_acc = dt_scores_df.loc[dt_scores_df.max_depth == max_depth, 'train_acc']
    val_acc = dt_scores_df.loc[dt_scores_df.max_depth == max_depth, 'val_acc']
    max_val_acc = dt_scores_df.val_acc.max()
    max_val_depth = dt_scores_df.loc[dt_scores_df.val_acc.idxmax(), 'max_depth']
    
    plt.figure(figsize=(12,6))
    plt.plot(dt_scores_df.max_depth, dt_scores_df.train_acc, label='train', marker='o')
    plt.plot(dt_scores_df.max_depth, dt_scores_df.val_acc, label='unseen validate', marker='o')
    plt.plot(max_depth, train_acc, label=f'train (max depth={max_depth})', marker='o', markersize=10, color='green')
    plt.plot(max_depth, val_acc, label=f'unseen validate (max depth={max_depth})', marker='o', markersize=10, color='red')
    plt.axvline(max_val_depth, color='gray', linestyle='--', label=f'max val acc ({max_val_acc:.3f}) at max depth={max_val_depth}')
    plt.xlabel('max depth')
    plt.ylabel('accuracy')
    plt.title('how does the accuracy change with max depth on train and validate?')
    plt.legend()
    plt.xticks(np.arange(0,10,1))
    plt.grid()
    return plt.show()

# -------------Random Forest--------------------------------
# Setting the random forest classifier
def get_rf(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf

# -----------------------------------------------
# Getting the score
def get_rf_score(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    scores_rf = rf.score(X_train, y_train)
    return scores_rf

# -----------------------------------------------
# Getting the plot confusion matrix
def get_rf_cm(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    plot = plot_confusion_matrix(rf, X_train, y_train)
    return plot

# -----------------------------------------------
# Getting the classification report
def get_rf_cr(rf, X_train, y_train):
    # The y_predict for train
    y_pred_rf = rf.predict(X_train)

    #see probability of predictions 
    y_proba_rf = rf.predict_proba(X_train)

    # print the classification report
    return print(classification_report(y_train, y_pred_rf))

# ------------------------------------------------
# Getting the accuracy 
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
# Comparing the random forest train and validation accuracy
def get_rf_train_val_acc(train, X_train, y_train, X_validate, y_validate):
    
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
    scores_rf = pd.concat(scores_all, ignore_index=True)
    scores_rf['difference'] = scores_rf.train_acc - scores_rf.val_acc
    baseline = (train.churn == 0).mean()
    print(f'Baseline: {baseline}')
    print(f'train data prediction: {scores_rf.iloc[8].train_acc}')
    print(f'validation data prediction: {scores_rf.iloc[8].val_acc}')

# -------------------------------------------------------
# Plotting a graph
def get_rf_plot(rf_scores_df): 
    plt.figure(figsize=(12,6))
    plt.plot(rf_scores_df.max_depth, rf_scores_df.train_acc, label='Train accuracy', marker='o')
    plt.plot(rf_scores_df.max_depth, rf_scores_df.val_acc, label='Validation accuracy', marker='o')
    plt.xlabel('Max depth and min samples leaf')
    plt.ylabel('Accuracy')

    plt.xticks([2,4,6,8,10],
              [('2, 2'),('4, 4'),('6, 3'),('8, 5'),('10, 7')]
              )

    plt.title('How does the accuracy change with max depth and min samples leaf on train and validate?')

    # find the row with the best validation accuracy
    best_row = rf_scores_df.loc[rf_scores_df.difference.abs().idxmin()]

    # add a red circle around the point for the best accuracy
    plt.plot(best_row.max_depth, best_row.val_acc, marker='o', mec='red', mew=2, ms=10, label=f'Best validation accuracy: {best_row.val_acc:.2f} (Max depth = {best_row.max_depth}, Min samples leaf = {best_row.min_samples_leaf})')

    plt.legend()
    return plt.show()



def get_rf_plot2(rf_scores_df, X_train, y_train, X_test, y_test): 
    # Train a random forest model for different combinations of max depth and min samples leaf
    max_depths = range(2, 11)
    min_samples_leaves = range(1, 6)
    test_accs = []
    for max_depth in max_depths:
        row = []
        for min_samples_leaf in min_samples_leaves:
            rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
            rf.fit(X_train, y_train)
            scores_rf = rf.score(X_train, y_train)
            test_acc = rf.score(X_test, y_test)
            row.append(test_acc)
        test_accs.append(row)

    # Create a heatmap
    sns.set(font_scale=1.2)
    ax = sns.heatmap(test_accs, annot=True, cmap='Blues', xticklabels=min_samples_leaves, yticklabels=max_depths, fmt='.3f')
    plt.xlabel('Min samples leaf')
    plt.ylabel('Max depth')
    plt.title('Test accuracy of random forest for different combinations of max depth and min samples leaf')

    # find the row with the best test accuracy
    best_row = np.argmax(np.max(test_accs, axis=1))
    best_col = np.argmax(np.max(test_accs, axis=0))

    # add a red square around the point for the best accuracy
    rect = plt.Rectangle((best_col, best_row), 1, 1, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    plt.legend(['Best 45 combinations of hyperparameters'], bbox_to_anchor=(1.3, 0.5), loc='upper left')
    return plt.show()

def get_rf_plot3(rf_scores_df, X_train, y_train, X_test, y_test): 
    # Define the hyperparameters to search
    max_depths = [2, 4, 6, 8, 10]
    min_samples_leaves = [2, 4, 5, 7, 9]

    # Train a random forest model for each combination of hyperparameters and record the training and testing accuracies
    results = []
    for max_depth in max_depths:
        for min_samples_leaf in min_samples_leaves:
            rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
            rf.fit(X_train, y_train)
            train_acc = rf.score(X_train, y_train)
            val_acc = rf.score(X_test, y_test)
            results.append({'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'train_acc': train_acc, 'val_acc': val_acc})

    # Convert the results to a pandas DataFrame
    results_df = pd.DataFrame(results)

    # Compute the differences between validation and test accuracy for each combination of hyperparameters
    results_df['val_test_diff'] = results_df['val_acc'] - results_df['train_acc']

    # Pivot the data to create a heatmap
    heatmap_data = pd.pivot_table(results_df, values='val_test_diff', index='max_depth', columns='min_samples_leaf')

    # Create a heatmap of the differences between validation and test accuracy
    sns.set(font_scale=1.2)
    ax = sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', xticklabels=min_samples_leaves, yticklabels=max_depths, fmt='.3f')
    plt.xlabel('Min samples leaf')
    plt.ylabel('Max depth')
    plt.title('Differences between validation and test accuracy for random forest')

    # find the cell with the smallest difference between validation and test accuracy
    best_row, best_col = np.unravel_index(np.argmin(heatmap_data.values), heatmap_data.shape)

    # add a red square around the cell with the smallest difference
    rect = plt.Rectangle((best_col, best_row), 1, 1, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    # add a legend for the red square
    plt.legend(['Best combination'], bbox_to_anchor=(1.3, 0.5), loc='center left')
    return plt.show()


# -------------Logistic Regression--------------------------------

def get_logit(X_train, y_train):
    logit = LogisticRegression(class_weight='balanced')
    logit.fit(X_train, y_train)
    return logit

# -------------------------------------------------------    
def get_logit_score(train, X_train, y_train, X_validate, y_validate):
    logit = LogisticRegression(class_weight='balanced')
    logit.fit(X_train, y_train)
    train_logit_score = logit.score(X_train, y_train)
    val_logit_score = logit.score(X_validate, y_validate)
    baseline = (train.churn == 0).mean()
    print(f'Baseline: {baseline}')
    print(f'train data prediction: {train_logit_score}')
    print(f'validation data prediction: {val_logit_score}')

# -----------------------------------------------
def get_logit_cm(logit,X_train, y_train):
    plot = plot_confusion_matrix(logit, X_train, y_train)
    return plot

# -----------------------------------------------

def get_logit_cr(logit, X_train, y_train):
    # The y_predict for train
    y_pred = logit.predict(X_train)

    #see probability of predictions 
    y_proba = logit.predict_proba(X_train)

    # print the classification report
    return print(classification_report(y_train, y_pred))

# -------------------------------------------------------------------------
def get_logit_train_val_acc(train, X_train, y_train, X_validate, y_validate):
    scores_all = []
    logit_class_weight = 'balanced'
    
    for C_value in [0.01, 0.1, 1, 10]:
        for penalty_value in ['l1', 'l2']:
            logit = LogisticRegression(random_state=123, C=C_value, penalty=penalty_value, solver='saga', max_iter=1000, class_weight=logit_class_weight)
            logit.fit(X_train, y_train)
            train_acc = logit.score(X_train, y_train)
            val_acc = logit.score(X_validate, y_validate)
            result = pd.DataFrame({'C': [C_value], 'penalty': [penalty_value], 'train_acc': [train_acc], 'val_acc': [val_acc]})
            scores_all.append(result)

    logit_scores_df = pd.concat(scores_all, ignore_index=True)
    logit_scores_df['difference'] = logit_scores_df.train_acc - logit_scores_df.val_acc


