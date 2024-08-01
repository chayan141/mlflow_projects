import mlflow
import pandas as pd
import os
import numpy as np
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,accuracy_score,precision_score,recall_score,confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import argparse


#read_data as dataframe

def get_data():
    url = 'https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv'
    #read_data as dataframe
    try:
        df = pd.read_csv(url,sep=';')
        return df
    except Exception as e:
        raise e
    
def evaluate(y_true,y_pred,pred_prob):
    # mae = mean_absolute_error(y_true,y_pred)
    # mse = mean_squared_error(y_true,y_pred)
    # rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    # r2 = r2_score(y_true,y_pred)
    # return mae,mse,rmse,r2 
    #roc : receiver operating characteristics curve, plots against true poritive rates (TPR) against the False Positive Rates (FPR)
    #AUC = 1: Perfect Classification, 0.5<AUC<1 : Good Classification, AUC = 0.5 : Moderate Classification, AUC<0.5 : Worst Classification
    
    accuracy = accuracy_score(y_true,y_pred)
    roc = roc_auc_score(y_true,pred_prob,multi_class='ovr')
    return accuracy,roc


def main(n_estimators,max_depth):
    df = get_data()
    
    #Train Test Split
    train,test= train_test_split(df)
    X_train = train.drop(['quality'],axis=1)
    X_test = test.drop(['quality'],axis=1)
    y_train = train[['quality']]
    y_test = test[['quality']]

    #model Training (using elastic net)
    # lr = ElasticNet()
    # lr.fit(X_train,y_train)
    # pred = lr.predict(X_test)


    #Tracking the Experiments in we will use mlflow

    #MLflow Tracking
    with mlflow.start_run():

        #model Training using Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        rf.fit(X_train,y_train)
        pred = rf.predict(X_test)
        pred_prob = rf.predict_proba(X_test)


        #evaluate model (for elastic net regression)
        # mae,mse,rmse,r2 = evaluate(y_test,pred)
        # print(f'MSE : {np.round(mse,2)}, mae = {np.round(mae,2)}, rmse = {np.round(rmse,2)}, r2_score = {np.round(r2,2)}')
        #value closer to 1 for r2 score : 
        # a large proportion of the variance in the dependent variable is predictable from the independent variables.
        #A higher R² indicates a better fit of the model to the data, but it does not indicate whether the predictions are accurate.

        #evaluate model (for random forest classifier)
        accuracy,roc = evaluate(y_test,pred,pred_prob)

        #MLFlow Track
        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_metric('accuracy',accuracy)
        mlflow.log_metric('roc_auc_score',roc)

        #mlflow model logging
        mlflow.sklearn.log_model(rf, 'wine_classifiation_rf')


        print(f"Accuracy: {accuracy}, roc_auc_score: {roc}")

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n", default=50, type = int)
    args.add_argument("--max_depth","-m", default=5, type = int)
    parse_args = args.parse_args()


    #argument parser : any input passing through the terminal
    try:
        main(n_estimators = parse_args.n_estimators,max_depth = parse_args.max_depth)
    except Exception as e:
        raise e

