import pandas as pd
import numpy as np 
import os 

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet 

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split

import argparse

def get_data():
    try:
        URL="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        #read data as csv file 
        df=pd.read_csv(URL,sep=';')
        return df
    except Exception as e:
        raise e 
    
def evaluate(y_pred,y_test):
    '''a=mean_absolute_error(y_pred,y_test)
    b=mean_squared_error(y_pred,y_test)
    c=np.sqrt(b)
    d=r2_score(y_pred,y_test)
    return a,b,c,d '''

    acc=accuracy_score(y_pred,y_test)
    return acc

    

def main(n_estimator,max_depth):
    df=get_data()
    train,test=train_test_split(df)
    X_train=train.drop(['quality'],axis=1)
    X_test=test.drop(['quality'],axis=1)
    y_train=train[['quality']]
    y_test=test[['quality']]
    '''
    lr=ElasticNet()
    lr.fit(X_train,y_train)
    pred=lr.predict(X_test)
    '''
    rf=RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth)
    rf.fit(X_train,y_train)
    pred=rf.predict(X_test)


    #evaluate the model 
    '''mae,mse,rmse,r2=evaluate(pred,y_test)
    print(f"mae {mae}")
    print(f"mse {mse}")
    print(f"rmse {rmse}")
    print(f"r2 score {r2}")'''

    #evaluate classification model 
    accuracy=evaluate(pred,y_test)
    print(f"the accuracy of random forest is {accuracy}")






if __name__=='__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument("--n_estimator","-n",default=50,type=int)
    arg.add_argument("--max_depth","-m",default=5,type=int)

    parse_args=arg.parse_args()

    try:
        main(n_estimator=parse_args.n_estimator,max_depth=parse_args.max_depth)
    except Exception as e :
        raise e 
