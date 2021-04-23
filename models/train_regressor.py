#Import packages

from time import time
import re
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score

#Import custom class and functions

import sys
sys.path.append("./app/customized_class")
from input_data import InputData
from dummy_estimator import DummyEstimator

def load_data(database_filepath):
    
    '''
    
    This function load a database of cleaned properties and remove non informative variables like this:
    
    -'l2', 'l3', 'l4', 'l5', 'l6', 'Region'
    -'missing_l2', missing_l3', 'missing_l4', 'missing_l5', 'missing_l6', 'missing_lat', 'missing_lon'
    
    - 'l2' is removed because is redundant with 'l2shp'
    - 'Region' is removed because a department(l2shp) belongs to a single region, therefore the department defines the region,
       and this can lead to collinearity problems.
    - 'missing_l2' and 'missing_price' are removed because are constant.(no missing values in this columns)
    - lat and lon are in this dataframe because are used for visualizations.
    
    In addition to this we also remove values for properties other than houses or apartments, because the model
    only include this categories.
    
    Params:
        database_filepath (string): Path to sqlLite database
    Returns:
        df(pandas DataFrame): Matrix with features for train model and visualizations (lat and lon columns) and
                              target column ('price')
        
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("Cleaned_prices",con=engine)
    
    columns_to_drop = ['l2', 'l3', 'l4', 'l5', 'l6','Region','missing_l2','missing_l3', 'missing_l4',
                       'missing_l5', 'missing_l6', 'missing_price']
    df = df.drop(columns_to_drop, axis=1)
    df = df[df['property_type'].isin(['Casa','Apartamento'])]
    
    return df

def adjust_data_for_model(df):
    
    '''
    This function the data in convenient format for the stage modelling. Some operations made are:
    
        1. Remove incomplete rows, that is, rows which have more than 2 missing fields in this list: 
           [rooms, log_surface_total, log_surface_covered, bathrooms]
        2. Exclude departments with less of 100 rows in the dataframe
        3. Include dummy variables for categorical variables: property_type, and l2shp (Department)
           using One-Hot Encoding because they are nominal variables. Here the original categorical variables
           are droped, except for l2shp because is ussefull for input median in missing values in a posterior step.
        4. Replace price for log10(price).
        5. Split the dataframe en covariates and target variable (X,y)
               
    Parameters:
    -----------
        df(pandas DataFrame): DataFrame with relevant columns and rows for modelling stage
    
    Returns:
    -----------
        
        df(pandas DataFrame): DataFrame with features adjusted for modelling stage
        
    '''
    
    # Step 1: Remove incomplete rows:
    
    columns = ['missing_rooms', 'missing_surface_total', 'missing_surface_covered','missing_bathrooms','missing_lat','missing_lon']
    counts = df[columns].apply(sum,axis=1)
    df = df[counts<=2]

    # Step 2: Exclude departments with less of 100 points.
    
    rows_by_departments = df['l2shp'].value_counts()
    departments_to_exclude = list(rows_by_departments[rows_by_departments<100].index)
    df = df[~df['l2shp'].isin(departments_to_exclude)]
    
    # Step 3: Include dummy variables:
    
    var_cat = df.select_dtypes(include=['object']).copy().columns
    for col in var_cat:
        try:
            
            if ((col!='l2shp') & (col!='property_type')):
                df = pd.concat([df.drop(col,axis=1),pd.get_dummies(df[col], prefix = col, prefix_sep = "_", drop_first = True, 
                                                                   dummy_na = False, dtype=int)],axis=1)
            else:
                df = pd.concat([df,pd.get_dummies(df[col], prefix = col, prefix_sep = "_", drop_first = True, 
                                                                   dummy_na = False,dtype=int)],axis=1)
                
        except Exception as e:
            print(col, "processing error")
            print(e)
        
    # Step 4. Replace price for log10(price):
    
    df['price'] = np.log10(df['price'])
    
    # Step 5. Split the dataframe en covariates and target variable (X,y)
    
    X = df.loc[:,df.columns!="price"]
    y = df['price']
    
    return X,y

def build_model():
    
    '''
    This function construct a pipeline with custom transformer and estimators. The pipeline is passed to grid search function
    for tuning parameter for estimators. The pipeline include FeatureUnion based in custom transformer.
    
    Params:
        None
    Returns:
        cv(GridSearch object): An object of class GridSearch fitting over train data. The object have an attribute "best_estimator_"
                               that contain the best model finded.
    
    '''
    
    pipeline = Pipeline([
             ('input', InputData()),
             ('scaler', StandardScaler()),
             ('clf', DummyEstimator())])
    
    # Estimator 1: LinearRegression (clasic model):
    
    fit_intercept = [False, True] 
    
    # Estimator 2: Stochastic Gradient Descent:

    # The gradient of the loss is estimated each sample at a time and the model is updated along the way with
    # a decreasing strength schedule (aka learning rate). 
    
    # Choosen loss functions for SGD
    
    loss_function_SGD =["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]
    
    # Epsilon parameter according loss function selected:
    
    epsilon_huber = [0.4,0.7,1]
    epsilon_epsilon_insensitive = [0.01,0.1,0.2]
    epsilon_squared_epsilon_insensitive = [0.01,0.1,0.2]
    learning_rate = ["invscaling", "adaptive"]
    
    # Estimator 3: Support Vector Regression with Linear Kernel
    
    # Analogously to SVM for classification problem, the model produced by Support Vector Regression depends only
    # on a subset of the training data, because the cost function ignores samples whose prediction is close to their target.
        
    loss_functions_SVR = ["epsilon_insensitive", "squared_epsilon_insensitive"]
    
    # Candidate learning algorithms and their hyperparameters
    
    # Note that the SGDRegressor is splitted in several versions because loss functions is related to specific epsilon
    # values

    search_space = [{'clf': [LinearRegression()],
                    'clf__fit_intercept': fit_intercept},
                    {'clf': [SGDRegressor()],
                     'clf__loss': ['squared_loss']},
                    {'clf': [SGDRegressor()],
                     'clf__loss': ['huber'],
                     'clf__epsilon': epsilon_huber,
                     'clf__learning_rate': learning_rate},
                    {'clf': [SGDRegressor()],
                     'clf__loss': ['epsilon_insensitive'],
                     'clf__epsilon': epsilon_epsilon_insensitive,
                     'clf__learning_rate': learning_rate},
                    {'clf': [SGDRegressor()],
                     'clf__loss': ['squared_epsilon_insensitive'],
                     'clf__epsilon': epsilon_squared_epsilon_insensitive,
                     'clf__learning_rate': learning_rate},
                    {'clf': [LinearSVR()],
                     'clf__loss': loss_functions_SVR}
                   ]

    #Create grid search

    cv = GridSearchCV(pipeline, search_space, n_jobs=-1)
    
    return cv

def save_data_to_evaluate_model(df,X_test,y_test,model):
    
    '''
    This function save data for evaluate the best model in gridsearch object fitted over train data using. The data saved is the
    X_test dataset with latitud, longitud, error, and squared_error for each pont in the sample.
    
    Params:
        
        df(pandas DataFrame): DataFrame with data prepared for modelling
        model(gridSearch object): gridSearch object fitted over train data.
        X_test(numpy array): array of string to be used for test model
        y_test(numpy array): Matrix with test dataset to evaluate model
        
    Return:
        result (string): Printed string with metric r_2 for model evaluated and save the residuals for visualization in map
    
    '''
    
    best_model = model.best_estimator_
    y_pred = best_model.predict(X_test)
    r2_model = r2_score(y_test.to_numpy(), y_pred)
    df_pred = pd.DataFrame({'y_pred':list(y_pred)}, index=X_test.index)
    df_results = pd.concat([X_test,df_pred,y_test],axis=1)
    df_results['errors'] = df_results['y_pred']-df_results['price']
    df_results['l2shp'] = df['l2shp']
    df_results['property_type'] = df['property_type']
    df_results['squared_errors'] = df_results['errors']**2
    df_to_save = df_results[['lat','lon','l2shp','errors','squared_errors','missing_lon','missing_lat']]
    df_to_save.to_csv("./app/source/test_errors.csv")
    print("   The R2 for the best model was {}".format(r2_model))

def save_model(model, model_filepath):
    
    '''
    
    This function save the gridSearch object. The object will be used to predict new prices of real state real estate properties in
    Colombia. We can make predictions over the original dataset and graph a Choroplet map to inspect the performance of trained model
    in each department based on errors of predictions. 
    
    Params:
        model (gridSearch object): Contain the best fitted gridSearch object over train data
        model_filepath: Path where the object will be stored.
    Return:
        None: This is a procedure. it store a pickle object that contain the model in the model_filepath location.
    
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    
    '''
    
    This function control the training flow and call the other functions for load, train, and save model
    
    '''
    
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df = load_data(database_filepath)
        
        print('Adjusting data for modeling purposes ...'.format(database_filepath))
        
        df, y = adjust_data_for_model(df)
        
        X, y = df, y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 100)

        print('Building model...')
        model = build_model()

        print('Training model...')
        start_time = time()
        model.fit(X_train, y_train)
        end_time = time()
        print("The time spent training the model was: {}".format(end_time-start_time))

        print('Saving Predictions to evaluate model...')
        save_data_to_evaluate_model(df,X_test,y_test,model)
        print('Saving Predictions to evaluate model (see the app for visualization in map)')

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the Properties Prices database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_regressor.py ../data/PropertiesPrices.db model.pkl')

if __name__ == '__main__':
    main()