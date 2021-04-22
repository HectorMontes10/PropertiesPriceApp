import sys
import os
import math
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(prices_filepath, regions_filepath):
    
    '''
    
    This function load the datasets co_properties.csv that contain prices of propierties in Colombia
    
    Params:
        prices_filepath (str): String that contain the path to co_properties file (csv file with price of properties in Colombia)
        regions_filepath (str): String that contain the path to regions.csv (csv file with geographic region in Colombia)
        
    Returns:
        df_prices, regions (tupla of pandas DataFrame): 
                               df_prices: This dataframe contain the following columns:
                               columns:
                                   id-->Id for each property
                                   ad_type-->constant column
                                   start_date, end_date, created_on-->date of start, end of creation for the sale offer
                                   lat, long-->Latitude and Longitude (geographic position)
                                   l1-->Country (constant column, all properties are in Colombia)
                                   l2, l3, l4, l5, l6-->Department, City, Zone, Locality, Neighborhood where property is located 
                                   rooms, bedrooms, bathrooms, surface_total, surface_covered-->Features of property
                                   price-->Target variable for prediction model
                                   currency-->Almost all prices are in COP currency
                                   price_period-->Constant column
                                   title, description--> ad title and description 
                                   property_type-->type possibles are: Otro, Apartamento, Casa, Lote, Oficina, PH etc.
                                   operation_type-->type possibles of operations are: Venta, Arriendo, Arriendo temporal
                               regions: This dataframe contain the following columns:
                               columns:
                                   l2-->Department
                                   Region-->Region where Department is located
                                   l2shp-->Department in other format for easy integration with shape file of department
                                   
    '''
    df_prices = pd.read_csv(prices_filepath)
    regions = pd.read_csv(regions_filepath,sep=";", encoding="latin-1")
    
    return (df_prices, regions)

def clean_data(df_prices):
    
    '''
    This function clean de df_prices dataframe to be used in the model. Some operations made are:
    
    1. Remove cases with currency different to COP
    2. Remove constants columns
    3. Choose cases with operation_type = Venta and remove operation type column
    4. Assign missing value to invalid values of variables: ('surface_total', 'surface_covered', 'price') 
    5. Create dummies variables for missing values in features. 
       1--> if the feature has a missing value
       0-->if the value is valid.
    6. Remove string and date variables no-used in model or maps.
    
    '''
    
    #Step 1: Remove cases with currency different to COP
    
    df_prices = df_prices[df_prices['currency']=="COP"]
    
    #Step 2: Remove constants columns:
    
    columns_to_remove = []
    for col in df_prices.columns:
        distinct_values = df_prices[col].unique()
        if len(distinct_values)==1:
            columns_to_remove.append(col)
    df_prices = df_prices.drop(columns_to_remove, axis=1)
    
    #Step 3: Choose cases with operation type = Venta
    
    df_prices = df_prices[df_prices.operation_type=="Venta"]
    df_prices = df_prices.drop(['operation_type'], axis=1)
    
    #Step 4: Assign missing value to invalid values of variables: ('surface_total','price')
    
    surface_total_mod = list(df_prices['surface_total'].apply(lambda x: float('NaN') if x<=0 else x))
    price_mod = list(df_prices['price'].apply(lambda x: float('NaN') if x<=0 else x))
    df_prices = df_prices.drop(['surface_total','price'], axis=1)
    df_prices['surface_total'] = surface_total_mod
    df_prices['price'] = price_mod
    
    #Step 5: Create dummies variables for missing values in features
    
    columns_for_model = ['lat', 'lon', 'rooms', 'l2', 'l3', 'l4', 'l5', 'l6', 'bedrooms','rooms', 'bedrooms', 
                         'bathrooms', 'surface_total', 'surface_covered', 'price']
    
    numeric_columns = df_prices.select_dtypes(include=np.number).columns    
    names_dummies = ['missing_'+col for col in columns_for_model]
    
    for i in range(len(columns_for_model)):
        if columns_for_model[i] in numeric_columns:
            df_prices[names_dummies[i]] = df_prices[columns_for_model[i]].isna().apply(lambda x: 1 if x else 0)
        df_prices[names_dummies[i]] = df_prices[columns_for_model[i]].isna().apply(lambda x: 1 if x else 0)
    
    #Step 6: Remove string and date variables no-used in model or maps.
    
    non_used = ['id','start_date','end_date','created_on','title','description', "price_period"]
    df_prices = df_prices.drop(non_used,axis=1)
     
    print("INFO[] The cleaned table has the following fields: ")
    print("\n{}\n".format(df_prices.columns))
    
    return df_prices

def join_data(df_prices, regions):
    
    '''
    
    This function merge the table df_prices to table regions, using the key column l2. This is usefull to construct
    Choroplet map using the column l2shp and Region present in region table.
    
    Params:
        df_prices (pandas DataFrame): Contain the cleaned df_prices table 
        regions(pandas DataFrame): Contain the information in regions.csv file        
    Returns:
        df_prices_full (pandas DataFrame): Contain the cleaned df_prices table with two additional columns: l2shp and Region 
    
    '''
    
    df_prices_full = pd.merge(df_prices, regions, on="l2")
    
    return df_prices_full
    

def save_data(df_prices_full, database_filename):
    
    '''
    
    This function save the table df_prices_full in a sqlLite database. This table will be used in modelling 
    and mapping stages
    
    Params:
        df_prices_full (pandas DataFrame): Contain the cleaned df_prices table with two additional columns: l2shp and Region 
        database_filename (String): Contain the path to location where table will be stored        
    Returns:
        This function is a procedure, it return None
    
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df_prices_full.to_sql('Cleaned_prices', engine, index=False, if_exists = 'replace')
    
def main():
    
    '''
    
    This function control the ETL flow and call the other functions for load, clean, and save data
    
    '''
    
    #sys_argv = ['process_data.py', 'co_properties.csv', 'regions.csv', 'PropertiesPrices.db'] 
    if len(sys.argv) == 4:

        df_prices_filepath, regions_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    df_prices: {}\n    regions: {}'
              .format(df_prices_filepath, regions_filepath))
        df, regions = load_data(df_prices_filepath, regions_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Joining data...')
        df = join_data(df,regions)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned and joined data saved to database!')
    
    else:
        print('Please provide the filepaths of the df_prices and regions '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the joined and cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'co_properties.csv regions.csv '\
              'PropertiesPrices.db')

if __name__ == '__main__':
    main()    