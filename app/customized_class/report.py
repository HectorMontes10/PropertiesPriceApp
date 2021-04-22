def create_missing_report(df, list_features, segment_by):
    
    '''
    
    This function create a report with information about missing values in a list of features. 
    Calculations are made on each category of the variable passed in "segment_by" parameter
    
    Params:
        df (pandas DataFrame): Contain the dataframe on which the calculations will be made 
        list_features (list of string): Names of features in the dataframe on which missing data is calculated
        segment_by (string): Name for variable used for segmentation.
    Returns:
        df_report (pandas DataFrame): Contain the report of missing values in features present in the list by each category
                                      in segment_by variable  
    
    '''
    import pandas as pd

    # Count of cases for eacg categories in segment_by column:
    
    counts = df[segment_by].value_counts()
    categories = list(counts.index)
    
    # Create a empty dataframe where missing calculations well be stored.
    
    count_name = ['count']
    perc_names = ['perc_'+x for x in list_features]
    missing_names = ['missing_'+x for x in list_features]
    columns = count_name + missing_names + perc_names
    df_report = pd.DataFrame(index=categories, columns=columns)
    
    # Fill the count column:
    
    df_report['count'] = counts
    
    # For each feature in list_features and categorie in segment_by column, count the missing values:

    for cat in categories:
        df_ = df[df[segment_by]==cat]
        missing_values = df_[missing_names].apply(sum, axis=0)
        perc_values = [x/df_report.loc[cat,'count'] for x in missing_values]
        df_report.loc[cat,missing_names] = missing_values
        df_report.loc[cat,perc_names] = perc_values
    return(df_report)