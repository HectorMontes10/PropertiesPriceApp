from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class InputData(BaseEstimator, TransformerMixin):
    
    '''
    This transformer do the following operations:
    
    1. Compute medians for each pair department-feature, then assign this value to missing values of features
       present in the include list.
    2. Replace extreme values for 0.98 quantile for a selected list of features in the include attribute.
    3. Compute log10 for a selected list od feature in the include_log attribute, and replace original variables.
    
    '''
    def __init__(self, include = None, include_log = None,  segmentation_col="l2shp"):
        
        self.medians_by_department =  None
        self.quantile98_by_feature = {}
        if include ==None:
            self.include = ['rooms', 'surface_total', 'surface_covered','bathrooms']
        else:
            self.include = include
        if include_log ==None:
            self.include_log = ['surface_total', 'surface_covered']
        else:
            self.include_log = include_log
        self.segmentation_col = segmentation_col  
        self.selected = []
        
    def __construct_medians(self, df):
        
        median = lambda col: col.median()
        categories = df[self.segmentation_col].unique()    
        selected = [x for x in df.columns if self.__check_name(x)]
        self.selected = selected
        df_ = df[selected]
        medians_by_department = pd.DataFrame(columns=selected,index=categories)
        for cat in categories:
            row_medians = df_[df['l2shp']==cat].apply(median)
            medians_by_department.loc[cat,:] = row_medians
        self.medians_by_department = medians_by_department
    
    def __check_name(self,name):
            
            to_remove = ['l2shp','missing','property_type']
            i = 0
            while (i<len(to_remove)):
                flat = True
                if to_remove[i] in name:
                    flat = False
                    break
                i += 1
            return flat

    def __construct_quantile(self, df):
                
        for col_name in self.include:
            col = df[~df[col_name].isna()][col_name]
            self.quantile98_by_feature[col_name] = np.quantile(col,0.98)
        print(self.quantile98_by_feature)
    
    def __create_col_without_extremes(self,col,col_name):
        quantile = self.quantile98_by_feature[col_name]
        col_mod = [x if x<=quantile else quantile for x in col]
        return col_mod
    
    def __compute_log10(self,df):
        
        log10_mod = lambda x: np.log10(x) if x>0 else x
        
        for col_name in self.include_log:
            df[col_name]= df[col_name].apply(log10_mod)
        return df 
    
    def fit(self, X, y = None):
        
        self.__construct_medians(X)
        self.__construct_quantile(X)
        
        return self

    def transform(self, X):
        
        # apply medians over missing data
        
        X_mod = pd.DataFrame(columns=X.columns)
        categories = X[self.segmentation_col].unique()
        for row_name in categories:
            subset = X[X[self.segmentation_col]==row_name]
            new_subset = pd.DataFrame(columns=X.columns)
            for col_name in X.columns:
                if col_name in self.selected:    
                    median = self.medians_by_department.loc[row_name,col_name]
                    #print('row_name={}, col_name={}, median={}'.format(row_name, col_name,median))
                    new_subset[col_name] = subset[col_name].fillna(median)
                else:
                    new_subset[col_name] = subset[col_name]
            X_mod = pd.concat([X_mod,new_subset],axis=0)
        
        for col_name in self.include:
            X_mod[col_name] = self.__create_col_without_extremes(list(X_mod[col_name]),col_name)

        X_mod = self.__compute_log10(X_mod)
        X_mod = X_mod.drop(['l2shp'],axis=1)
        return X_mod