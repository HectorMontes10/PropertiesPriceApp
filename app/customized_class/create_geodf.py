def construct_geodf(df, properties_types, path_shape):
    
    '''
    
    This function builds a geopandas dataframe with all the information needed for mapping. Implement missing value
    filters and cleanups for lon and lat features.
    
    Parameters:
    -----------
        df(pandas DataFrame): Contain the information of properties prices and the regions where the propertie is located
        properties_types(string): List of type of properties for which the map will be displayed.
        path_shape(String): Path to shapefile with the Department Layer, this source is public, and are avaliable in:
                            https://sites.google.com/site/seriescol/shapes
         
    Returns:
    -----------
        geod_df(geodataframe): geopandas dataframe with regions information and other variables
    
    '''
    
    import geopandas as gpd
    import warnings
    import pandas as pd
    
    #Step 1: Load an filter data:
    
    df = df[(df['missing_lon']==0) & (df['missing_lat']==0)]
    df = df[df['property_type'].isin(properties_types)]
    
    #Step 2: load the shape file using geopandas

    deptos = gpd.GeoDataFrame.from_file(path_shape)
    
    #Step 3: transform geographic coordinates in shape file to convenient system (EPSG:4326)
    #        This is important for correct visualization the maps in folium

    warnings.filterwarnings("ignore")
    deptos = deptos.to_crs("EPSG:4326")
    
    #Step 4: Compute medians for each department

    medians = df[['price','l2shp']].groupby(['l2shp']).median()
    medians.reset_index(inplace=True)
    medians = medians[['l2shp','price']]
    medians.columns = ['NOMBRE_DPT','price']

    # Step 5: Calculating data from cundinamarca to use in the Bogotá polygon
    #         (Bogotá is inconveniently separated in the shapefile)

    cund_p = medians[medians.NOMBRE_DPT=="CUNDINAMARCA"]
    cund_p = cund_p.iloc[0].price
    medians = medians.append({'NOMBRE_DPT':'SANTAFE DE BOGOTA D.C','price':cund_p}, ignore_index=True)
    
    # Step 6: Merge data of price to geodataframe:

    deptos = pd.merge(deptos, medians, on="NOMBRE_DPT")
    
    #Step 7: Retain only informative columns for this map:
    
    deptos = deptos[['NOMBRE_DPT','price','geometry']]
    
    return deptos