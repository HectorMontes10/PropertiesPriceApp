def create_choropleth(geodf, col_to_plot, col_id, factor_scale):
    
    '''
    
    Construct a Choropleth folium map based on geodataframe and an column to plot.
    More information can be associated with the map for include more columns in the input geopandas.
    This is importante for other uses, such as inserting marks or enabling filters. 
    
    Adjust factor_scale for pretty visualization to upper bar.
    
    Parameters:
    -----------
        geodf (geopandas DataFrame): Geodataframe with the information that we want associate to map.
        col_to_plot (str): Column that we will be plot
        col_id(str): Column for to be used like id of cases.
        factor_scale (float): re-scaler value for transform values in column to plotting.
    Returns:
    -----------
        map_ (folium map object): object folium with Choropleth.
        geodf_geo_json: Object json with information linked to map
    
    '''
    
    import folium
    
    #Create a geodict. This will be the data attribute for the choropleth
    
    geodf_dict = geodf[[col_id,col_to_plot]]
    geodf_dict[col_to_plot] = [x/factor_scale for x in list(geodf_dict[col_to_plot])]
        
    #Create a geo_json. This contain the geographic coordinates for each point in the map
    
    #It is very important to fit an index to the geodataframe before exporting it to geo_json, because folium uses
    #it for indexing
    
    geodf = geodf.set_index(col_id)
    geodf_geo_json = geodf.to_json()

    bins = list(round(geodf_dict[col_to_plot].quantile([0, 0.15, 0.45, 0.60, 0.80, 0.90, 1]),1))
    map_ = folium.Map(location=[5.170035, -74.914305], tiles='cartodbpositron', zoom_start=6)

    folium.Choropleth(
        geo_data=geodf_geo_json,
        name="choropleth",
        data=geodf_dict,
        columns=[col_id, col_to_plot],
        key_on="feature.id",
        fill_color="BuPu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="re-scaled Median of price(Millions COP)",
        bins = bins,
        reset = True
    ).add_to(map_)

    folium.LayerControl().add_to(map_)
    
    return map_, geodf_geo_json