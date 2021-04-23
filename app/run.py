import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib
import plotly.graph_objs as go_
from flask import current_app, flash, jsonify, make_response, redirect, request, url_for
import folium
from folium import plugins
from folium.plugins import HeatMap

#Import custom class and functions:

import sys
sys.path.append("./customized_class")
from report import create_missing_report
from create_choropleth import create_choropleth
from create_geodf import construct_geodf
#from load_map import load_source_map

# load data for missing report

engine = create_engine('sqlite:///../data/PropertiesPrices.db')
df = pd.read_sql_table('Cleaned_prices', engine)
options = list(df['property_type'].unique())
list_features = ["rooms", "bedrooms","bathrooms","surface_total","surface_covered","lat","lon"]
types_for_maps = ['Casa', 'Apartamento']
types_for_errors =['Casa', 'Apartamento']

#Create a base map for index page:

obj_map = folium.Map(location=[5.170035, -74.914305], tiles='cartodbpositron', zoom_start=6)
data = df.head(1000)
data = data[(data['missing_lon']==0) & (data['missing_lat']==0)]
HeatMap(data=data[['lat', 'lon']], radius=10).add_to(obj_map)
obj_map.save("templates/map.html")
#base_map = load_source_map(obj_map)

# load model

#model = joblib.load("../models/classifier.pkl")
#model = model.best_estimator_

#Star flask app 

app = Flask(__name__)

# index webpage displays cool visuals and receives user input text for model

@app.route('/')
@app.route('/index')
def index():
    
    segment_by = 'property_type'
    df_report = create_missing_report(df,list_features,segment_by)
    columns_for_map = ['perc_'+x for x in list_features]

    fig = go_.Figure(data=go_.Heatmap(z = df_report[columns_for_map],
                                    x = list_features,
                                    y = list(df_report.index),hoverongaps = False))
    graphs = [
        
        fig
        
    ]
    
    # encode plotly graphs in JSON
    
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON, options = options, list_features=list_features,
                          types = types_for_maps, source_map="\map",
                          types_errors = types_for_errors)

# web page that handles user query and displays model results

@app.route('/ajax_add', methods = ['POST','GET'])
def ajax_add():
    if request.method =='POST':
        
        response = request.form
        categories = response.getlist('categories[]')
        features = response.getlist('features[]')
        segment_by = 'property_type'
        
        df_=df[df['property_type'].isin(categories)]
        df_report = create_missing_report(df_,features,segment_by)
        
        columns_for_map = ['perc_'+x for x in features]
        fig = go_.Figure(data=go_.Heatmap(z = df_report[columns_for_map],
                                        x = features,
                                        y = list(df_report.index),hoverongaps = False))
        graphs = [

            fig

        ]

        # encode plotly graphs in JSON
        
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/ajax_add_2', methods = ['POST','GET'])
def ajax_add_2():
    if request.method =='POST':
        
    # load data for choropleths
        response = request.form
        properties_types = response.getlist('property_type[]')
        geodf = construct_geodf(df,properties_types,'price',"./source/depto.shp")
        map_, geodf_geo_json = create_choropleth(geodf,'price','NOMBRE_DPT',1000000)
        map_.save("templates/Choropleth_map.html")
    return render_template('Choropleth_map.html')

@app.route('/ajax_add_3', methods = ['POST','GET'])
def ajax_add_3():
    if request.method =='POST':
        
    # load data for choropleths
        response = request.form
        properties_types = response.getlist('property_type[]')
        df_errors = pd.read_csv("./source/test_errors.csv")
        geodf = construct_geodf(df_errors,properties_types,"squared_errors","./source/depto.shp")
        map_, geodf_geo_json = create_choropleth(geodf,"squared_errors",'NOMBRE_DPT',0.001)
        map_.save("templates/Choropleth_map_errors.html")
    return render_template('Choropleth_map_errors.html') 

@app.route('/map')
def map():
    return render_template("map.html")
@app.route('/Choropleth_map')
def Choropleth_map():
    return render_template("Choropleth_map.html")
@app.route('/Choropleth_map_errors')
def Choropleth_map_errors():
    return render_template("Choropleth_map_errors.html")
@app.route('/')
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()