# Price of real estate properties in Colombia

### Summary:

In this project we carry out the analysis of the data available in Kaggle "Colombia Housing Properties Price".

You can download the file from [here](https://www.kaggle.com/julianusugaortiz/colombia-housing-properties-price/download)

This dataset includes real information on property prices in Colombia. Feel free to download the zip file from above link, unzip it, and then place the "co_properties.csv" file inside the **\data** directory of this project.

Before proceeding to use this app it is very important that you take a look at my other github project where I have implemented the exploratory data analysis stage [here](https://github.com/HectorMontes10/Properties_price)

This project does not focus so much on these aspects but on the modeling and deployment in a web app of some visualizations using folium maps to support the understanding of the data and the evaluation of the model.

However, how this project is part of a broader business need, that consists of building regression models for real estate properties in Colombia, you will find on the website some questions that were raised, even though not all the visualizations are on the site.

These are the useful things you will find in this project:

1. The project is a simple app, that uses javascript, and ajax for the frontend, python for the backend, and flask for the communication between the user and the server. In that sense, this is a simple way to understand the basic stages of building a website under the architecture already mentioned. Keeps it simple but complete.

2. We have entered the additional challenge of drawing thematic maps based on data and publishing them on the wep. For this, we have used two great resources: geopandas and folium. These packages together are powerful to present nice visualizations in your app giving you control of the data layer (geopandas) and presentation (folium). The examples are simple: A choroplet for median prices and errors of a prediction model, but complete enough to demonstrate its power.

3. We train a prediction model using pipelines and gridsearch. The first to automate the cleaning and data preparation stage, making the code very readable and maintainable, and avoiding data linking between the training and testing stages. The second to improve the tuning of parameters through search using cross validation.

I sincerely hope that the project is useful for you to understand how an end-to-end web solution is structured. Feel free to use this code for educational or learning purposes, And edit it as much as you consider necessary.

Feedback to improve it will be highly appreciated. 

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/co_properties.csv data/regions.csv data/PropertiesPrices.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_regressor.py data/PropertiesPrices.db models/regressor.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or localhost:3001

Make sure to create the rules in the firewall to allow http / TCP traffic and grant python permissions so that the execution of the app locally works correctly. Enable listening on port 3001.

Enjoy it

### Files:

The structure for this project is:

- app
   - template
      - master.html  # main page of web app
      - map.html  # The default map before to send information in the app for Choropleth_map
      - Choropleth_map # The Choropleth map produce based on the user interaction with the app
   - static
      - bootstrap-select.css  #css style for selector list in the app
      - bootstrap-select.js  #javascript code for aesthetic presentation of the selection list (this resources [here](http://paulrose.com/bootstrap-select-sass/)
      - geodf.gif (an animated gif to explain the geodatabase aspect)
      - geodf.html (an html table to explain the geodatabase aspect)
   - source
      - depto.dbf
      - depto.prj
      - depto.shp  # Department layer usefull for construct the Choropleth_map.html (this resources [here](https://sites.google.com/site/seriescol/shapes)
      - depto.shx 
   - customized_class
      - create_choropleth.py  #Create the Choropleth based on geodatabase
      - create_geodf.py  #Construct an geodatabase based on shapefile an dataframe of features in pandas.
      - dummy_estimator.py  #Usefull train the model with several estimator. The credit are  for [Brian Spiering](https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search)
      - input_data.py  #Customized transform for input median over missing values and quantile98 over extreme data
      - report.py  #Construct a missing values report for display in the app using plotly graph.     
   - run.py  # Flask file that runs app
- data
   - co_properties.csv  #Data to process 
   - regions.csv  #Data to process
   - process_data.py #python script usefull for preprocessing data (clean database of prices)
   - PropertiesPrices.db   # Database where clean database prices is stored
   - preprocess_data.ipynb  #A jupyter notebook for inspect data used in this model (take a look before starting this project)
- models
   - train_regressor.py  #python script usefull for training model over data (clean database of prices) 
   - regressor.pkl  #saved best model
   - train_regressor.ipynb  #A jupyter notebook for study modelling stage(take a look before starting this project)

Notes:

1. Deleting regressor.pkl model is possible but once this is done you will have to run train_regressor.py again to obtain a new trained model to be used by your application. Model training may take time, depending on the capabilities of your server. training_regressor.py implements gridsearch for tuning parameters, you can edit the parameter search space if you wish for more intensive or less intensive training. You can also change the list of estimators used. In this case we have focused on three: a classical multiple linear regression model, a stochastic descending gradient model, and a vector support regressor. The documentation can be consulted on the sklearn site:

- [Stochastic Gradient Descending](https://scikit-learn.org/stable/modules/sgd.html)
- [Support Vector Regressor](https://scikit-learn.org/stable/modules/svm.html#svm-regression)
- [Linear Model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

2. PropertiesPrices.db is a database that can be deleted, but once this is done it will be necessary to run process_data.py again to create a new clean database of prices. You can edit the script to customize the cleanup tasks on the co_properties.csv and regions.csv files.

3. The documentation of the other functions can be found inside the respective .py file

### Requeriments:

- numpy 1.19.2
- pandas==1.2.4
- sklean==0.22.2
- sqlalchemy==1.4.9
- plotly==4.14.3
- flask==1.1.2
- folium==0.12.0
- geopandas==0.9.0
- scipy==1.19.2
