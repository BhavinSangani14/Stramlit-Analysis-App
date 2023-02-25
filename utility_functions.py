import pandas as pd 
import streamlit as st 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np 


## Function to generate python code for visualization plots
def python_code(file_type, uploaded_file, column_1, column_2, plot_type, color_col=None):
    
    if color_col != None:   
        code = f"""#Python Code
import pandas as pd
import plotly.express as px
data = pd.read_{file_type}("{uploaded_file.name}") 
first_column = "{column_1}"
second_column = "{column_2}"
color_column = "{color_col}"
fig = px.{plot_type}(data, x = "{column_1}", y = "{column_2}", color = "{color_col}", template="simple_white")
fig.show()
"""

    if color_col == None:   
        code = f"""#Python Code
import pandas as pd
import plotly.express as px
data = pd.read_{file_type}("{uploaded_file.name}") 
first_column = "{column_1}"
second_column = "{column_2}"
fig = px.{plot_type}(data, x = "{column_1}", y = "{column_2}", template="simple_white")
fig.show()
"""

    return code
        

## Function to generate python code for ML Model Architecture
def model_architecture_code(dic):
    layers_list = []
    for layer, feat in dic.items():
        num_neu = feat["Neurons"]
        acti_fun = feat["Activation"]
        layer_name = feat["Layer Name"]
        layers_list.append(f"model.add(tf.keras.layers.Dense(units = {num_neu}, activaion = '{acti_fun}', name = '{layer_name}'))")
        
    code = """/n""".join(layers_list)
    return code
        
    


## Function to separate the numerical and categorical columns 
def process_columns(df):
    
    # All columns
    cols = list(df.columns)
    
    #Categorical columns
    cat_cols = list(df.select_dtypes(include = ["object"]).columns)
    
    #Numerical columns
    num_cols = list(df.drop(columns = cat_cols).columns)
    
    return cols, cat_cols, num_cols 


def Linear_Regression(df, target_col, feature_columns):
    
    y = df[target_col]
    df = df[feature_columns]
    #fill null values
    df.fillna(df.mean(), inplace = True)
    
    # creating Feature matrix and target column
    X = df[feature_columns]
    
    
    #creating categorical and Numerical columns list
    cols, cat_cols, num_cols = process_columns(X)
    
    #Onehot encoding of catgegorical columns
    if len(cat_cols)!=0:
        cat_df = pd.get_dummies(X[cat_cols], drop_first=True)
        X = pd.concat([X, cat_df], axis=1)
        X.drop(columns = cat_cols, inplace = True)
    
    #Scaling numerical columns
    if len(num_cols)!= 0:
        scaler = StandardScaler()
        scaler.fit(X[num_cols].values)
        X[num_cols] = scaler.transform(X[num_cols].values)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    #make predction
    pred = model.predict(X)
    
    #calculate MSE
    MSE = mean_squared_error(pred, y)
    
    return model, np.round(MSE, 2)



def Logistic_Regression(df, target_col, feature_columns):
    
    #fill null values
    df.fillna(method = "ffill", inplace = True)
    
    # creating Feature matrix and target column
    X = df[feature_columns]
    y = df[target_col]
    
    #creating categorical and Numerical columns list
    cols, cat_cols, num_cols = process_columns(X)
    
    #Onehot encoding of catgegorical columns
    if len(cat_cols)!=0:
        cat_df = pd.get_dummies(X[cat_cols], drop_first=True)
        X = pd.concat([X, cat_df], axis=1)
        X.drop(columns = cat_cols, inplace = True)
    
    #Scaling numerical columns
    if len(num_cols)!= 0:
        scaler = StandardScaler()
        scaler.fit(X[num_cols].values)
        X[num_cols] = scaler.transform(X[num_cols].values)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    #make predction
    pred = model.predict(X)
    
    #calculate MSE
    MSE = mean_squared_error(pred, y)
    
    return model, np.round(MSE, 2)
    
    
    
    
    
    
    
    



    