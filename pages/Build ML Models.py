# Import neccessary libraries
import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px
from PIL import Image
from utility_functions import python_code
import tensorflow as tf 
import streamlit.components.v1 as components 
from utility_functions import process_columns, Linear_Regression
print(tf.__version__)


# Page setup 
page_icon = Image.open("Images/page_icon.png")
st.set_page_config(page_title="Data Explorer", page_icon=page_icon, layout="wide")

#To hide menu and footer
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Render the h1 block, contained in a frame of size 200x200.
components.html("""<html>
                <head>
                <style> 
                h1 {text-align: center;}
                </style>
                </head>
                <body text = "black" bgcolor = "white"><h1> Build ML Models </h1>
                </body>
                </html>""", width=1000, height=90)



# User input file
uploaded_file = st.file_uploader("Choose a file", type=["csv","xlsx"])

# Create Pandas DataFrame
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
        cols, cat_cols, num_cols = process_columns(df)
    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
        cols, cat_cols, num_cols = process_columns(df)
    else:
        st.write("Please choose file any of (csv, excel) type")
        
    st.dataframe(df)
    

#Selection of Model type
if not uploaded_file:
    model_type = st.radio("Model Type", ["Regression", "Classification"], disabled=True)
else:
    model_type = st.radio("Model Type", ["Regression", "Classification"], disabled=False)

#Selection of ML Model type
if model_type == "Regression":
    ML_model = st.radio("Choose Model", ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Neural Network"], disabled=False)
if model_type == "Classification":
    ML_model = st.radio("Choose Model", ["KNN", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Neural Network"], disabled=False)
    
if ML_model == "Linear Regression" and model_type == "Regression" and uploaded_file:
    
    target_col = st.selectbox("Select Target Column", num_cols)
    st.write(f"Target Column : {target_col}")
    
    feat_col_options = cols.copy()
    feat_col_options.remove(target_col)
    feature_cols = st.multiselect("Select Feature Columns", feat_col_options)
    st.write(f"Feature_cols    :    {', '.join(feature_cols)}")
    
    build = st.button("Build Model")
    if build:
    #Build Model
        model, MSE = Linear_Regression(df, target_col, feature_cols)
        st.write(f"Model : Linear Regression")
        st.write(f"MSE of Linear Regression Model : {MSE}")
        

if ML_model == "Logistic Regression" and model_type == "Classification":
    
    target_col = st.selectbox("Select Target Column", cat_cols)
    st.write(f"Target Column : {target_col}")
    
    feat_col_options = cols.copy()
    feat_col_options.remove(target_col)
    feature_cols = st.multiselect("Select Feature Columns", feat_col_options)
    st.write(f"Feature_cols    :    {', '.join(feature_cols)}")
    
    build = st.button("Build Model")
    if build:
    #Build Model
        model, MSE = Logistic_Regression(df, target_col, feature_cols)
        st.write(f"Model : Logistic Regression")
        st.write(f"MSE of Linear Regression Model : {MSE}")
    