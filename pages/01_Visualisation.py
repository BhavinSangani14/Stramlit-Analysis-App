# Import neccessary libraries
import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px
from PIL import Image
# from utility_functions import python_code
import tensorflow as tf 
import streamlit.components.v1 as components 
print(tf.__version__)


# Page setup 
page_icon = Image.open("Images/page_icon.png")
st.set_page_config(page_title="Visualization", page_icon=page_icon, layout="wide")


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
                <body text = "black" bgcolor = "white"><h1> Visualization </h1>
                </body>
                </html>""", width=1000, height=90)


# Utility Function
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

# User input file
uploaded_file = st.file_uploader("Choose a file", type=["csv","xlsx"])

# Create Pandas DataFrame
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
    else:
        st.write("Please choose file any of (csv, excel) type")
        
    st.dataframe(df)

if not uploaded_file:
    st.radio("Plot Type", ["Scatter", "Line Chart", "Bar Chart", "Area Chart", "Funnel Chart"], disabled=True)
else:
    plot_type = st.radio("Plot Type", ["Scatter", "Line Chart", "Bar Chart", "Area Chart", "Funnel Chart"], disabled=False)
    
    if plot_type == "Scatter":
        columns = list(df.columns)
        column_1 = st.selectbox("Choose 1st column(X)", columns)
        column_2 = st.selectbox("Choose 2nd column(Y)", columns)
        scatter_col, option_col = st.columns([4,1])
        color_col = option_col.selectbox("Choose column", columns)
        scatter_plot = px.scatter(df, x = column_1, y = column_2, color = color_col, template="simple_white")
        scatter_col.plotly_chart(scatter_plot, use_container_width=True)
        generate_code = st.button("Generate Python Code")
        if generate_code:
            code = python_code(file_type, uploaded_file, column_1, column_2, "scatter", color_col)
            st.code(code, language = "python")
            
        
            
    if plot_type == "Line Chart":
        columns = list(df.columns)
        column_1 = st.selectbox("Choose 1st column", columns)
        column_2 = st.selectbox("Choose 2nd column", columns)
        df.sort_values(by = column_1, inplace = True)
        line_chart = px.line(df, x = column_1, y = column_2)
        st.plotly_chart(line_chart, use_container_width=True)
        generate_code = st.button("Generate Python Code")
        if generate_code:
            code = python_code(file_type, uploaded_file, column_1, column_2, "line")
            st.code(code, language = "python")
        
    if plot_type == "Bar Chart":
        columns = list(df.columns)
        column_1 = st.selectbox("Choose 1st column", columns)
        column_2 = st.selectbox("Choose 2nd column", columns)
        bar = px.bar(df, x = column_1, y = column_2)
        st.plotly_chart(bar, use_container_width=True)
        generate_code = st.button("Generate Python Code")
        if generate_code:
            code = python_code(file_type, uploaded_file, column_1, column_2, "bar")
            st.code(code, language = "python")
            
    if plot_type == "Area Chart":
        columns = list(df.columns)
        column_1 = st.selectbox("Choose 1st column", columns)
        column_2 = st.selectbox("Choose 2nd column", columns)
        # df.sort_values(by = column_1, inplace = True)
        area_chart = px.area(df, x = column_1, y = column_2)
        st.plotly_chart(area_chart, use_container_width=True)
        generate_code = st.button("Generate Python Code")
        if generate_code:
            code = python_code(file_type, uploaded_file, column_1, column_2, "area")
            st.code(code, language = "python")
            
    if plot_type == "Funnel Chart":
        try:
            columns = list(df.columns)
            column_1 = st.selectbox("Choose 1st column(X)", columns)
            column_2 = st.selectbox("Choose 2nd column(Y)", columns)
            df = df[[column_1, column_2]]
            df = df.groupby(by = column_2).sum().reset_index().sort_values(by = column_1, ascending=False)
            area_chart = px.funnel(df, x = column_1, y = column_2)
            st.plotly_chart(area_chart, use_container_width=True)
            generate_code = st.button("Generate Python Code")
            if generate_code:
                code = python_code(file_type, uploaded_file, column_1, column_2, "funnel")
                st.code(code, language = "python")
        except:
            st.error("Please select different columns")
        
        
        
        
    
    
    
