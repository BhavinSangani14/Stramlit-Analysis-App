import streamlit as st 

def python_code(file_type, uploaded_file, column_1, column_2, plot_type, color_col=None):
    
    if color_col != None:   
        code = f"""#Python Code
import pandas as pd
import plotly.express as px
data = pd.read_{file_type}({uploaded_file.name}) 
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
data = pd.read_{file_type}({uploaded_file.name}) 
first_column = "{column_1}"
second_column = "{column_2}"
fig = px.{plot_type}(data, x = "{column_1}", y = "{column_2}", template="simple_white")
fig.show()
"""

    return code
        
