import pandas as pd 


def process_columns(df):
    
    # All columns
    cols = list(df.columns)
    
    #Categorical columns
    cat_cols = list(df.select_dtypes(include = ["object"]).columns)
    
    #Numerical columns
    num_cols = list(df.drop(columns = cat_cols).columns)
    
    return cols, cat_cols, num_cols 
    