import streamlit as st 
from PIL import Image

page_icon = Image.open("Images/page_icon.png")
st.set_page_config(page_title="Data Explorer", page_icon=page_icon, layout="wide")

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)