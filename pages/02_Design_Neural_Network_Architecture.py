import streamlit as st 
import tensorflow as tf 
import pandas as pd 
from tensorflow.keras.utils import plot_model
from generate_python_code import ann_viz, model_architecture_code
import base64


model_type = st.radio("Choose a model type", ["Feed Forward Neural Network", "Convolutional Neural Network"])

API_type = st.radio("Choose API Type", ["Sequential", "Functional"])

input_shape = (1,200)
num_hidden_layers = st.number_input("Select number of hidden dense layers", min_value=0, step=1)
next_button = st.button("Next")
if next:
    dic = {}
    hid_layers_list = ["hid"]
    
    for i in range(num_hidden_layers):
        neurons, activation, name = st.columns([1,1,1])
        globals()[f"num_neu_{i}"] = neurons.number_input(f"Select number of neurons in layer {i+1}", min_value=1, step=1)
        globals()[f"act_fun_{i}"] = activation.selectbox("Choose Activation function", ["Linear", "Relu", "Softmax", "Sigmoid", "Tanh"], key=f"{i}")
        globals()[f"layer_name_{i}"] = name.text_input("Choose a unique layer name", key={i})
        d = {"Neurons" : globals()[f"num_neu_{i}"], "Activation" : globals()[f"act_fun_{i}"], "Layer Name" : globals()[f"layer_name_{i}"]}
        dic[f"Layer_{i}"] = d
plot = st.button("Plot Model Architecture")
if plot:
    model = tf.keras.Sequential()
    for layer, feat in dic.items():
        num_neu = feat["Neurons"]
        acti_fun = feat["Activation"]
        layer_name = feat["Layer Name"]
        model.add(tf.keras.layers.Dense(num_neu, activation=acti_fun.lower(), name = layer_name))
    model.build(input_shape)
    dic_col, plot_col = st.columns([1,1])
    plot_model(model, show_shapes = True,to_file = "Images/model.png")
    dic_col.write(dic)
    plot_col.image("Images/model.png")
    model_code = model_architecture_code(dic)
    st.write("#python code")
    st.write(f"Hello world{'/n'.join(['1','2','3'])}")
    
        
    

# file_name = "Images/model.png"
# plot_model(model, show_shapes = True, to_file = file_name)
# ann_viz(model, filename=file_name)

# with open(file_name,"rb") as f:
#     base64_pdf = base64.b64encode(f.read()).decode('utf-8')
# pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
# st.markdown(pdf_display, unsafe_allow_html=True)
# st.image(file_name)
