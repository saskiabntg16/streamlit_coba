import streamlit as st
import pandas as pd
import numpy as np
import pickle

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction'])

if app_mode=='Home': 
    st.title('Iris Classification Prediction') 
    st.markdown('Dataset :') 
    df=pd.read_csv('iris.csv') #Read our data dataset
    st.write(df.head()) 
