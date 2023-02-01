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

    if app_mode=='Home': ## if someone selects the Home Tab
st.markdown('Dataset :') ## Display string formatted as Markdown.
st.write(df.head()) #write and display out dataset using the command df.head

    elif app_mode == 'Prediction':
## specify our inputs
    st.subheader('Fill in Iris to Get Prediction ')
    st.sidebar.header("Other Details :")
    prop = {'Iris Setosa': 1, 'Iris Versicolor': 2, 'Iris Virginica': 3}
    
    SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
    SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
    PetalLengthCm = st.slider('PetalLengthCm',0.0, 3.0)
    PetalWidthCm = st.slider('PetalWidthCm:', 0.0, 2.0)
    
    salary = st.sidebar.radio("Select Iris ",tuple(prop.keys()))

    Iris Setosa,Iris Versicolor,Iris Virginica=0,0,0
    if Iris == 'Iris Versicolor':
        Iris > 20
    elif Iris == 'Iris Setosa':
        Iris Setosa < 10
    else Iris == 'Iris Virginica':
        Iris Virginica >10 or <20


    subdata={
        'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm,
        
        'Iris':[Iris Setosa,Iris Versicolor,Iris Virginica],
        }

    features = [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, subdata['Iris'][0],subdata['Iris'][1], subdata['Iris'][2]]

    results = np.array(features).reshape(1, -1)
