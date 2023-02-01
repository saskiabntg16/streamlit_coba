import streamlit as st
import pandas as pd
import numpy as np
import pickle #to load a saved model
import base64 #to open .gif files in streamlit app

model = pickle.load(open('model.pkl', 'rb'))

@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
        
app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages

if app_mode=='Home':
    st.title('IRIS CLASSIFICATION PREDICTION') 
    st.write("Saskia Bintang Maharani")
    st.write("2019230047")
    st.markdown('Dataset :')
    data=pd.read_csv('iris.csv')
    st.write(data.head())
    st.markdown('Iris Setosa VS Iris Versicolor VS Iris Virginica ')
    st.bar_chart(data[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']].head(20))

elif app_mode == 'Prediction':
    st.image('iris.png')
    st.write("Please Insert Values, to Get Iris Classification Prediction")
    SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
    SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
    PetalLengthCm = st.slider('PetalLengthCm',0.0, 3.0)
    PetalWidthCm = st.slider('PetalWidthCm:', 0.0, 2.0)
data = {'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm}

features = pd.DataFrame(data, index=[0])

pred_proba = model.predict_proba(features)
#or
prediction = model.predict(features)

st.subheader('Prediction Percentages:') 
st.write('**Probablity of Iris Class being Iris-setosa is ( in % )**:',pred_proba[0][0]*100)
st.write('**Probablity of Isis Class being Iris-versicolor is ( in % )**:',pred_proba[0][1]*100)
st.write('**Probablity of Isis Class being Iris-virginica ( in % )**:',pred_proba[0][2]*100)




