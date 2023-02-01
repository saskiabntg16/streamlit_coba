import streamlit as st
import pandas as pd
import pickle
from PIL import Image

model = pickle.load(open('model.pkl', 'rb'))

st.header("IRIS CLASSIFICATION PREDICTION:")

st.write("Please insert values, to get Iris class prediction")

SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
PetalLengthCm = st.slider('PetalLengthCm',0.0, 3.0)
PetalWidthCm = st.slider('SkiPetalWidthCm:', 0.0, 2.0)

data = {'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm}

features = pd.DataFrame(data, index=[0])

predict = model.predict(features)

st.subheader('Prediction Percentages:')

st.write('**Probablity of Iris Class being Iris-setosa is ( in % )**:',predict[0][0]*100)
st.write('**Probablity of Isis Class being Iris-versicolor is ( in % )**:',predict[0][1]*100)
st.write('**Probablity of Isis Class being Iris-virginica ( in % )**:',pred_predict[0][2]*100)


