import streamlit as st
import pandas as pd
import numpy as np
import pickle #to load a saved model
import base64 #to open .gif files in streamlit app

@st.cache(suppress_st_warning=True)
        
app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages

if app_mode=='Home':
    st.title('IRIS CLASSIFICATION PREDICTION :') 
    st.image('image.jpg')
    st.markdown('Dataset :')
    data=pd.read_csv('iris.csv')
    st.write(data.head())
    st.markdown('Iris Setosa VS Iris Versicolor VS Iris Virginica ')
    st.bar_chart(data[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']].head(20))


elif app_mode == 'Prediction':
    st.subheader('YOU need to fill all necessary informations in order to get a reply to your loan request !')

    SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
    SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
    PetalLengthCm = st.slider('PetalLengthCm',0.0, 3.0)
    PetalWidthCm = st.slider('PetalWidthCm:', 0.0, 2.0)
    data = {'SepalLengthCm': SepalLengthCm,
            'SepalWidthCm': SepalWidthCm,
            'PetalLengthCm': PetalLengthCm,
            'PetalWidthCm': PetalWidthCm}

    data1={
     'SepalLengthCm':SepalLengthCm,
     'SepalWidthCm':SepalWidthCm,
     'PetalLengthCm':PetalLengthCm,
     'PetalWidthCm':PetalWidthCm,
    }
 
    feature_list=[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]
    single_sample = np.array(feature_list).reshape(1,-1)
    if st.button("Click to Predict"):
     file_ = open("iris.jpg", "rb")
     contents = file_.read()
     data_url = base64.b64encode(contents).decode("utf-8")
     file_.close()
        
     file = open("iris.jpg", "rb")
     contents = file.read()
     data_url_no = base64.b64encode(contents).decode("utf-8")
     file.close()
        
     loaded_model = pickle.load(open('model.pkl', 'rb'))
     prediction = loaded_model.predict(single_sample)
     if prediction[0] == 0 :
         st.error(
 'According to our Calculations, you will not get the loan from Bank'
 )
         st.markdown(
 f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">',
 unsafe_allow_html=True,)
     elif prediction[0] == 1 :
         st.success(
 'Congratulations!! you will get the loan from Bank'
 )
         st.markdown(
 f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
 unsafe_allow_html=True,
 )
