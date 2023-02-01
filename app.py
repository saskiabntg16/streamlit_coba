import streamlit as st
import pandas as pd
import numpy as np
import pickle #to load a saved model
import base64 #to open .gif files in streamlit app

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
    st.title('IRIS CLASSIFICATION PREDICTION :') 
    st.image('image.jpg')
    st.markdown('Dataset :')
    data=pd.read_csv('iris.csv')
    st.write(data.head())
    st.markdown('Iris Setosa VS Iris Versicolor VS Iris Virginica ')
    st.bar_chart(data[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']].head(20))


elif app_mode == 'Prediction':
    st.image('slider-short-3.jpg')
    st.subheader('YOU need to fill all necessary informations in order to get a reply to your loan request !')
    st.sidebar.header("Informations about the client :")
    gender_dict = {"Male":1,"Female":2}
    feature_dict = {"No":1,"Yes":2}
    edu={'Graduate':1,'Not Graduate':2}
    prop={'Rural':1,'Urban':2,'Semiurban':3}

    SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
    SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
    PetalLengthCm = st.slider('PetalLengthCm',0.0, 3.0)
    PetalWidthCm = st.slider('PetalWidthCm:', 0.0, 2.0)
    data = {'SepalLengthCm': SepalLengthCm,
            'SepalWidthCm': SepalWidthCm,
            'PetalLengthCm': PetalLengthCm,
            'PetalWidthCm': PetalWidthCm}
    Gender=st.sidebar.radio('Gender',tuple(gender_dict.keys()))
    Married=st.sidebar.radio('Married',tuple(feature_dict.keys()))
    Self_Employed=st.sidebar.radio('Self Employed',tuple(feature_dict.keys()))
    Dependents=st.sidebar.radio('Dependents',options=['0','1' , '2' , '3+'])
    Education=st.sidebar.radio('Education',tuple(edu.keys()))
    Property_Area=st.sidebar.radio('Property_Area',tuple(prop.keys()))
    class_0 , class_3 , class_1,class_2 = 0,0,0,0 
    if Dependents == '0':
        class_0 = 1
    elif Dependents == '1':
        class_1 = 1
    elif Dependents == '2' :
        class_2 = 1
    else:
        class_3= 1
    
    Rural,Urban,Semiurban=0,0,0
    if Property_Area == 'Urban' :
        Urban = 1
    elif Property_Area == 'Semiurban' :
        Semiurban = 1
    else :
        Rural=1
 

    data1={
     'Gender':Gender,
     'Married':Married,
     'Dependents':[class_0,class_1,class_2,class_3],
     'Education':Education,
     'SepalLengthCm':SepalLengthCm,
     'SepalWidthCm':SepalWidthCm,
     'PetalLengthCm':PetalLengthCm,
     'PetalWidthCm':PetalWidthCm,
     'Self Employed':Self_Employed,
     'Property_Area':[Rural,Urban,Semiurban],
    }
 
    feature_list=[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,get_value(Gender,gender_dict),get_fvalue(Married),data1['Dependents'][0],data1['Dependents'][1],data1['Dependents'][2],data1['Dependents'][3],get_value(Education,edu),get_fvalue(Self_Employed),data1['Property_Area'][0],data1['Property_Area'][1],data1['Property_Area'][2]]
 
    single_sample = np.array(feature_list).reshape(1,-1)
    if st.button("Click to Predict"):
     file_ = open("6m-rain.gif", "rb")
     contents = file_.read()
     data_url = base64.b64encode(contents).decode("utf-8")
     file_.close()
        
     file = open("green-cola-no.gif", "rb")
     contents = file.read()
     data_url_no = base64.b64encode(contents).decode("utf-8")
     file.close()
        
     loaded_model = pickle.load(open('Random_Forest.sav', 'rb'))
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
