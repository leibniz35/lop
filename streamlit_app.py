#Stroke Detection WebApp

#Importing Libs

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
from urllib.request import urlretrieve


url = ("http://dl.dropboxusercontent.com/s/ykgj9vnkoj6cef1/stroketrain.csv?raw=1")
filename = "stroketrain.csv"
urlretrieve(url,filename)




st.write("""
#Stroke Detection
""")

#Getting the Data

df = pd.read_csv('stroketrain.csv')

#setting a subheader
st.subheader('Data Information: ')
#Show a data as a table
st.dataframe(df)

#show statistics on the table
st.write(df.describe())

#Splitting the data
X = df.iloc[:, 1:10].values
Y = df.iloc[:,-1].values

#Splitting the dataset into %80 Training and %20 Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20 , random_state=0)

#Getting input from the user
def get_user_input():
    list1 = ['Male', 'Female', 'Other']
    gender = st.sidebar.selectbox('Gender', list1)
    age = st.sidebar.slider('age', 0, 100, 0)
    list2 = ['0', '1']
    st.sidebar.write('If you have hypertension select 1, if you do not have hypertension select 0')
    hypertension = st.sidebar.selectbox('hypertension', list2)
    st.sidebar.write('If you have heart disease select 1, if you do not have heart disease select 0')
    heart_disease = st.sidebar.selectbox('Heart Disease', list2)
    list3 = ['No', 'Yes']
    ever_married = st.sidebar.selectbox('Ever Married', list3)
    list4 = ['children', 'Govt_jov', 'Never_worked', 'Private', 'Self-employed']
    work_type = st.sidebar.selectbox('Work Type', list4)
    list5 = ['Rural', 'Urban']
    Residence_type = st.sidebar.selectbox('Residence Type', list5)
    avg_glucose_level = st.sidebar.slider('Average Glucose level', 0, 250, 0)
    bmi = st.sidebar.slider('BMI', 0.0, 80.0, 0.0)
    list6 = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
    smoking_status = st.sidebar.selectbox('Smoking Status', list6)

    #store a dictionary into a variable
    user_data = {'gender': gender,
                 'age': age,
                 'hypertension': hypertension,
                 'heart_disease': heart_disease,
                 'ever_married': ever_married,
                 'work_type': work_type,
                 'Residence_type': Residence_type,
                 'avg_glucose_level': avg_glucose_level,
                 'bmi': bmi,
                 'smoking_status': smoking_status

    }

    #Transform the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features
#Storing the user input
user_input = get_user_input()

#Setting a subheader and display the user input
st.subheader('User Input:')
st.write(user_input)

#Creating and training the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show the models metrics
st.subheader('Model Test Accuracy Score: ')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100)+'%' )

#Store the models predictions in a variable
prediction =  RandomForestClassifier.predict(user_input)

#Setting a subheader and displaying the detection
st.subheader('Detection: ')
st.write(prediction)