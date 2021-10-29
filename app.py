#Stroke Detection WebApp
#Importing Libs


import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

path = Path()

st.write("""
#Stroke Detection
""")


#Getting the Data

df = pd.read_csv(path/"train.csv")

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


    age = st.sidebar.slider('age', 0, 100, 0)
    list2 = ['0', '1']
    st.sidebar.write('If you have hypertension select 1, if you do not have hypertension select 0')
    hypertension = st.sidebar.selectbox('hypertension', list2)
    st.sidebar.write('If you have heart disease select 1, if you do not have heart disease select 0')
    heart_disease = st.sidebar.selectbox('Heart Disease', list2)
    list3 = ['No', 'Yes']

    avg_glucose_level = st.sidebar.slider('Average Glucose level', 0, 250, 0)
    bmi = st.sidebar.slider('BMI', 0.0, 80.0, 0.0)


    #store a dictionary into a variable
    user_data = {'age': age,
                 'hypertension': hypertension,
                 'heart_disease': heart_disease,
                 'avg_glucose_level': avg_glucose_level,
                 'bmi': bmi
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
