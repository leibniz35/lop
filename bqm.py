#Stroke Detection WebApp

#Importing Libs

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
from urllib.request import urlretrieve
from sklearn.ensemble import RandomForestClassifier as RFC


st.write("""
#Stroke Detection
""")
url = ("http://dl.dropboxusercontent.com/s/ykgj9vnkoj6cef1/stroke.csv?raw=1")

filename = "stroketrain.csv"
urlretrieve(url,filename)
#Getting the Data

df = pd.read_csv('stroketrain.csv')

st.dataframe(df)
#remove the rows with missing values
df = df.drop(['id'], axis=1)

df = df.fillna(df['bmi'].median())
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

for i in range(df.shape[1]):
    if df.iloc[:,i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(df.iloc[:,i].values))
        df.iloc[:,i] = lbl.transform(list(df.iloc[:,i].values))


#setting a subheader
st.subheader('Data Information: ')
#Show a data as a table
st.dataframe(df)

#show statistics on the table
st.write(df.describe())

#Splitting the data
X = df.iloc[:, 0:10].values
Y = df.iloc[:,-1].values

#Splitting the dataset into %80 Training and %20 Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20 , random_state=0)



#Getting input from the user
def get_user_input():
    list1 = ['0', '1', '2']
    st.sidebar.write("0 = Female, 1 = Male, 2 = Other")
    gender = st.sidebar.selectbox('Gender', list1)
    age = st.sidebar.slider('age', 0, 100, 0)
    list2 = ['0', '1']
    st.sidebar.write('If you have hypertension select 1, if you do not have hypertension select 0')
    hypertension = st.sidebar.selectbox('hypertension', list2)
    st.sidebar.write('If you have heart disease select 1, if you do not have heart disease select 0')
    heart_disease = st.sidebar.selectbox('Heart Disease', list2)
    list3 = ['0', '1']
    st.sidebar.write("1 = Yes, 0 = No")
    ever_married = st.sidebar.selectbox('Ever Married', list3)
    list4 = ['0', '1', '2', '3', '4']
    st.sidebar.write("0 = govt_job, 1 = Never Work, 2 = Private, 3 = self employed, 4 = child")                
    work_type = st.sidebar.selectbox('Work Type', list4)
    list5 = ['0', '1']
    st.sidebar.write("0 = Rural, 1 = urban")
    Residence_type = st.sidebar.selectbox('Residence Type', list5)
    avg_glucose_level = st.sidebar.slider('Average Glucose level', 0, 250, 0)
    bmi = st.sidebar.slider('BMI', 0.0, 80.0, 0.0)
    list6 = ['0', '1', '2', '3']
    st.sidebar.write("0 = Unknown, 1 = formerly smokes, 2 = Never, 3 = Smokes")
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

    
    
    
    
    
x = df.drop('stroke', axis=1)
t = df['stroke']


x = np.array(x)
t = np.array(t)

t = t.ravel()
x = x.astype('float32')
t = t.astype('int32')
print('x shape:', x.shape)
print('t shape:', t.shape)

clf = RFC(n_estimators=192,
          criterion='gini',# 'gini' or 'entropy'
          max_depth=19,
          min_samples_split=2,
          max_features='auto',# 'auto'(='sqrt') or 'log2'
          n_jobs=-1,
          random_state=2525,
          verbose=1)# 0 or 1


predict = clf.predict_proba(x)[:, 1] # This grabs the positive class prediction
score = roc_auc_score(t, predict)
prediction = RandomForestClassifier.predict(user_input)


st.subheader('Detection: ')
st.write(prediction)
