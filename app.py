import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

data=pd.read_csv('diabetes.csv')
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

model=pickle.load(open('model.pkl','rb'))

st.title('Diabetes Prediction')
col1,col2=st.columns(2)

p=col1.text_input('Enter Pregnancies')
g=col2.text_input('Enter Glucose')
bp=col1.text_input('Enter BP')
sk=col2.text_input('Enter SKin Thickness')
i=col1.text_input('Enter Insulin')
bmi=col2.text_input('Enter BMI')
dpf=col1.text_input('Enter Diabetes Pedigree Function')
age=col2.text_input('Enter age')
if st.button('Predict'):
    data=[[p,g,bp,sk,i,bmi,dpf,age]]
    X_dummy=np.concatenate((X_test,data))
    sc=StandardScaler()
    X_dummy=sc.fit_transform(X_dummy)
    st.write(X_dummy[-1])
    out=model.predict([X_dummy[-1]])[0]
    st.write(out)
    if out==0:
        st.success('No Diabetes')
    elif out==1:
        st.error('Diabetes Found --> Eat Sugarfree (approximately), do exercise')
    
