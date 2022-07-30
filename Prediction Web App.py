# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",150)

# %%
import streamlit as st
import pickle as pkl

# %%
x2 = pd.read_csv("trainfinal.csv")
y2 = pd.read_csv("yfinal.csv")

# %%
model = pkl.load(open("the_model.pkl", "rb"))

# %%
st.title("Bike Sharing Prediction")
st.image("bike.jpg")

# %%
st.write("""
# Bike Sharing Prediction Machine Learning Web App

Welcome to my machine learning app where you can predict the number of bike shares based on historical data





## Features Describtion

- Hour : the time you want to predict [1-24]

- Temperature : what the temperature is ?

- Workingday : is today is a working day ? [1 or 0]

- Season_summer : is the season is summer ? [1 or 0]

- Month : what month we are at ? [1-12]

- Humidity : how much humidity is there ? 

- Season_winter : is the season is winter ? [1 or 0]

- Weathersit rain : is the weather condition is rain ? [1 or 0]
""")

# %%
st.write("## DataFrame Overview")

st.dataframe(x2.sample(7))



# %%


hour = st.number_input("Enter the time in hours",min_value=0,max_value=23)
temp = st.number_input("Enter the temperature",min_value=0,max_value=1)
workingday = st.number_input("is today a working day? [1,or,0]",min_value=0,max_value=1,step=1)
seasonsummer = st.number_input("is the season is summer? [1,or,0]",min_value=0,max_value=1,step=1)
month = st.number_input("what the current month [1 to 12]",min_value=1,max_value=12)
humidity = st.number_input("what is the humidity?",min_value=0,max_value=1)
seasonwinter = st.number_input("is the current season is winter? [1,or,0]",min_value=0,max_value=1,step=1)
weathersitrain = st.number_input("is the weather rainy? [1,or,0]",min_value=0,max_value=1,step=1)
values = [hour,temp,workingday,seasonsummer,month,humidity,seasonwinter,weathersitrain]
values = np.array(values)
if st.button("Predict the number of share rides today"):
    pred = model.predict(values.reshape(1, -1))
    st.success(f"The expected number of share rides today is {int(pred)} rides with 80% Probability ")
    
streamlit run Prediction Web App.py    
