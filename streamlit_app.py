import streamlit as st
import pandas as pd

st.title("🐧 Penguin Detection App 🐧")

st.info("This app detect the species of penguin according to given data")

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/My-Data-Science-Projects/Penguin-Classification-ML-App/refs/heads/main/penguins_data.csv')
  df
  
