import streamlit as st
import pandas as pd

st.title("🐧 Penguin Detection App")

st.info("This is ML app to detect the species of penguin according to given data")

st.markdown("""
    <style>
    .centered-info > div {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Applying the centered CSS to the info box
st.markdown('<div class="centered-info">', unsafe_allow_html=True)
st.info("This is a centered info message.")
st.markdown('</div>', unsafe_allow_html=True)

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/My-Data-Science-Projects/Penguin-Classification-ML-App/refs/heads/main/penguins_data.csv')
  df
  
  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
