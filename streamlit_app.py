import streamlit as st
import pandas as pd

st.markdown(
    """
    <style>
    .css-10trblm.e16nr0p33 {  
        text-align: center;
    }
    /* Center align the st.info element */
    div[data-testid="stAlert"] {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🐧 Penguin Detection App")

st.info("This is ML app to detect the species of penguin according to given data")

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
