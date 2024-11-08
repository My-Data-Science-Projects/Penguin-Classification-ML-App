import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.markdown(
    """
    <div style="text-align: center;">
        <h1>🐧 Penguin Detection App 🐧</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[data-testid="stAlert"] {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.info("This is ML app to detect the species of penguin according to given data")

with st.expander('Data'):
  df = pd.read_csv('https://raw.githubusercontent.com/My-Data-Science-Projects/Penguin-Classification-ML-App/refs/heads/main/penguins_data.csv')
  df
  X_raw = df.drop('species', axis=1)
  y_raw = df.species

with st.expander('Data Visualization'):
  st.info("Bill Length (mm) vs. Body Mass (g)")
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

  st.info("Flipper Length (mm) vs. Body Mass (g)")
  st.scatter_chart(data=df, x='flipper_length_mm', y='body_mass_g', color='species')

with st.sidebar:
  st.header('Input Features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))
  
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input Data'):
  input_df

encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})


penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
# st.success("**Predicted Species :** " + str(penguins_species[prediction][0]))
st.markdown(
    f'<div style="font-size:24px; background-color:green;"><b>Predicted Species :</b> {penguins_species[prediction][0]}</div>',
    unsafe_allow_html=True
)

st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, 
             hide_index=True,
            use_container_width=True)
