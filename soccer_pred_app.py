import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


TEAMS = ["América (MG)","Atl Goianiense","Atl Paranaense","Atlético Mineiro","Avaí","Bahia","Botaforgo (RJ)", "Bragantino", "CSA", "Ceará",
         "Chapecoense", " Corinthians", "Coritiba","Cruzeiro", "Flamengo", "Fluminense", "Fortaleza", "Goiás", "Grêmio", "Internacional",
         "Palmeiras"," Paraná", "Ponte Preta", "Santos", "Sport Recife", "São Paulo" , "Vasco da Gama",  "Vitória"]


st.title('Brazilian Soccer Predictions')

st.write("Created by --- Felipe Rattes, Ewa Soltysik and Andres Sanche --- Le Wagon Batch 487")

#Create the Choose Option
home_team = st.selectbox("Choose Home Team", list(TEAMS))
away_team = st.selectbox("Choose Away Team", list(TEAMS))

df = pd.read_csv("Data/featured_final_data.csv")
df = df.drop(["Unnamed: 0"], axis=1)

if st.checkbox('Show dataframe'):
    st.write(df)


st.write('You selected:   ', home_team, "VS ", away_team)


expander = st.beta_expander("FAQ")
expander.write("This page will advise you on real brazilian football predictions on future matches.")
expander.write("Last update of dataframe 20/11/20")

def predictions(home_team, away_team):
    target = ["result"]
    X = df.drop(target, axis="columns")
    y = df[target]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=7)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    model.score(X_test, y_test)

    match25 = 1.0,0.890625,0.772727,0.716216,0.54,0.519231,0.0,0.0,0.066667,0.25,0.0,0.0,0.096774,0.159292,0.873832,0.884943,0.44,0.475,0.090909,0.2,0.526316,0.344828,0.208333,0.222222,0.2,0.2,0.615385,0.615385,0.696774,0.574468,0.565217,0.509434,0.819444,0.719512,0.714286,0.769231,0.3,0.28125,0.8,0.8,0.404762,0.398438,0.5,0.5,0.95122,0.875,0.675862,0.707064,0.305556,0.23913,0.702703,0.541667,0.518293,0.369565,0.308824,0.215753,0.375,0.264706,0.061224,0.085366,0.630435,0.685185,0.754386,0.754386,0.066667,0.0,0.111111,0.0,0.333333,0.0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0

    st.write(model.predict_proba([match25]))

    if home_team == "Atlético Mineiro":
        return st.write("da")

    return st.write("ea")


#st.write(home_team)

st.write(predictions(home_team, away_team))

