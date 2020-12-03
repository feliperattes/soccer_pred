from flask import Flask, escape, request
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

#@app.route('/')
def hello():
    # get param from http://127.0.0.1:5000/?name=value
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'

TEAMS = ["Atl Goianiense","Atl Paranaense","Atlético Mineiro","Bahia","Botafogo (RJ)", "Bragantino", "Ceará",
         " Corinthians", "Coritiba", "Flamengo", "Fluminense", "Fortaleza", "Goiás", "Grêmio", "Internacional",
         "Palmeiras", "Santos", "Sport Recife", "São Paulo" , "Vasco da Gama"]


st.title('Brazilian Soccer Predictions')

st.write("Created by --- Felipe Rattes, Ewa Soltysik and Andres Sanche --- Le Wagon Batch 487")

#Create the Choose Option
home_team = st.selectbox("Choose Home Team", list(TEAMS))
away_team = st.selectbox("Choose Away Team", list(TEAMS))

data = pd.read_csv("Data/featured_data.csv")
final_data = pd.read_csv("Data/final_data.csv")
#final_data = final_data.drop(["Unnamed: 0"], axis=1)


a = final_data.loc[:, 'home_avrg_Home_Poss': 'result'].columns[:-1]
b = final_data.loc[:, 'América (MG)_HT':'Vitória_HT'].columns
c = final_data.loc[:, 'América (MG)_AT':].columns
d = final_data.loc[:, 'result': 'number_of_corners'].columns

norm = MinMaxScaler()
norm = norm.fit(final_data[a])
norm_features = norm.transform(final_data[a])
x = pd.DataFrame(norm_features, columns=a)

final_data = pd.concat([x, final_data[b], final_data[c], final_data[d]], axis=1)

if st.checkbox('Show dataframe'):
    st.write(final_data)


st.write('You selected:   ', home_team, "VS ", away_team)



X = final_data.drop(['result', 'number_of_goals', 'number_of_corners'], axis=1)


# LOGISTIC REGRESSION FOR THE OUTCOME OF MATCH

#define the target for this model
y_result = final_data['result']

#split the data
X_train, X_test, y_result_train, y_result_test = train_test_split(X, y_result, test_size=0.30)

#create and fil the model
model_lr_outcome = LogisticRegression(C = 0.1)
model_lr_outcome.fit(X_train, y_result_train)



# LOGISTIC REGRESSION FOR THE NUMBER OF CORNERS

#creating targets for all classes
final_data['over_8.5'] = np.where(final_data['number_of_corners'] > 8.5, 1, 0)
final_data['over_9.5'] = np.where(final_data['number_of_corners'] > 9.5, 1, 0)
final_data['over_10.5'] = np.where(final_data['number_of_corners'] > 10.5, 1, 0)
final_data['over_11.5'] = np.where(final_data['number_of_corners'] > 11.5, 1, 0)
final_data['over_12.5'] = np.where(final_data['number_of_corners'] > 12.5, 1, 0)
final_data['over_13.5'] = np.where(final_data['number_of_corners'] > 13.5, 1, 0)

y_cor_85 = final_data['over_8.5']
y_cor_95 = final_data['over_9.5']
y_cor_105 = final_data['over_10.5']
y_cor_115 = final_data['over_11.5']
y_cor_125 = final_data['over_12.5']
y_cor_135 = final_data['over_13.5']

# train and test data for each model
X_train, X_test, y_cor_85_train, y_cor_85_test = train_test_split(X, y_cor_85, test_size=0.30)
X_train, X_test, y_cor_95_train, y_cor_95_test = train_test_split(X, y_cor_95, test_size=0.30)
X_train, X_test, y_cor_105_train, y_cor_105_test = train_test_split(X, y_cor_105, test_size=0.30)
X_train, X_test, y_cor_115_train, y_cor_115_test = train_test_split(X, y_cor_115, test_size=0.30)
X_train, X_test, y_cor_125_train, y_cor_125_test = train_test_split(X, y_cor_125, test_size=0.30)
X_train, X_test, y_cor_135_train, y_cor_135_test = train_test_split(X, y_cor_135, test_size=0.30)

# instantiate and fit each model
model_lr_cor_85 = LogisticRegression(max_iter=500)
model_lr_cor_85.fit(X_train, y_cor_85_train)

model_lr_cor_95 = LogisticRegression(max_iter=500)
model_lr_cor_95.fit(X_train, y_cor_95_train)

model_lr_cor_105 = LogisticRegression(max_iter=500)
model_lr_cor_105.fit(X_train, y_cor_105_train)

model_lr_cor_115 = LogisticRegression(max_iter=500)
model_lr_cor_115.fit(X_train, y_cor_115_train)

model_lr_cor_125 = LogisticRegression(max_iter=500)
model_lr_cor_125.fit(X_train, y_cor_125_train)

model_lr_cor_135 = LogisticRegression(max_iter=500)
model_lr_cor_135.fit(X_train, y_cor_135_train)


# LOGISTIC REGRESSION FOR THE NUMBER OF GOALS

# create targets for each class
final_data['over_0.5'] = np.where(final_data['number_of_goals'] > 0.5, 1, 0)
final_data['over_1.5'] = np.where(final_data['number_of_goals'] > 1.5, 1, 0)
final_data['over_2.5'] = np.where(final_data['number_of_goals'] > 2.5, 1, 0)
final_data['over_3.5'] = np.where(final_data['number_of_goals'] > 3.5, 1, 0)
final_data['over_4.5'] = np.where(final_data['number_of_goals'] > 4.5, 1, 0)

y_g_05 = final_data['over_0.5']
y_g_15 = final_data['over_1.5']
y_g_25 = final_data['over_2.5']
y_g_35 = final_data['over_3.5']
y_g_45 = final_data['over_4.5']


# train and test data for each model
X_train, X_test, y_g_05_train, y_g_05_test = train_test_split(X, y_g_05, test_size=0.30)
X_train, X_test, y_g_15_train, y_g_15_test = train_test_split(X, y_g_15, test_size=0.30)
X_train, X_test, y_g_25_train, y_g_25_test = train_test_split(X, y_g_25, test_size=0.30)
X_train, X_test, y_g_35_train, y_g_35_test = train_test_split(X, y_g_35, test_size=0.30)
X_train, X_test, y_g_45_train, y_g_45_test = train_test_split(X, y_g_45, test_size=0.30)

# instantiate and fit each model
model_lr_g_05 = LogisticRegression(max_iter=500)
model_lr_g_05.fit(X_train, y_g_05_train)

model_lr_g_15 = LogisticRegression(max_iter=500)
model_lr_g_15.fit(X_train, y_g_15_train)

model_lr_g_25 = LogisticRegression(max_iter=500)
model_lr_g_25.fit(X_train, y_g_25_train)

model_lr_g_35 = LogisticRegression(max_iter=500)
model_lr_g_35.fit(X_train, y_g_35_train)

model_lr_g_45 = LogisticRegression(max_iter=500)
model_lr_g_45.fit(X_train, y_g_45_train)


final_data.reset_index(inplace=True)


def home_average_season_pred(data, home_team, variavel, data_pred):

    media_home = []

    data_filter = data.loc[data["HT"]==home_team]
    mean = data_filter[variavel].mean()
    #print(mean)
    media_home.append(mean)
    data_pred["home_avrg_"+variavel] = media_home
    return data_pred


def away_average_season_pred(data, away_team, variavel, data_pred):

    media_away = []

    data_filter = data.loc[data["AT"]==away_team]
    mean = data_filter[variavel].mean()
    media_away.append(mean)

    data_pred["away_avrg_"+variavel] = media_away
    return data_pred

def home_average_last_3_pred(data, home_team, away_team, variavel, data_pred):

    i = data["index"].shape[0]
    media_home = []
    oi = data.loc[data["index"]<=i]
    oi = oi.loc[oi["HT"]==home_team]
    oi= oi.reset_index(drop=True)
    oi = oi[-3:]
    oi = oi[variavel].mean()
    media_home.append(oi)

    data_pred["last_3_home_avrg_"+variavel] = pd.DataFrame(media_home)
    return data_pred

def away_average_last_3_pred(data, home_team, away_team, variavel, data_pred):

    i = data["index"].shape[0]
    media_away = []

    oi = data.loc[data["index"]<=i]
    oi = oi.loc[oi["AT"]==away_team]
    oi= oi.reset_index(drop=True)
    oi = oi[-3:]
    oi = oi[variavel].mean()
    media_away.append(oi)

    data_pred["last_3_away_avrg_"+variavel] = pd.DataFrame(media_away)
    return data_pred

def sequence_5_pred(data, home_team, away_team, data_pred):
    '''
    Description: Picks the last 5 games and calculates how many points the team scored, victory = 3
    loss = 0, draw = 1.

    Input:
        - None
    Output:
        - Sequence of the last 5 games points
    '''
    i = data["index"].shape[0]

    sequences_home = []
    oi = data.loc[data["index"]<=i]
    oi = oi.loc[oi["HT"]==home_team]
    oi= oi.reset_index(drop=True)
    oi = oi[-5:]
    oi = oi['points_result_home'].rolling(5).sum()
    sequences_home.append(oi.values[-1:])

    data_pred["home_pnts_lst_5"] = sequences_home[0]


    sequences_away = []
    oi = data.loc[data["index"]<=i]
    oi = oi.loc[oi["AT"]==away_team]
    oi= oi.reset_index(drop=True)
    oi = oi[-5:]
    oi = oi['points_result_away'].rolling(5).sum()
    sequences_away.append(oi.values[-1:])

    data_pred["away_pnts_lst_5"] = sequences_away[0]
    return data_pred

def sequence_3_pred(data, home_team, away_team, data_pred):
    '''
    Description: Picks the last 3 games and calculates how many points the team scored, victory = 3
    loss = 0, draw = 1.

    Input:
        - None
    Output:
        - Sequence of the last 3 games points
    '''
    i = data["index"].shape[0]

    sequences_home = []
    oi = data.loc[data["index"]<=i]
    oi = oi.loc[oi["HT"]==home_team]
    oi= oi.reset_index(drop=True)
    oi = oi[-3:]
    oi = oi['points_result_home'].rolling(3).sum()
    sequences_home.append(oi.values[-1:])

    data_pred["home_pnts_lst_3"] = sequences_home[0]

    sequences_away = []
    #for i, j in zip(data["Index"],data["AT"]):
    oi = data.loc[data["index"]<=i]
    oi = oi.loc[oi["AT"]==away_team]
    oi= oi.reset_index(drop=True)
    oi = oi[-3:]
    oi = oi['points_result_away'].rolling(3, min_periods=1).sum()
    sequences_away.append(oi.values[-1:])

    data_pred["away_pnts_lst_3"] = sequences_away[0]
    return data_pred

def sequence_1_pred(data, home_team, away_team, data_pred):
    '''
    Description: Picks the last 3 games and calculates how many points the team scored, victory = 3
    loss = 0, draw = 1.

    Input:
        - None
    Output:
        - Sequence of the last 3 games points
    '''
    i = data['index'].shape[0]

    sequences_home = []

    oi = data.loc[data["index"]<=i]
    oi = oi.loc[oi["HT"]==home_team]
    oi= oi.reset_index(drop=True)
    oi = oi[-1:]
    oi = oi['points_result_home'].rolling(1).sum()
    sequences_home.append(oi.values[-1:])

    data_pred["home_pnts_lst_game"] = sequences_home[0]


    sequences_away = []

    oi = data.loc[data["index"]<=i]
    oi = oi.loc[oi["AT"]==away_team]
    oi= oi.reset_index(drop=True)
    oi = oi[-3:]
    oi = oi['points_result_away'].rolling(1).sum()
    sequences_away.append(oi.values[-1:])

    data_pred["away_pnts_lst_game"] = sequences_away[0]
    return data_pred


def teams_encoded(data, home_team, away_team):

    data[home_encoded] = 0
    data[away_encoded] = 0

    a = f'{home_team}' + '_HT'
    b = f'{away_team}' + '_AT'

    data[[a]] = 1
    data[[b]] = 1

    return data

#home_team = home_team
away_team = away_team

home_encoded = final_data.loc[:, 'América (MG)_HT':'Vitória_HT'].columns
away_encoded = final_data.loc[:, 'América (MG)_AT':].columns[:-14]


home_features = ['Home_Poss', 'Home_PA', 'Home_ShoT', 'Home_Saves', 'HomeFouls', 'HomeCorners', 'HomeCrosses', 'HomeTouches',
                 'HomeTackles', 'HomeInterceptions', 'HomeAerials', 'HomeClearances', 'HomeOffsides', 'HomeGoalKicks',
                 'HomeThrowIns', 'HomeLongBalls']
away_features = ['Away_Poss', 'Away_PA', 'Away_ShoT', 'Away_Saves', 'AwayFouls', 'AwayCorners', 'AwayCrosses', 'AwayTouches',
                 'AwayTackles', 'AwayInterceptions', 'AwayAerials', 'AwayClearances', 'AwayOffsides', 'AwayGoalKicks',
                 'AwayThrowIns', 'AwayLongBalls']


def run_all(home_team, away_team):

    data_pred = pd.DataFrame()

    # Calculate the statistics of previous matches
    for i in home_features:
        data_pred = home_average_season_pred(data, home_team, i, data_pred)
        data_pred = home_average_last_3_pred(data, home_team, away_team, i, data_pred)

    for i in away_features:
        data_pred = away_average_season_pred(data, away_team, i, data_pred)
        data_pred = away_average_last_3_pred(data, home_team, away_team, i, data_pred)

    data_pred = sequence_5_pred(data, home_team, away_team, data_pred)
    data_pred = sequence_3_pred(data, home_team, away_team, data_pred)
    data_pred = sequence_1_pred(data, home_team, away_team, data_pred)

    #return data_pred
    data_pred = data_pred
    data_pred_columns = data_pred.columns

    # Concatenate the statistics with the dummified columns
    norm_predic = norm.transform(data_pred)
    norm_predic = pd.DataFrame(norm_predic, columns=data_pred_columns)
    #return norm_predic

    data_pred_concat = pd.concat([norm_predic, final_data[home_encoded], final_data[away_encoded]], axis=1)
    data_pred_concat.dropna(inplace=True)

    # Select the correct home_team and away_team
    data_pred_concat = teams_encoded(data_pred_concat, home_team, away_team)

    #Adjust this part to get info from final_final
    cols = final_data.columns[1:-14]

    return data_pred_concat[cols]

data_pred = run_all(home_team, away_team)
#data_pred






left_column, middle_column, right_column = st.beta_columns(3)






with left_column:
    st.subheader("Probability of Winning")
    st.write (f"{home_team} : ", '{:.1%}'.format(model_lr_outcome.predict_proba(data_pred)[0][1]))
    st.write(f"{away_team} : ", '{:.1%}'.format(model_lr_outcome.predict_proba(data_pred)[0][0]))
    st.write("Draw: ", '{:.1%}'.format(model_lr_outcome.predict_proba(data_pred)[0][2]))


with middle_column:
    # PROBABILITY OF PARTICULAR NUMBERS OF CORNERS
    st.subheader("Number of Corners")
    st.write("Over 8.5  -----", '{:.1%}'.format(model_lr_cor_85.predict_proba(data_pred)[0][1]))
    st.write("Over 9.5  -----", '{:.1%}'.format(model_lr_cor_95.predict_proba(data_pred)[0][1]))
    st.write("Over 10.5 -----", '{:.1%}'.format(model_lr_cor_105.predict_proba(data_pred)[0][1]))
    st.write("Over 11.5 -----", '{:.1%}'.format(model_lr_cor_115.predict_proba(data_pred)[0][1]))
    st.write("Over 12.5 -----", '{:.1%}'.format(model_lr_cor_125.predict_proba(data_pred)[0][1]))
    st.write("Over 13.5 -----", '{:.1%}'.format(model_lr_cor_135.predict_proba(data_pred)[0][1]))

with right_column:
    # PROBABILITY OF PARTICULAR NUMBERS OF GOALS
    st.subheader("Number of Goals")
    st.write("Over 0.5 -----", '{:.1%}'.format(model_lr_g_05.predict_proba(data_pred)[0][1]))
    st.write("Over 1.5 -----", '{:.1%}'.format(model_lr_g_15.predict_proba(data_pred)[0][1]))
    st.write("Over 2.5 -----", '{:.1%}'.format(model_lr_g_25.predict_proba(data_pred)[0][1]))
    st.write("Over 3.5 -----", '{:.1%}'.format(model_lr_g_35.predict_proba(data_pred)[0][1]))
    st.write("Over 4.5 -----", '{:.1%}'.format(model_lr_g_45.predict_proba(data_pred)[0][1]))



expander = st.beta_expander("FAQ")
expander.write("This page will advise you on real brazilian football predictions on future matches.")
expander.write("Last update of dataframe 20/11/20")
