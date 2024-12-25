import pandas as pd
import numpy as np
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns 
import matplotlib.pyplot as plt 
import streamlit as st
import lxml

st.header('Premier League Match Prediction for 2024/25')
home = st.sidebar.selectbox('Select home team', ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
                                          'Ipswich Town', 'leicester city', 'Liverpool', 'Manchester Utd', 'Manchester City', 'Newcastle Utd', "Nott'ham Forest", 'Southampton',
                                           'Tottenham', 'West Ham', 'Wolves'])

away = st.sidebar.selectbox('Select away team', ['Aston Villa', 'Arsenal', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
                                          'Ipswich Town', 'leicester city', 'Liverpool', 'Manchester Utd', 'Manchester City', 'Newcastle Utd', "Nott'ham Forest", 'Southampton',
                                           'Tottenham', 'West Ham', 'Wolves'])
button = st.sidebar.button('Predict')
#Dataset is webscraped from fbref and football data websites
url='https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures'
ligue_list = []

ligue_list.append(pd.read_html(url,index_col=False,flavor='lxml')[0])
ligue_list = pd.concat(ligue_list, axis=0, ignore_index=True)
pl_df = ligue_list[ligue_list['Wk'].notna()] # drop na

pl_df=pl_df.rename(columns={'Home':'HomeTeam','Away':'AwayTeam'})
pl_df=pl_df.rename(columns={'xG':'XGHome','xG.1':'XGAway'})
pl_df['HomeGoals'] = pl_df['Score'].str[0]
pl_df['AwayGoals'] = pl_df['Score'].str[2]
pl_df=pl_df[['Date', 'Time','HomeTeam', 'AwayTeam','XGHome', 'Score', 'XGAway','HomeGoals','AwayGoals']]

pl_df=pl_df[['Date', 'Time','HomeTeam', 'AwayTeam','XGHome', 'Score', 'XGAway','HomeGoals','AwayGoals']]


df = pd.concat([pl_df[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           pl_df[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])
df = df.dropna()
df['goals'] = df['goals'].astype('int')
df.head()
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df[['team','opponent','home']])


# building the poisson model
formula = 'goals ~ team + opponent + home'
model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()


# predicting the scores
home_goals = int(model.predict(pd.DataFrame(data={'team': home, 'opponent': away, 'home':1}, index=[1])))
away_goals = int(model.predict(pd.DataFrame(data={'team': away, 'opponent': home, 'home':0}, index=[1])))
total_goals = home_goals + away_goals

def predict_match(model, homeTeam, awayTeam, max_goals=10):
    '''Predict the odds of a home win, draw and away win returned in matrix form'''
    home_goals = model.predict(pd.DataFrame(data={'team': homeTeam, 'opponent':awayTeam, 'home': 1}, index=[1])).values[0]
    away_goals = model.predict(pd.DataFrame(data={'team': awayTeam, 'opponent': homeTeam, 'home':0}, index=[1])).values[0]
    pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals, away_goals]]
    return(np.outer(np.array(pred[0]), np.array(pred[1])))

# getting the odds
score_matrix = predict_match(model, home, away, max_goals=10)

# odds of a home win
home_win = np.sum(np.tril(score_matrix, -1)) * 100
home_win = round(home_win)
# odds of a draw
draw = np.sum(np.diag(score_matrix)) * 100
draw = round(draw)
# odds of an away win
away_win = np.sum(np.triu(score_matrix, 1)) * 100
away_win = round(away_win)

def get_scores():
    ''' Display results'''
    # select only one team
    if home == away:
        st.error('You can\'t predict the same team')
        return None
    st.write(f'Score Prediction betweeen {home} and {away}')
    st.write(f'{home} ------   {home_goals}:{away_goals} ------  {away}')
    st.write(f'Odds of a home win is {home_win}%')
    st.write(f'Odds of an away win is {away_win}%')
    st.write(f'Odds of a draw is {draw}%')
if button:
    get_scores()
