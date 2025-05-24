import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import rink_plot

event_markers = {
    'faceoff':'X',
    'hit':'P',
    'blocked-shot':'v',
    'missed-shot':'o',
    'shot-on-goal':'D',
    'goal':'*',
    'giveaway':'1',
    'takeaway':'2',
    'penalty':''
    }  

def wsba_rink():
    return rink_plot.rink(setting='full', vertical=False)

def colors(df):
    away_abbr = list(df['away_team_abbr'])[0]
    home_abbr = list(df['home_team_abbr'])[0]
    season = list(df['season'])[0]
    team_colors={'away':'secondary','home':'primary'}
    team_data = pd.read_csv('teaminfo/nhl_teaminfo.csv')

    team_info ={
        away_abbr:'#000000' if list(team_data.loc[team_data['WSBA']==f'{away_abbr}{season}','Secondary Color'])[0]=='#FFFFFF' else list(team_data.loc[team_data['WSBA']==f'{away_abbr}{season}',f'{team_colors['away'].capitalize()} Color'])[0],
        home_abbr: list(team_data.loc[team_data['WSBA']==f'{home_abbr}{season}',f'{team_colors['home'].capitalize()} Color'])[0],
    }

    return team_info

def prep(df,events):
    df = df.loc[df['event_type'].isin(events)]
    df['xG'] = df['xG'].fillna(0)
    df['size'] = np.where(df['xG']<=0,40,df['xG']*400)
    
    df['marker'] = df['event_type'].replace(event_markers)

    df['Description'] = df['description']
    df['Team'] = df['event_team_abbr']
    df['x'] = df['x_adj']
    df['y'] = df['y_adj']
    df['Event Distance from Attacking Net'] = df['event_distance']
    df['Event Angle to Attacking Net'] = df['event_angle']
    df['xG'] = df['xG']*100
    return df