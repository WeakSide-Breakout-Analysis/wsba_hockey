import pandas as pd
import numpy as np
from .xg_model import *

## AGGREGATE FUNCTIONS ##

## GLOBAL VARIABLES ##
shot_types = ['wrist','deflected','tip-in','slap','backhand','snap','wrap-around','poke','bat','cradle','between-legs']
fenwick_events = ['missed-shot','shot-on-goal','goal']

def calc_indv(pbp):
    indv = (
        pbp.loc[pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot"])].groupby(['event_player_1_id','event_team_abbr','season']).agg(
        Gi=('event_type', lambda x: (x == "goal").sum()),
        Fi=('event_type', lambda x: (x != "blocked-shot").sum()),
        xGi=('xG', 'sum'),
        Rush=('rush_mod',lambda x: (x > 0).sum())
    ).reset_index().rename(columns={'event_player_1_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season'})
    )
    
    rush_xg = (
        pbp.loc[(pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot"]))&(pbp['rush_mod']>0)].groupby(['event_player_1_id','event_team_abbr','season']).agg(
        Rush_G=('event_type', lambda x:(x == 'goal').sum()),
        Rush_xG=('xG','sum')
    ).reset_index().rename(columns={'event_player_1_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season', 'Rush_G':'Rush G','Rush_xG':'Rush xG'})
    )

    a1 = (
        pbp.loc[pbp['event_type'].isin(["goal"])].groupby(['event_player_2_id', 'event_team_abbr','season']).agg(
        A1=('event_type','count')
    ).reset_index().rename(columns={'event_player_2_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season'})
    )

    a2 = (
        pbp.loc[pbp['event_type'].isin(["goal"])].groupby(['event_player_3_id', 'event_team_abbr', 'season']).agg(
        A2=('event_type','count')
    ).reset_index().rename(columns={'event_player_3_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season'})
    )
    indv = pd.merge(indv,rush_xg,how='outer',on=['ID','Team','Season'])
    indv = pd.merge(indv,a1,how='outer',on=['ID','Team','Season'])
    indv = pd.merge(indv,a2,how='outer',on=['ID','Team','Season'])

    #Shot Types
    for type in shot_types:
        shot = (
            pbp.loc[(pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot"])&(pbp['shot_type']==type))].groupby(['event_player_1_id', 'event_team_abbr', 'season']).agg(
            Gi=('event_type', lambda x: (x == "goal").sum()),
            Fi=('event_type', lambda x: (x != "blocked-shot").sum()),
            xGi=('xG', 'sum'),
        ).reset_index().rename(columns={'event_player_1_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season'})
        )

        shot = shot.rename(columns={
            'Gi':f'{type.capitalize()}Gi',
            'Fi':f'{type.capitalize()}Fi',
            'xGi':f'{type.capitalize()}xGi',
        })
        indv = pd.merge(indv,shot,how='outer',on=['ID','Team','Season'])

    indv[['Gi','A1','A2']] = indv[['Gi','A1','A2']].fillna(0)

    indv['P1'] = indv['Gi']+indv['A1']
    indv['P'] = indv['P1']+indv['A2']
    indv['xGi/Fi'] = indv['xGi']/indv['Fi']
    indv['Gi/xGi'] = indv['Gi']/indv['xGi']
    indv['Fshi%'] = indv['Gi']/indv['Fi']

    return indv

def calc_onice(pbp):
    # Convert player on-ice columns to vectors
    pbp['home_on_ice'] = pbp['home_on_1_id'].astype(str) + ";" + pbp['home_on_2_id'].astype(str) + ";" + pbp['home_on_3_id'].astype(str) + ";" + pbp['home_on_4_id'].astype(str) + ";" + pbp['home_on_5_id'].astype(str) + ";" + pbp['home_on_6_id'].astype(str)
    pbp['away_on_ice'] = pbp['away_on_1_id'].astype(str) + ";" + pbp['away_on_2_id'].astype(str) + ";" + pbp['away_on_3_id'].astype(str) + ";" + pbp['away_on_4_id'].astype(str) + ";" + pbp['away_on_5_id'].astype(str) + ";" + pbp['away_on_6_id'].astype(str)
    
    # Remove NA players
    pbp['home_on_ice'] = pbp['home_on_ice'].str.replace(';nan', '', regex=True)
    pbp['away_on_ice'] = pbp['away_on_ice'].str.replace(';nan', '', regex=True)
    
    def process_team_stats(df, on_ice_col, team_col, opp_col):
        df = df[['season','game_id', 'event_num', team_col, opp_col, 'event_type', 'event_team_abbr', on_ice_col,'event_length','xG']].copy()
        df[on_ice_col] = df[on_ice_col].str.split(';')
        df = df.explode(on_ice_col)
        df = df.rename(columns={on_ice_col: 'ID', 'season': 'Season'})
        df['xGF'] = np.where(df['event_team_abbr'] == df[team_col], df['xG'], 0)
        df['xGA'] = np.where(df['event_team_abbr'] == df[opp_col], df['xG'], 0)
        df['GF'] = np.where((df['event_type'] == "goal") & (df['event_team_abbr'] == df[team_col]), 1, 0)
        df['GA'] = np.where((df['event_type'] == "goal") & (df['event_team_abbr'] == df[opp_col]), 1, 0)
        df['FF'] = np.where((df['event_type'].isin(fenwick_events)) & (df['event_team_abbr'] == df[team_col]), 1, 0)
        df['FA'] = np.where((df['event_type'].isin(fenwick_events)) & (df['event_team_abbr'] == df[opp_col]), 1, 0)

        stats = df.groupby(['ID',team_col,'Season']).agg(
            GP=('game_id','nunique'),
            TOI=('event_length','sum'),
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum')
        ).reset_index()
        
        return stats.rename(columns={team_col:"Team"})
    
    home_stats = process_team_stats(pbp, 'home_on_ice', 'home_team_abbr', 'away_team_abbr')
    away_stats = process_team_stats(pbp, 'away_on_ice', 'away_team_abbr', 'home_team_abbr')

    onice_stats = pd.concat([home_stats,away_stats]).groupby(['ID','Team','Season']).agg(
            GP=('GP','sum'),
            TOI=('TOI','sum'),
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum')
    ).reset_index()

    onice_stats['xGF/FF'] = onice_stats['xGF']/onice_stats['FF']
    onice_stats['GF/xGF'] = onice_stats['GF']/onice_stats['xGF']
    onice_stats['FshF%'] = onice_stats['GF']/onice_stats['FF']
    onice_stats['xGA/FA'] = onice_stats['xGA']/onice_stats['FA']
    onice_stats['GA/xGA'] = onice_stats['GA']/onice_stats['xGA']
    onice_stats['FshA%'] = onice_stats['GA']/onice_stats['FA']

    return onice_stats

def calc_team(pbp):
    teams = []
    for team in [('away','home'),('home','away')]:
        pbp['xGF'] = np.where(pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'], pbp['xG'], 0)
        pbp['xGA'] = np.where(pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'], pbp['xG'], 0)
        pbp['GF'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['GA'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['FF'] = np.where((pbp['event_type'].isin(fenwick_events)) & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['FA'] = np.where((pbp['event_type'].isin(fenwick_events)) & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['RushF'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'])&(pbp['rush_mod']>0), 1, 0)
        pbp['RushA'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'])&(pbp['rush_mod']>0), 1, 0)
        pbp['RushFxG'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'])&(pbp['rush_mod']>0), pbp['xG'], 0)
        pbp['RushAxG'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'])&(pbp['rush_mod']>0), pbp['xG'], 0)
        pbp['RushFG'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'])&(pbp['rush_mod']>0), 1, 0)
        pbp['RushAG'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'])&(pbp['rush_mod']>0), 1, 0)

        stats = pbp.groupby([f'{team[0]}_team_abbr','season']).agg(
            GP=('game_id','nunique'),
            TOI=('event_length','sum'),
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum'),
            RushF=('RushF','sum'),
            RushA=('RushA','sum'),
            RushFxG=('RushFxG','sum'),
            RushAxG=('RushAxG','sum'),
            RushFG=('RushFG','sum'),
            RushAG=('RushAG','sum'),
        ).reset_index().rename(columns={f'{team[0]}_team_abbr':"Team",'season':"Season"})
        teams.append(stats)
    
    onice_stats = pd.concat(teams).groupby(['Team','Season']).agg(
            GP=('GP','sum'),
            TOI=('TOI','sum'),
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum'),
            RushF=('RushF','sum'),
            RushA=('RushA','sum'),
            RushFxG=('RushFxG','sum'),
            RushAxG=('RushAxG','sum'),
            RushFG=('RushFG','sum'),
            RushAG=('RushAG','sum'),
    ).reset_index()

    onice_stats['xGF/FF'] = onice_stats['xGF']/onice_stats['FF']
    onice_stats['GF/xGF'] = onice_stats['GF']/onice_stats['xGF']
    onice_stats['FshF%'] = onice_stats['GF']/onice_stats['FF']
    onice_stats['xGA/FA'] = onice_stats['xGA']/onice_stats['FA']
    onice_stats['GA/xGA'] = onice_stats['GA']/onice_stats['xGA']
    onice_stats['FshA%'] = onice_stats['GA']/onice_stats['FA']

    return onice_stats