import pandas as pd
import numpy as np
from tools.xg_model import *

def calc_indv(pbp):
    indv = (
        pbp.loc[pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot"])].groupby(['event_player_1_id','event_team_abbr','season']).agg(
        G=('event_type', lambda x: (x == "goal").sum()),
        iFF=('event_type', lambda x: (x != "blocked-shot").sum()),
        ixG=('xG', 'sum'),
        Rush=('rush_mod',lambda x: (x > 0).sum())
    ).reset_index().rename(columns={'event_player_1_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season'})
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

    indv = pd.merge(indv,a1,how='outer',on=['ID','Team','Season'])
    indv = pd.merge(indv,a2,how='outer',on=['ID','Team','Season'])

    indv[['G','A1','A2']] = indv[['G','A1','A2']].fillna(0)

    indv['P1'] = indv['G']+indv['A1']
    indv['P'] = indv['P1']+indv['A2']
    indv['ixG/iFF'] = indv['ixG']/indv['iFF']
    indv['G/ixG'] = indv['G']/indv['ixG']
    indv['iFsh%'] = indv['G']/indv['iFF']

    return indv

def calc_onice(pbp):
    # Filter for specific event types
    fenwick_events = ['missed-shot','shot-on-goal','goal']

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

def calc_toi(pbp):
    # Convert player on-ice columns to vectors
    pbp['home_on_ice'] = pbp['home_on_1_id'].astype(str) + ";" + pbp['home_on_2_id'].astype(str) + ";" + pbp['home_on_3_id'].astype(str) + ";" + pbp['home_on_4_id'].astype(str) + ";" + pbp['home_on_5_id'].astype(str) + ";" + pbp['home_on_6_id'].astype(str)
    pbp['away_on_ice'] = pbp['away_on_1_id'].astype(str) + ";" + pbp['away_on_2_id'].astype(str) + ";" + pbp['away_on_3_id'].astype(str) + ";" + pbp['away_on_4_id'].astype(str) + ";" + pbp['away_on_5_id'].astype(str) + ";" + pbp['away_on_6_id'].astype(str)
    
    # Remove NA players
    pbp['home_on_ice'] = pbp['home_on_ice'].str.replace(';nan', '', regex=True)
    pbp['away_on_ice'] = pbp['away_on_ice'].str.replace(';nan', '', regex=True)
