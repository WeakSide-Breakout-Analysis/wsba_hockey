import pandas as pd
import numpy as np
from tools.xg_model import *

def calc_indv(pbp):
    indv = (
        pbp.loc[pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot"])].groupby(['event_player_1_id','event_team_abbr']).agg(
        G=('event_type', lambda x: (x == "goal").sum()),
        iFF=('event_type', lambda x: (x != "blocked-shot").sum()),
        ixG=('xG', 'sum'),
    ).reset_index().rename(columns={'event_player_1_id': 'ID', 'event_team_abbr': 'Team'})
    )

    a1 = (
        pbp.loc[pbp['event_type'].isin(["goal"])].groupby(['event_player_2_id', 'event_team_abbr']).agg(
        A1=('event_type','count')
    ).reset_index().rename(columns={'event_player_2_id': 'ID', 'event_team_abbr': 'Team'})
    )

    a2 = (
        pbp.loc[pbp['event_type'].isin(["goal"])].groupby(['event_player_3_name', 'event_player_3_id', 'event_team_abbr']).agg(
        A2=('event_type','count')
    ).reset_index().rename(columns={'event_player_3_id': 'ID', 'event_team_abbr': 'Team'})
    )

    indv = pd.merge(indv,a1,how='outer',on=['ID','Team'])
    indv = pd.merge(indv,a2,how='outer',on=['ID','Team'])

    indv['ixG/iFF'] = indv['ixG']/indv['iFF']
    indv['G/ixG'] = indv['G']/indv['ixG']
    indv['iFsh%'] = indv['G']/indv['iFF']

    return indv

def calc_onice(pbp):
    # Filter for specific event types
    pbp_new = pbp.loc[pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot"])]
    
    # Convert player on-ice columns to vectors
    pbp_new['home_on_ice'] = pbp_new['home_on_1_id'].astype(str) + ";" + pbp_new['home_on_2_id'].astype(str) + ";" + pbp_new['home_on_3_id'].astype(str) + ";" + pbp_new['home_on_4_id'].astype(str) + ";" + pbp_new['home_on_5_id'].astype(str) + ";" + pbp_new['home_on_6_id'].astype(str)
    pbp_new['away_on_ice'] = pbp_new['away_on_1_id'].astype(str) + ";" + pbp_new['away_on_2_id'].astype(str) + ";" + pbp_new['away_on_3_id'].astype(str) + ";" + pbp_new['away_on_4_id'].astype(str) + ";" + pbp_new['away_on_5_id'].astype(str) + ";" + pbp_new['away_on_6_id'].astype(str)
    
    # Remove NA players
    pbp_new['home_on_ice'] = pbp_new['home_on_ice'].str.replace(';nan', '', regex=True)
    pbp_new['away_on_ice'] = pbp_new['away_on_ice'].str.replace(';nan', '', regex=True)
    
    def process_team_stats(df, on_ice_col, team_col, opp_col):
        df = df[['game_id', 'event_num', team_col, opp_col, 'event_type', 'event_team_abbr', on_ice_col,'xG']].copy()
        df[on_ice_col] = df[on_ice_col].str.split(';')
        df = df.explode(on_ice_col)
        df = df.rename(columns={on_ice_col: 'ID'})
        df['xGF'] = np.where(df['event_team_abbr'] == df[team_col], df['xG'], 0)
        df['xGA'] = np.where(df['event_team_abbr'] == df[opp_col], df['xG'], 0)
        df['GF'] = np.where((df['event_type'] == "goal") & (df['event_team_abbr'] == df[team_col]), 1, 0)
        df['GA'] = np.where((df['event_type'] == "goal") & (df['event_team_abbr'] == df[opp_col]), 1, 0)
        df['FF'] = np.where((df['event_type'] != "blocked-shot") & (df['event_team_abbr'] == df[team_col]), 1, 0)
        df['FA'] = np.where((df['event_type'] != "blocked-shot") & (df['event_team_abbr'] == df[opp_col]), 1, 0)

        stats = df.groupby(['ID',team_col]).agg(
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum')
        ).reset_index()
        
        return stats.rename(columns={team_col:"Team"})
    
    home_stats = process_team_stats(pbp_new, 'home_on_ice', 'home_team_abbr', 'away_team_abbr')
    away_stats = process_team_stats(pbp_new, 'away_on_ice', 'away_team_abbr', 'home_team_abbr')

    onice_stats = pd.concat([home_stats,away_stats]).groupby(['ID','Team']).agg(
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

def create_timeline(pbp):
    #Filter non-change events
    data = pbp.loc[pbp['event_type']=='change']

    shifts = data.drop_duplicates(subset=['period','seconds_elapsed'],keep='last')
    secs = pd.DataFrame()
    secs['seconds_elapsed'] = range(shifts['seconds_elapsed'].max())

    on_ice_col = ['game_id','away_team_abbr','home_team_abbr','strength_state','away_on_1','away_on_2','away_on_3','away_on_4','away_on_5','away_on_6',
                    'away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id',
                    'home_on_1','home_on_2','home_on_3','home_on_4','home_on_5','home_on_6',
                    'home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id',
                    'away_goalie','home_goalie','away_goalie_id','home_goalie_id']

    shifts = pd.merge(shifts,secs,how='right')
    for col in on_ice_col:
        shifts[col] = shifts[col].ffill()

    shifts = shifts[['seconds_elapsed']+on_ice_col]
    shifts['away_on_ice'] = (shifts['away_on_1_id'].astype(str)
                        .str.cat(shifts['away_on_2_id'].astype(str),sep=";",na_rep="")
                        .str.cat(shifts['away_on_3_id'].astype(str),sep=";",na_rep="")
                        .str.cat(shifts['away_on_3_id'].astype(str),sep=";",na_rep="")
                        .str.cat(shifts['away_on_5_id'].astype(str),sep=";",na_rep="")
                        .str.cat(shifts['away_on_6_id'].astype(str),sep=";",na_rep="")).replace(";nan","",regex=True)
    
    shifts['home_on_ice'] = (shifts['home_on_1_id'].astype(str)
                        .str.cat(shifts['home_on_2_id'].astype(str),sep=";",na_rep="")
                        .str.cat(shifts['home_on_3_id'].astype(str),sep=";",na_rep="")
                        .str.cat(shifts['home_on_3_id'].astype(str),sep=";",na_rep="")
                        .str.cat(shifts['home_on_5_id'].astype(str),sep=";",na_rep="")
                        .str.cat(shifts['home_on_6_id'].astype(str),sep=";",na_rep="")).replace(";nan","",regex=True)    

    return shifts

def calc_toi(pbp):
    shifts = create_timeline(pbp)

    #Split shifts and Expand on_ice into individual player records
    away_shifts = shifts.assign(
        on_ice=shifts['away_on_ice'].str.split(';'),
    ).explode('on_ice')
    
    away_shifts = away_shifts.loc[away_shifts.index > 0]
    
    home_shifts = shifts.assign(
        on_ice=shifts['home_on_ice'].str.split(';'),
    ).explode('on_ice')
    
    home_shifts = home_shifts.loc[home_shifts.index > 0]

    #Calculate Time on Ice (TOI) for Away Team
    away_toi_df = away_shifts.groupby(['on_ice',"away_team_abbr"]).agg(
        GP=('game_id', lambda x: x.nunique()),
        TOI=('seconds_elapsed', 'count')
    ).reset_index()

    away_toi_df = away_toi_df.loc[away_toi_df['on_ice'].notna()].rename(columns={"on_ice": "ID","away_team_abbr":"Team"})
    
    #Calculate Time on Ice (TOI) for Home Team
    home_toi_df = home_shifts.groupby(['on_ice',"home_team_abbr"]).agg(
        GP=('game_id', lambda x: x.nunique()),
        TOI=('seconds_elapsed', 'count')
    ).reset_index()
    
    home_toi_df = home_toi_df.loc[home_toi_df['on_ice'].notna()].rename(columns={"on_ice": "ID","home_team_abbr":"Team"})

    #Combine Data
    toi_df = pd.concat([away_toi_df,home_toi_df])
    
    toi_df['TOI'] = toi_df['TOI'] / 60
    toi_df['ID'] = toi_df['ID'].replace(r'^\s*$', np.nan, regex=True)

    return toi_df