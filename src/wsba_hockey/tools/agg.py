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

def calc_toi(pbp):
    dfs = []
    for team in ['home','away']:
        max_secs = int(pbp['seconds_elapsed'].max())+1

        on_ice_col = ['away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id',"away_goalie_id",
                'home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id',"home_goalie_id"]

        pbp = pbp.loc[pbp['event_type']=='change']
        pbp[on_ice_col] = pbp[on_ice_col].fillna("REMOVE")

        timeline = pd.DataFrame()
        timeline['seconds_elapsed'] = range(max_secs)

        info_col = ['season','season_type','game_id','game_date',
            'away_team_abbr','home_team_abbr','period','period_type',
            "seconds_elapsed","away_skaters","home_skaters","strength_state",]

        timeline = pd.merge(timeline,pbp[info_col+on_ice_col].drop_duplicates(subset=['seconds_elapsed'],keep='last'),how="left",on=['seconds_elapsed'])

        timeline[info_col+on_ice_col] = timeline[info_col+on_ice_col].ffill()
        timeline = timeline.replace({
            "REMOVE":np.nan
        })

        away_on = ['away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id']
        home_on = ['home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id']

        timeline['away_skaters'] = timeline[away_on].replace(r'^\s*$', np.nan, regex=True).notna().sum(axis=1)
        timeline['home_skaters'] = timeline[home_on].replace(r'^\s*$', np.nan, regex=True).notna().sum(axis=1)

        timeline['strength_state'] = timeline['away_skaters'].astype(str) + "v" + timeline['home_skaters'].astype(str)

        shifts = timeline[info_col+on_ice_col]
        shifts['on_ice'] = (shifts[f'{team}_on_1_id'].astype(str)
                            .str.cat(shifts[f'{team}_on_2_id'],sep=";",na_rep="")
                            .str.cat(shifts[f'{team}_on_3_id'],sep=";",na_rep="")
                            .str.cat(shifts[f'{team}_on_3_id'],sep=";",na_rep="")
                            .str.cat(shifts[f'{team}_on_5_id'],sep=";",na_rep="")
                            .str.cat(shifts[f'{team}_on_6_id'],sep=";",na_rep=""))
        
        # Expand on_ice into individual player records
        full_shifts = shifts.assign(
            on_ice=shifts['on_ice'].str.split(';'),
        ).explode('on_ice')
        
        full_shifts = full_shifts.loc[full_shifts.index > 0]

        # Calculate Time on Ice (TOI)
        toi_df = full_shifts.groupby(['on_ice',f"{team}_team_abbr"]).agg(
            GP=('game_id', lambda x: x.nunique()),
            TOI=('seconds_elapsed', 'count')
        ).reset_index()
        
        toi_df['TOI'] = toi_df['TOI'] / 60
        toi_df['on_ice'] = toi_df['on_ice'].replace(r'^\s*$', np.nan, regex=True)

        dfs.append(toi_df.loc[toi_df['on_ice'].notna()].rename(columns={"on_ice": "ID",f"{team}_team_abbr":"Team"}))

    return pd.concat(dfs).groupby(['ID','Team'], as_index=False).agg(
        GP = ('GP','sum'),
        TOI = ("TOI",'sum')
    )