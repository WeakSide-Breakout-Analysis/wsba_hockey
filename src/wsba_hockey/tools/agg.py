import pandas as pd
import numpy as np
from tools.xg_model import *

def calc_indv(pbp):
    indv = (
        pbp.loc[pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot"])].groupby(['event_player_1_name', 'event_team_abbr']).agg(
        G=('event_type', lambda x: (x == "goal").sum()),
        iFF=('event_type', lambda x: (x != "blocked-shot").sum()),
        ixG=('xG', 'sum'),
    ).reset_index().rename(columns={'event_player_1_name': 'Player', 'event_team_abbr': 'Team'})
    )

    a1 = (
        pbp.loc[pbp['event_type'].isin(["goal"])].groupby(['event_player_2_name', 'event_team_abbr']).agg(
        A1=('event_type','count')
    ).reset_index().rename(columns={'event_player_2_name': 'Player', 'event_team_abbr': 'Team'})
    )

    a2 = (
        pbp.loc[pbp['event_type'].isin(["goal"])].groupby(['event_player_3_name', 'event_team_abbr']).agg(
        A2=('event_type','count')
    ).reset_index().rename(columns={'event_player_3_name': 'Player', 'event_team_abbr': 'Team'})
    )

    indv = pd.merge(indv,a1,how='outer',on=['Player','Team'])
    indv = pd.merge(indv,a2,how='outer',on=['Player','Team'])

    indv['ixG/iFF'] = indv['ixG']/indv['iFF']
    indv['G/ixG'] = indv['G']/indv['ixG']
    indv['iFsh%'] = indv['G']/indv['iFF']

    return indv

def calc_onice(pbp):
    # Filter for specific event types
    pbp_new = pbp.loc[pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot"])]
    
    # Convert player on-ice columns to vectors
    pbp_new['home_on_ice'] = pbp_new['home_on_1'].astype(str) + ";" + pbp_new['home_on_2'].astype(str) + ";" + pbp_new['home_on_3'].astype(str) + ";" + pbp_new['home_on_4'].astype(str) + ";" + pbp_new['home_on_5'].astype(str) + ";" + pbp_new['home_on_6'].astype(str)
    pbp_new['away_on_ice'] = pbp_new['away_on_1'].astype(str) + ";" + pbp_new['away_on_2'].astype(str) + ";" + pbp_new['away_on_3'].astype(str) + ";" + pbp_new['away_on_4'].astype(str) + ";" + pbp_new['away_on_5'].astype(str) + ";" + pbp_new['away_on_6'].astype(str)
    
    # Remove NA players
    pbp_new['home_on_ice'] = pbp_new['home_on_ice'].str.replace(';nan', '', regex=True)
    pbp_new['away_on_ice'] = pbp_new['away_on_ice'].str.replace(';nan', '', regex=True)
    
    def process_team_stats(df, on_ice_col, team_col, opp_col):
        df = df[['game_id', 'event_num', team_col, opp_col, 'event_type', 'event_team_abbr', on_ice_col,'xG']].copy()
        df[on_ice_col] = df[on_ice_col].str.split(';')
        df = df.explode(on_ice_col)
        df = df.rename(columns={on_ice_col: 'Player'})
        df['xGF'] = np.where(df['event_team_abbr'] == df[team_col], df['xG'], 0)
        df['xGA'] = np.where(df['event_team_abbr'] == df[opp_col], df['xG'], 0)
        df['GF'] = np.where((df['event_type'] == "goal") & (df['event_team_abbr'] == df[team_col]), 1, 0)
        df['GA'] = np.where((df['event_type'] == "goal") & (df['event_team_abbr'] == df[opp_col]), 1, 0)
        df['FF'] = np.where((df['event_type'] != "blocked-shot") & (df['event_team_abbr'] == df[team_col]), 1, 0)
        df['FA'] = np.where((df['event_type'] != "blocked-shot") & (df['event_team_abbr'] == df[opp_col]), 1, 0)

        stats = df.groupby(['Player',team_col]).agg(
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

    onice_stats =pd.concat([home_stats,away_stats]).groupby(['Player','Team']).agg(
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
    
        on_ice_col = ['away_on_1','away_on_2','away_on_3','away_on_4','away_on_5','away_on_6',"away_goalie",
                'home_on_1','home_on_2','home_on_3','home_on_4','home_on_5','home_on_6',"home_goalie"]

        pbp = pbp.loc[pbp['event_type']=='change']
        pbp[on_ice_col] = pbp[on_ice_col].fillna("REMOVE")

        timeline = pd.DataFrame()
        timeline['seconds_elapsed'] = range(max_secs)

        info_col = ['season','season_type','game_id','game_date',
            'away_team_abbr','home_team_abbr','period','period_type',
            "seconds_elapsed","away_skaters","home_skaters","strength_state",]

        timeline = pd.merge(timeline,pbp[info_col+on_ice_col].drop_duplicates(subset=['seconds_elapsed'],keep='last'),how="outer",on=['seconds_elapsed'])
        timeline.to_csv("sample_timeline_pre.csv",index=False)

        timeline[info_col+on_ice_col] = timeline[info_col+on_ice_col].ffill()
        timeline = timeline.replace({
            "REMOVE":np.nan
        })

        away_on = ['away_on_1','away_on_2','away_on_3','away_on_4','away_on_5','away_on_6']
        home_on = ['home_on_1','home_on_2','home_on_3','home_on_4','home_on_5','home_on_6']

        timeline['away_skaters'] = timeline[away_on].replace(r'^\s*$', np.nan, regex=True).notna().sum(axis=1)
        timeline['home_skaters'] = timeline[home_on].replace(r'^\s*$', np.nan, regex=True).notna().sum(axis=1)

        timeline['strength_state'] = timeline['away_skaters'].astype(str) + "v" + timeline['home_skaters'].astype(str)

        shifts = timeline[info_col+on_ice_col]
        shifts['on_ice'] = (shifts[f'{team}_on_1'].astype(str)
                            .str.cat(shifts[f'{team}_on_2'],sep=";",na_rep="")
                            .str.cat(shifts[f'{team}_on_3'],sep=";",na_rep="")
                            .str.cat(shifts[f'{team}_on_4'],sep=";",na_rep="")
                            .str.cat(shifts[f'{team}_on_5'],sep=";",na_rep="")
                            .str.cat(shifts[f'{team}_on_6'],sep=";",na_rep=""))
        
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

        dfs.append(toi_df.loc[toi_df['on_ice'].notna()].rename(columns={"on_ice": "Player",f"{team}_team_abbr":"Team"}))

    return pd.concat(dfs).groupby(['Player','Team'], as_index=False).agg(
        GP = ('GP','sum'),
        TOI = ("TOI",'sum')
    )

def nhl_calculate_stats(pbp,season,season_types,game_strength,xg="moneypuck"):
    print("Calculating stats for play-by-play and shifts data provided in the frame: " + season + "...")
    
    #Check if xG column exists and apply model if it does not
    try:
        pbp['xG']
    except KeyError:
        if xg == 'wsba':
            pbp = wsba_xG(pbp)
        else:
            pbp = moneypuck_xG(pbp)

    pbp = pbp.loc[(pbp['season_type'].isin(season_types)) & (pbp['period'] < 5)]
    
    # Filter by game strength if not "all"
    if game_strength != "all":
        pbp = pbp.loc[pbp['strength_state'].isin(game_strength)]

    indv_stats = calc_indv(pbp)
    onice_stats = calc_onice(pbp)
    #info = calc_toi(pbp)

    complete = pd.merge(indv_stats,onice_stats,how="outer",on=['Player','Team'])
    #complete = pd.merge(complete,info,how="outer",on=['Player','Team'])
    complete['Season'] = season
    complete['GC%'] = complete['G']/complete['GF']
    complete['AC%'] = (complete['A1']+complete['A2'])/complete['GF']
    complete['GI%'] = (complete['G']+complete['A1']+complete['A2'])/complete['GF']
    complete['FC%'] = complete['iFF']/complete['FF']
    complete['xGC%'] = complete['ixG']/complete['xGF']
    #complete['RC%'] = complete['Rush']/complete['iFF']
    #complete['AVG_Rush_POW'] = complete['Rush_POW']/complete['Rush']

    return complete[[
       'Player',"Season","Team",
        "G","A1","A2","iFF","ixG",'ixG/iFF',"G/ixG","iFsh%",
        "GF","FF","xGF","xGF/FF","GF/xGF","FshF%",
        "GA","FA","xGA","xGA/FA","GA/xGA","FshA%",
        "GC%","AC%","GI%","FC%","xGC%"
    ]].fillna(0)