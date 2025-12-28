import json
import pandas as pd
import numpy as np
from wsba_hockey.tools.xg_model import *

## AGGREGATE FUNCTIONS ##

## GLOBAL VARIABLES ##
shot_types = ['wrist','deflected','tip-in','slap','backhand','snap','wrap-around','poke','bat','cradle','between-legs']
fenwick_events = ['missed-shot','shot-on-goal','goal']

strengths_list = {
    'EV':['5v5','4v4','3v3'],
    'PP':['5v4','5v3','4v3'],
    'SH':['4v5','3v5','3v4']
}

per_sixty = ['Fi','xGi','Gi','A1','A2','P1','P','Si','OZF','NZF','DZF','FF','FA','xGF','xGA','GF','GA','SF','SA','CF','CA','HF','HA','Give','Take','Penl','Penl2','Penl5','Draw','PIM','Block','GSAx']

def calc_indv(pbp,game_strength,second_group):
    # Filter by game strength if not "all"
    if game_strength != "all":
        pbp = pbp.loc[pbp['strength_state'].isin(game_strength)]
        
    #Add second event-team column for necessary situations
    pbp['event_team_abbr_2'] = np.where(pbp['event_team_abbr'].notna(),
        np.where(pbp['event_team_abbr']==pbp['home_team_abbr'],pbp['away_team_abbr'],pbp['home_team_abbr']),np.nan)

    #Change second event team to goal-scoring team for goal events
    pbp['event_team_abbr_2'] = np.where(pbp['event_type']=='goal',pbp['event_team_abbr'],pbp['event_team_abbr_2'])

    #Determine how to group
    raw_group_1 = ['event_player_1_id','event_team_abbr']+second_group
    raw_group_2 = ['event_player_2_id','event_team_abbr_2']+second_group
    raw_group_3 = ['event_player_3_id','event_team_abbr']+second_group
    clean_group = ['ID','Team','Season']+(['Game'] if 'game_id' in second_group else [])

    #First event player stats
    ep1 = (
        pbp.loc[pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot","blocked-shot",'hit','giveaway','takeaway','faceoff','penalty'])].groupby(raw_group_1).agg(
        Gi=('event_type', lambda x: (x == "goal").sum()),
        Si=('event_type', lambda x: (x.isin(['shot-on-goal','goal'])).sum()),
        Fi=('event_type', lambda x: (x.isin(fenwick_events)).sum()),
        Ci=('event_type', lambda x: (x.isin(fenwick_events+['blocked-shot'])).sum()),
        xGi=('xG', 'sum'),
        HF=('event_type',lambda x: (x=='hit').sum()),
        Give=('event_type',lambda x: (x=='giveaway').sum()),
        Take=('event_type',lambda x: (x=='takeaway').sum()),
        Penl=('event_type',lambda x: (x=='penalty').sum()),
        Penl2=('penalty_duration',lambda x: (x==2).sum()),
        Penl5=('penalty_duration',lambda x: (x==5).sum()),
        PIM=('penalty_duration','sum'),
        FW=('event_type',lambda x: (x=='faceoff').sum())
    ).reset_index().rename(columns={'event_player_1_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season', 'game_id':'Game'})
    )

    #Second event player stats
    ep2 = (
        pbp.loc[(pbp['event_type'].isin(['goal','blocked-shot','hit','faceoff','penalty']))&~(pbp['description'].str.lower().str.contains('blocked by teammate',na=False))].groupby(raw_group_2).agg(
        A1=('event_type',lambda x: (x=='goal').sum()),
        HA=('event_type',lambda x: (x=='hit').sum()),
        Draw=('event_type',lambda x: (x=='penalty').sum()),
        FL=('event_type',lambda x: (x=='faceoff').sum()),
        Block=('event_type',lambda x:(x=='blocked-shot').sum())
    ).reset_index().rename(columns={'event_player_2_id': 'ID', 'event_team_abbr_2': 'Team', 'season': 'Season', 'game_id':'Game'})
    )

    #Third event player stats
    ep3 = (
        pbp.loc[pbp['event_type'].isin(["goal"])].groupby(raw_group_3).agg(
        A2=('event_type', 'count')
    ).reset_index().rename(columns={'event_player_3_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season', 'game_id':'Game'})
    )
    
    #Rush events
    rush = (
        pbp.loc[(pbp['event_type'].isin(fenwick_events))&(pbp['rush']>0)].groupby(raw_group_1).agg(
        Rush=('event_type','count'),
        Rush_G=('event_type',lambda x: (x == 'goal').sum()),
        Rush_xG=('xG','sum')
    ).reset_index().rename(columns={'event_player_1_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season', 'game_id':'Game', 'Rush_G': 'Rush G', 'Rush_xG': 'Rush xG'})
    )

    indv = pd.merge(ep1,ep2,how='outer',on=clean_group)
    indv = pd.merge(indv,ep3,how='outer',on=clean_group)
    indv = pd.merge(indv,rush,how='outer',on=clean_group)

    #Shot Types
    for type in shot_types:
        shot = (
            pbp.loc[(pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot"])&(pbp['shot_type']==type))].groupby(raw_group_1).agg(
            Gi=('event_type', lambda x: (x == "goal").sum()),
            Si=('event_type', lambda x: (x.isin(['shot-on-goal','goal'])).sum()),
            Fi=('event_type', lambda x: (x != "blocked-shot").sum()),
            xGi=('xG', 'sum'),
        ).reset_index().rename(columns={'event_player_1_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season', 'game_id':'Game'})
        )

        shot = shot.rename(columns={
            'Gi':f'{type.capitalize()}Gi',
            'Si':f'{type.capitalize()}Si',
            'Fi':f'{type.capitalize()}Fi',
            'xGi':f'{type.capitalize()}xGi',
        })
        indv = pd.merge(indv,shot,how='outer',on=clean_group)

    indv[['Gi','A1','A2','Penl','Draw','FW','FL']] = indv[['Gi','A1','A2','Penl','Draw','FW','FL']].fillna(0)

    indv['P1'] = indv['Gi']+indv['A1']
    indv['P'] = indv['P1']+indv['A2']
    indv['Shi%'] = indv['Gi']/indv['Si']
    indv['xGi/Fi'] = indv['xGi']/indv['Fi']
    indv['Gi/xGi'] = indv['Gi']/indv['xGi']
    indv['Fshi%'] = indv['Gi']/indv['Fi']
    indv['F'] = indv['FW']+indv['FL']
    indv['F%'] = indv['FW']/indv['F']
    indv['PM%'] = indv['Take']/(indv['Give']+indv['Take'])
    indv['HF%'] = indv['HF']/(indv['HF']+indv['HA'])
    indv['PENL%'] = indv['Draw']/(indv['Draw']+indv['Penl'])

    return indv

def calc_onice(pbp,game_strength,second_group):
    #Convert player on-ice columns to vectors
    pbp['home_on_ice'] = pbp['home_on_1_id'].astype(str) + ";" + pbp['home_on_2_id'].astype(str) + ";" + pbp['home_on_3_id'].astype(str) + ";" + pbp['home_on_4_id'].astype(str) + ";" + pbp['home_on_5_id'].astype(str) + ";" + pbp['home_on_6_id'].astype(str)
    pbp['away_on_ice'] = pbp['away_on_1_id'].astype(str) + ";" + pbp['away_on_2_id'].astype(str) + ";" + pbp['away_on_3_id'].astype(str) + ";" + pbp['away_on_4_id'].astype(str) + ";" + pbp['away_on_5_id'].astype(str) + ";" + pbp['away_on_6_id'].astype(str)
    
    #Remove NA players
    pbp['home_on_ice'] = pbp['home_on_ice'].str.replace(';nan', '', regex=True)
    pbp['away_on_ice'] = pbp['away_on_ice'].str.replace(';nan', '', regex=True)

    def process_team_stats(df, on_ice_col, team_col, opp_col, game_strength):
        df = df[['season','game_id','strength_state','event_num', team_col, opp_col, 'event_type', 'event_team_venue','event_team_abbr', on_ice_col,'ids_on','shift_type','event_length','zone_code','xG']].copy()

        #Flip strength state (when necessary) and filter by game strength if not "all"
        if game_strength != "all":
            if game_strength not in ['3v3','4v4','5v5']:
                for strength in game_strength:
                    df['strength_state'] = np.where(np.logical_and(df['event_team_venue']==opp_col[0:4],df['strength_state']==strength[::-1]),strength,df['strength_state'])

            df = df.loc[df['strength_state'].isin(game_strength)]

        df[on_ice_col] = df[on_ice_col].str.split(';')
        df = df.explode(on_ice_col)
        df = df.rename(columns={on_ice_col: 'ID', 'season': 'Season'})
        df['xGF'] = np.where(df['event_team_abbr'] == df[team_col], df['xG'], 0)
        df['xGA'] = np.where(df['event_team_abbr'] == df[opp_col], df['xG'], 0)
        df['GF'] = np.where((df['event_type'] == "goal") & (df['event_team_abbr'] == df[team_col]), 1, 0)
        df['GA'] = np.where((df['event_type'] == "goal") & (df['event_team_abbr'] == df[opp_col]), 1, 0)
        df['SF'] = np.where((df['event_type'].isin(['shot-on-goal','goal'])) & (df['event_team_abbr'] == df[team_col]), 1, 0)
        df['SA'] = np.where((df['event_type'].isin(['shot-on-goal','goal'])) & (df['event_team_abbr'] == df[opp_col]), 1, 0)
        df['FF'] = np.where((df['event_type'].isin(fenwick_events)) & (df['event_team_abbr'] == df[team_col]), 1, 0)
        df['FA'] = np.where((df['event_type'].isin(fenwick_events)) & (df['event_team_abbr'] == df[opp_col]), 1, 0)
        df['CF'] = np.where((df['event_type'].isin(fenwick_events+['blocked-shot'])) & (df['event_team_abbr'] == df[team_col]), 1, 0)
        df['CA'] = np.where((df['event_type'].isin(fenwick_events+['blocked-shot'])) & (df['event_team_abbr'] == df[opp_col]), 1, 0)
        df['OZF'] = np.where((df['event_type']=='faceoff') & ((df['zone_code']=='O')&((df['event_team_abbr'] == df[team_col])) | (df['zone_code']=='D')&((df['event_team_abbr'] == df[opp_col]))), 1, 0)
        df['NZF'] = np.where((df['zone_code']=='N') & (df['event_team_abbr']==df[team_col]),1,0)
        df['DZF'] = np.where((df['event_type']=='faceoff') & ((df['zone_code']=='D')&((df['event_team_abbr'] == df[team_col])) | (df['zone_code']=='O')&((df['event_team_abbr'] == df[opp_col]))), 1, 0)

        stats = df.groupby(['ID',team_col,'Season']+(['game_id'] if 'game_id' in second_group else [])).agg(
            GP=('game_id','nunique'),
            TOI=('event_length','sum'),
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            SF=('SF', 'sum'),
            SA=('SA', 'sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum'),
            CF=('CF','sum'),
            CA=('CA','sum'),
            OZF=('OZF','sum'),
            NZF=('NZF','sum'),
            DZF=('DZF','sum')
        ).reset_index()
        
        return stats.rename(columns={team_col:"Team", 'game_id':'Game'})
    
    home_stats = process_team_stats(pbp, 'home_on_ice', 'home_team_abbr', 'away_team_abbr',game_strength)
    away_stats = process_team_stats(pbp, 'away_on_ice', 'away_team_abbr', 'home_team_abbr',game_strength)

    onice_stats = pd.concat([home_stats,away_stats]).groupby(['ID','Team','Season']+(['Game'] if 'game_id' in second_group else [])).agg(
            GP=('GP','sum'),
            TOI=('TOI','sum'),
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            SF=('SF', 'sum'),
            SA=('SA', 'sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum'),
            CF=('CF','sum'),
            CA=('CA','sum'),
            OZF=('OZF','sum'),
            NZF=('NZF','sum'),
            DZF=('DZF','sum')
    ).reset_index()

    onice_stats['ShF%'] = onice_stats['GF']/onice_stats['SF']
    onice_stats['xGF/FF'] = onice_stats['xGF']/onice_stats['FF']
    onice_stats['GF/xGF'] = onice_stats['GF']/onice_stats['xGF']
    onice_stats['FshF%'] = onice_stats['GF']/onice_stats['FF']
    onice_stats['ShA%'] = onice_stats['GA']/onice_stats['SA']
    onice_stats['xGA/FA'] = onice_stats['xGA']/onice_stats['FA']
    onice_stats['GA/xGA'] = onice_stats['GA']/onice_stats['xGA']
    onice_stats['FshA%'] = onice_stats['GA']/onice_stats['FA']
    onice_stats['OZF%'] = onice_stats['OZF']/(onice_stats['OZF']+onice_stats['NZF']+onice_stats['DZF'])
    onice_stats['NZF%'] = onice_stats['NZF']/(onice_stats['OZF']+onice_stats['NZF']+onice_stats['DZF'])
    onice_stats['DZF%'] = onice_stats['DZF']/(onice_stats['OZF']+onice_stats['NZF']+onice_stats['DZF'])
    onice_stats['GSAx'] = onice_stats['xGA']-onice_stats['GA']

    return onice_stats

def calc_team(pbp,game_strength,second_group):
    teams = []
    for team in [('away','home'),('home','away')]:
        #Flip strength state (when necessary) and filter by game strength if not "all"
        if game_strength != "all":
            if game_strength not in ['3v3','4v4','5v5']:
                for strength in game_strength:
                    pbp['strength_state'] = np.where(np.logical_and(pbp['event_team_venue']==team[1],pbp['strength_state']==strength[::-1]),strength,pbp['strength_state'])

            pbp = pbp.loc[pbp['strength_state'].isin(game_strength)]

        pbp['xGF'] = np.where(pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'], pbp['xG'], 0)
        pbp['xGA'] = np.where(pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'], pbp['xG'], 0)
        pbp['GF'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['GA'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['SF'] = np.where((pbp['event_type'].isin(['shot-on-goal','goal'])) & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['SA'] = np.where((pbp['event_type'].isin(['shot-on-goal','goal'])) & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['FF'] = np.where((pbp['event_type'].isin(fenwick_events)) & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['FA'] = np.where((pbp['event_type'].isin(fenwick_events)) & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['CF'] = np.where((pbp['event_type'].isin(fenwick_events+['blocked-shot'])) & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['CA'] = np.where((pbp['event_type'].isin(fenwick_events+['blocked-shot'])) & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['HF'] =  np.where((pbp['event_type']=='hit') & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['HA'] = np.where((pbp['event_type']=='hit') & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['Penl'] = np.where((pbp['event_type']=='penalty') & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['Penl2'] = np.where((pbp['event_type']=='penalty') & (pbp['penalty_duration']==2) & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['Penl5'] = np.where((pbp['event_type']=='penalty') & (pbp['penalty_duration']==5) & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['PIM'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), pbp['penalty_duration'], 0)
        pbp['Draw'] = np.where((pbp['event_type']=='penalty') & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['Give'] = np.where((pbp['event_type']=='giveaway') & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['Take'] = np.where((pbp['event_type']=='takeaway') & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['Block'] = pbp['CA'] - pbp['FA']
        pbp['RushF'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'])&(pbp['rush']>0), 1, 0)
        pbp['RushA'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'])&(pbp['rush']>0), 1, 0)
        pbp['RushFxG'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'])&(pbp['rush']>0), pbp['xG'], 0)
        pbp['RushAxG'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'])&(pbp['rush']>0), pbp['xG'], 0)
        pbp['RushFG'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'])&(pbp['rush']>0), 1, 0)
        pbp['RushAG'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'])&(pbp['rush']>0), 1, 0)

        stats = pbp.groupby([f'{team[0]}_team_abbr']+second_group).agg(
            GP=('game_id','nunique'),
            TOI=('event_length','sum'),
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            SF=('SF','sum'),
            SA=('SA','sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum'),
            CF=('CF','sum'),
            CA=('CA','sum'),
            HF=('HF','sum'),
            HA=('HA','sum'),
            Penl=('Penl','sum'),
            Penl2=('Penl2','sum'),
            Penl5=('Penl5','sum'),
            PIM=('PIM','sum'),
            Draw=('Draw','sum'),
            Give=('Give','sum'),
            Take=('Take','sum'),
            Block=('Block','sum'),
            RushF=('RushF','sum'),
            RushA=('RushA','sum'),
            RushFxG=('RushFxG','sum'),
            RushAxG=('RushAxG','sum'),
            RushFG=('RushFG','sum'),
            RushAG=('RushAG','sum'),
        ).reset_index().rename(columns={f'{team[0]}_team_abbr':"Team",'season':"Season",'game_id':'Game'})
        teams.append(stats)
    
    onice_stats = pd.concat(teams).groupby(['Team','Season']+(['Game'] if 'game_id' in second_group else [])).agg(
            GP=('GP','sum'),
            TOI=('TOI','sum'),
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            SF=('SF','sum'),
            SA=('SA','sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum'),
            CF=('CF','sum'),
            CA=('CA','sum'),
            HF=('HF','sum'),
            HA=('HA','sum'),
            Penl=('Penl','sum'),
            Penl2=('Penl2','sum'),
            Penl5=('Penl5','sum'),
            PIM=('PIM','sum'),
            Draw=('Draw','sum'),
            Give=('Give','sum'),
            Take=('Take','sum'),
            Block=('Block','sum'),
            RushF=('RushF','sum'),
            RushA=('RushA','sum'),
            RushFxG=('RushFxG','sum'),
            RushAxG=('RushAxG','sum'),
            RushFG=('RushFG','sum'),
            RushAG=('RushAG','sum'),
    ).reset_index()

    onice_stats['ShF%'] = onice_stats['GF']/onice_stats['SF']
    onice_stats['xGF/FF'] = onice_stats['xGF']/onice_stats['FF']
    onice_stats['GF/xGF'] = onice_stats['GF']/onice_stats['xGF']
    onice_stats['FshF%'] = onice_stats['GF']/onice_stats['FF']
    onice_stats['ShA%'] = onice_stats['GA']/onice_stats['SA']
    onice_stats['xGA/FA'] = onice_stats['xGA']/onice_stats['FA']
    onice_stats['GA/xGA'] = onice_stats['GA']/onice_stats['xGA']
    onice_stats['FshA%'] = onice_stats['GA']/onice_stats['FA']
    onice_stats['PM%'] = onice_stats['Take']/(onice_stats['Give']+onice_stats['Take'])
    onice_stats['HF%'] = onice_stats['HF']/(onice_stats['HF']+onice_stats['HA'])
    onice_stats['PENL%'] = onice_stats['Draw']/(onice_stats['Draw']+onice_stats['Penl'])
    onice_stats['GSAx'] = onice_stats['xGA']/onice_stats['GA']

    return onice_stats

def calc_goalie(pbp,game_strength,second_group):
    teams=[]
    for team in [('away','home'),('home','away')]:
        #Flip strength state (when necessary) and filter by game strength if not "all"
        if game_strength != "all":
            if game_strength not in ['3v3','4v4','5v5']:
                for strength in game_strength:
                    pbp['strength_state'] = np.where(np.logical_and(pbp['event_team_venue']==team[1],pbp['strength_state']==strength[::-1]),strength,pbp['strength_state'])

            pbp = pbp.loc[pbp['strength_state'].isin(game_strength)]

        pbp['xGF'] = np.where(pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'], pbp['xG'], 0)
        pbp['xGA'] = np.where(pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'], pbp['xG'], 0)
        pbp['GF'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['GA'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['SF'] = np.where((pbp['event_type'].isin(['shot-on-goal','goal'])) & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['SA'] = np.where((pbp['event_type'].isin(['shot-on-goal','goal'])) & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['FF'] = np.where((pbp['event_type'].isin(fenwick_events)) & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['FA'] = np.where((pbp['event_type'].isin(fenwick_events)) & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['CF'] = np.where((pbp['event_type'].isin(fenwick_events+['blocked-shot'])) & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr']), 1, 0)
        pbp['CA'] = np.where((pbp['event_type'].isin(fenwick_events+['blocked-shot'])) & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr']), 1, 0)
        pbp['RushF'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'])&(pbp['rush']>0), 1, 0)
        pbp['RushA'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'])&(pbp['rush']>0), 1, 0)
        pbp['RushFxG'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'])&(pbp['rush']>0), pbp['xG'], 0)
        pbp['RushAxG'] = np.where((pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'])&(pbp['rush']>0), pbp['xG'], 0)
        pbp['RushFG'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[0]}_team_abbr'])&(pbp['rush']>0), 1, 0)
        pbp['RushAG'] = np.where((pbp['event_type'] == "goal") & (pbp['event_team_abbr'] == pbp[f'{team[1]}_team_abbr'])&(pbp['rush']>0), 1, 0)

        stats = pbp.groupby([f'{team[0]}_goalie_id',f'{team[0]}_team_abbr']+second_group).agg(
            GP=('game_id','nunique'),
            TOI=('event_length','sum'),
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            SF=('SF', 'sum'),
            SA=('SA', 'sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum'),
            CF=('CF','sum'),
            CA=('CA','sum'),
            RushF=('RushF','sum'),
            RushA=('RushA','sum'),
            RushFxG=('RushFxG','sum'),
            RushAxG=('RushAxG','sum'),
            RushFG=('RushFG','sum'),
            RushAG=('RushAG','sum'),
        ).reset_index().rename(columns={f'{team[0]}_goalie_id':"ID",f'{team[0]}_team_abbr':"Team",'season':"Season",'game_id':'Game'})
        teams.append(stats)
    
    onice_stats = pd.concat(teams).groupby(['ID','Team','Season']+(['Game'] if 'game_id' in second_group else [])).agg(
            GP=('GP','sum'),
            TOI=('TOI','sum'),
            FF=('FF', 'sum'),
            FA=('FA', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            SF=('SF', 'sum'),
            SA=('SA', 'sum'),
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum'),
            CF=('CF','sum'),
            CA=('CA','sum'),
            RushF=('RushF','sum'),
            RushA=('RushA','sum'),
            RushFxG=('RushFxG','sum'),
            RushAxG=('RushAxG','sum'),
            RushFG=('RushFG','sum'),
            RushAG=('RushAG','sum'),
    ).reset_index()

    onice_stats['ShF%'] = onice_stats['GF']/onice_stats['SF']
    onice_stats['xGF/FF'] = onice_stats['xGF']/onice_stats['FF']
    onice_stats['GF/xGF'] = onice_stats['GF']/onice_stats['xGF']
    onice_stats['FshF%'] = onice_stats['GF']/onice_stats['FF']
    onice_stats['ShA%'] = onice_stats['GA']/onice_stats['SA']
    onice_stats['xGA/FA'] = onice_stats['xGA']/onice_stats['FA']
    onice_stats['GA/xGA'] = onice_stats['GA']/onice_stats['xGA']
    onice_stats['FshA%'] = onice_stats['GA']/onice_stats['FA']
    onice_stats['GSAx'] = onice_stats['xGA']-onice_stats['GA']

    return onice_stats

def calc_game_score_features(pbp,type):
    clean_group = ['ID','Team','Season','Game']
    second_group = ['season','game_id']

    team_stats = calc_team(pbp,'all',['season','game_id'])[['Team','Season','Game','GF','GA']].rename(columns={'GF':'Team GF', 'GA':'Team GA'})

    if type == 'skater':
        df = calc_indv(pbp,'all',second_group)[
            clean_group+
            ['P','PENL%','PM%','F%']
        ]
        
        for key, strengths in strengths_list.items():
            indv = calc_indv(pbp,strengths,second_group)[
                clean_group+
                ['xGi']
            ]
            onice = calc_onice(pbp,strengths,second_group)[
                clean_group+
                ['xGF','xGA']
            ]

            indv['ID'] = indv['ID'].astype(float)
            onice['ID'] = onice['ID'].astype(float)

            stats = pd.merge(indv,onice,how='left')
            stats['xGC%'] = stats['xGi']/stats['xGF']
            stats = stats.rename(columns={col:f'{key}_{col}' for col in stats.columns if col not in clean_group})

            df = pd.merge(df,stats,how='left')
        
        team_stats['T-GD'] = team_stats['Team GF'] - team_stats['Team GA']
        
        df = pd.merge(df,team_stats,how='left')
        score = df.replace([np.inf,-np.inf],np.nan).fillna(0)

    else:  
        score = calc_goalie(pbp,'all',['season','game_id'])[['ID','Season','Team','Game','xGF','xGA','GA/xGA']]
        score = pd.merge(score,team_stats,how='left')

        score['xGF%'] = score['xGF']/(score['xGF']+score['xGA'])
        score['T-GD'] = score['Team GF'] - score['Team GA']
        score = score.drop(columns=['xGF','xGA','Team GF','Team GA']).replace([np.inf,-np.inf],np.nan).fillna(0)

    return score

def shooting_impacts(agg, type):
    #Given stats table generated from the nhl_calculate_stats function, return table with shot impacts
    #Only 5v5 is supported as of now

    #param 'agg' - stats table
    #param 'type' - type of stats to calculate ('skater', 'goalie', or 'team')

    #COMPOSITE IMPACT EVALUATIONS:

    #SR = Shot Rate
    #SQ = Shot Quality
    #FN = Finishing

    #I = Impact

    #INDV = Individual
    #OOFF = On-Ice Offense
    #ODEF = On-Ice Defense

    #Grouping-Metric Code: XXXX-YYI

    #Goal Composition Formula
    #The aggregation of goals is composed of three factors: shot rate, shot quality, and finishing
    #These are represented by their own metrics in which:
    #Goals = (Fenwick*(League Average Fenwick SH%)) + ((xGoals/Fenwick - League Average Fenwick SH%)*Fenwick) + (Goals - xGoals)

    def goal_comp(fenwick, xg_fen, xg, g, fsh):
        rate = fenwick * fsh
        qual = (xg_fen - fsh) * fenwick
        fini = g - xg
        return rate + qual + fini

    def calc_group(pos, group):
        #Have to set these columns for compatibility with df.apply
        suf = group[1]

        pos['fsh'] = pos[f'Fsh{suf}%']
        pos['fenwick'] = pos[f'F{suf}/60']
        pos['xg'] = pos[f'xG{suf}/60']
        pos['g'] = pos[f'G{suf}/60']
        pos['xg_fen'] = pos[f'xG{suf}/F{suf}']
        pos['finishing'] = pos[f'G{suf}/xG{suf}']

        #Find average for position in frame
        avg_fen = pos['fenwick'].mean()
        avg_xg = pos['xg'].mean()
        avg_g = pos['g'].mean()
        avg_fsh = avg_g / avg_fen
        avg_xg_fen = avg_xg / avg_fen

        #Calculate composite percentiles
        pos[f'{group[0]}-SR'] = pos['fenwick'].rank(pct=True)
        pos[f'{group[0]}-SQ'] = pos['xg_fen'].rank(pct=True)
        pos[f'{group[0]}-FN'] = pos['finishing'].rank(pct=True)

        #Calculate shot rate, shot quality, and finishing impacts
        pos[f'{group[0]}-SRI'] = pos['g'] - pos.apply(
            lambda x: goal_comp(avg_fen, x.xg_fen, x.xg, x.g, avg_fsh), axis=1
        )
        pos[f'{group[0]}-SQI'] = pos['g'] - pos.apply(
            lambda x: goal_comp(x.fenwick, avg_xg_fen, x.xg, x.g, avg_fsh), axis=1
        )
        pos[f'{group[0]}-FNI'] = pos['g'] - pos.apply(
            lambda x: goal_comp(x.fenwick, x.xg_fen, avg_xg, avg_g, avg_fsh), axis=1
        )

        #Convert impacts to totals
        pos[f'{group[0]}-SRI-T'] = (pos[f'{group[0]}-SRI'] / 60) * pos['TOI']
        pos[f'{group[0]}-SQI-T'] = (pos[f'{group[0]}-SQI'] / 60) * pos['TOI']
        pos[f'{group[0]}-FNI-T'] = (pos[f'{group[0]}-FNI'] / 60) * pos['TOI']

        return pos.drop(columns=['fsh','fenwick','xg_fen','xg','g','finishing'])

    if type in ('goalie', 'team'):
        pos = agg[agg['TOI'] >= 150]
        non = agg[agg['TOI'] < 150]

        #Loop through all groupings generating impacts
        for group in [('OOFF','F'), ('ODEF','A')]:
            pos = calc_group(pos, group)

        #Flip against metric percentiles
        pos['ODEF-SR'] = 1 - pos['ODEF-SR']
        pos['ODEF-SQ'] = 1 - pos['ODEF-SQ']
        pos['ODEF-FN'] = 1 - pos['ODEF-FN']

        #Extraneous Values
        pos['EGF'] = pos['OOFF-SRI'] + pos['OOFF-SQI'] + pos['OOFF-FNI']
        pos['ExGF'] = pos['OOFF-SRI'] + pos['OOFF-SQI']
        pos['EGA'] = pos['ODEF-SRI'] + pos['ODEF-SQI'] + pos['ODEF-FNI']
        pos['ExGA'] = pos['ODEF-SRI'] + pos['ODEF-SQI']

        #...and their percentiles and totals
        for stat in ['EGF','ExGF','EGA','ExGA']:
            pos[f'{stat}-P'] = pos[stat].rank(pct=True)
            pos[f'{stat}-T'] = (pos[stat] / 60) * pos['TOI']

        #Goal Composites...
        pos['NetGI'] = pos['EGF'] - pos['EGA']
        pos['NetxGI'] = pos['ExGF'] - pos['ExGA']
        pos['Team-Adjusted-EGI'] = pos['ODEF-FNI'] - pos['ExGA']
        pos['GISAx'] = pos['ExGA'] - pos['EGA']

        #...and their percentiles and totals
        for stat in ['NetGI','NetxGI','Team-Adjusted-EGI','GISAx']:
            pos[f'{stat}-P'] = pos[stat].rank(pct=True)
            pos[f'{stat}-T'] = (pos[stat] / 60) * pos['TOI']

        #Per-sixty stats and extra Stats... and their percentiles
        for stat in per_sixty:
            try:
                pos[f'{stat}/60-P'] = pos[f'{stat}/60'].rank(pct=True)
            except KeyError:
                pass
        
        #Flip percentiles for against stats
        for stat in ['FA','xGA','GA','CA','HA','Give','Penl','Penl2','Penl5']:
            try:
                pos[f'{stat}/60-P'] = 1-pos[f'{stat}/60-P']
            except KeyError:
                pass

        for p in ['GF%','SF%','xGF%','FF%','CF%']:
            pos[f'{p}-P'] = pos[p].rank(pct=True)

        #Add extra metrics
        pos['RushF/60'] = (pos['RushF']/pos['TOI'])*60
        pos['RushA/60'] = (pos['RushA']/pos['TOI'])*60
        pos['RushesFF'] = pos['RushF/60'].rank(pct=True)
        pos['RushesFA'] = 1 - pos['RushA/60'].rank(pct=True)
        pos['RushFxG/60'] = (pos['RushFxG']/pos['TOI'])*60
        pos['RushAxG/60'] = (pos['RushAxG']/pos['TOI'])*60
        pos['RushesxGF'] = pos['RushFxG/60'].rank(pct=True)
        pos['RushesxGA'] = 1 - pos['RushAxG/60'].rank(pct=True)
        pos['RushFG/60'] = (pos['RushFG']/pos['TOI'])*60
        pos['RushAG/60'] = (pos['RushAG']/pos['TOI'])*60
        pos['RushesGF'] = pos['RushFG/60'].rank(pct=True)
        pos['RushesGA'] = 1 - pos['RushAG/60'].rank(pct=True)

        sort_cols = ['Goalie','Season','Team'] if type == 'goalie' else ['Season','Team']
        return pd.concat([pos,non]).sort_values(sort_cols)

    else:
        #Remove skaters with less than 150 minutes of TOI then split between forwards and dmen
        forwards = agg[(agg['Position'] != 'D') & (agg['TOI'] >= 150)]
        defensemen = agg[(agg['Position'] == 'D') & (agg['TOI'] >= 150)]
        non_players = agg[agg['TOI'] < 150]

        #Loop through both positions, all groupings generating impacts
        frames = []
        for pos in [forwards, defensemen]:
            df = pos.copy()
            for group in [('INDV','i'), ('OOFF','F'), ('ODEF','A')]:
                df = calc_group(df, group)

            #Flip against metric percentiles
            df['ODEF-SR'] = 1 - df['ODEF-SR']
            df['ODEF-SQ'] = 1 - df['ODEF-SQ']
            df['ODEF-FN'] = 1 - df['ODEF-FN']

            #Extraneous Values
            df['EGi'] = df['INDV-SRI'] + df['INDV-SQI'] + df['INDV-FNI']
            df['ExGi'] = df['INDV-SRI'] + df['INDV-SQI']
            df['EGF'] = df['OOFF-SRI'] + df['OOFF-SQI'] + df['OOFF-FNI']
            df['ExGF'] = df['OOFF-SRI'] + df['OOFF-SQI']
            df['EGA'] = df['ODEF-SRI'] + df['ODEF-SQI'] + df['ODEF-FNI']
            df['ExGA'] = df['ODEF-SRI'] + df['ODEF-SQI']

            #...and their percentiles and totals
            for stat in ['EGi','ExGi','EGF','ExGF','EGA','ExGA']:
                df[f'{stat}-P'] = df[stat].rank(pct=True)
                df[f'{stat}-T'] = (df[stat] / 60) * df['TOI']

            #Goal Composites...
            df['NetGI'] = df['EGF'] - df['EGA']
            df['NetxGI'] = df['ExGF'] - df['ExGA']
            df['Team-Adjusted-EGI'] = df['ODEF-FNI'] - df['ExGA']
            df['GISAx'] = df['ExGA'] - df['EGA']
            df['LiEG'] = df['EGF'] - df['EGi']
            df['LiExG'] = df['ExGF'] - df['ExGi']
            df['LiGIn'] = df['LiEG'] * df['AC%']
            df['LixGIn'] = df['LiExG'] * df['AC%']
            df['ALiGIn'] = df['LiGIn'] - df['LixGIn']
            df['CompGI'] = df['EGi'] + df['LiGIn']
            df['LiRelGI'] = df['CompGI'] - (df['EGF'] - df['CompGI'])

            #...and their percentiles and totals
            for stat in ['NetGI','NetxGI','Team-Adjusted-EGI','GISAx','LiEG','LiExG','LiGIn','LixGIn','ALiGIn','CompGI','LiRelGI']:
                df[f'{stat}-P'] = df[stat].rank(pct=True)
                df[f'{stat}-T'] = (df[stat] / 60) * df['TOI']            
            
            #Per-sixty stats and extra stats... and their percentiles
            for stat in per_sixty:
                try:
                    df[f'{stat}/60-P'] = df[f'{stat}/60'].rank(pct=True)
                except KeyError:
                    pass

            for p in ['GC%','AC%','GI%','FC%','xGC%','GF%','SF%','xGF%','FF%','CF%']:
                df[f'{p}-P'] = df[p].rank(pct=True)
            
            #Rush stats... and their percentiles
            df['RushFi/60'] = (df['Rush']/df['TOI'])*60
            df['RushxGi/60'] = (df['Rush xG']/df['TOI'])*60
            df['RushesxGi'] = df['RushxGi/60'].rank(pct=True)
            df['RushesFi'] = df['RushFi/60'].rank(pct=True)

            frames.append(df)

        complete = pd.concat(frames)

        #Flip percentiles for against stats
        for stat in ['FA','xGA','GA','CA','HA','Give','Penl','Penl2','Penl5']:
            try:
                complete[f'{stat}/60-P'] = 1-complete[f'{stat}/60-P']
            except KeyError:
                pass

        #Add back skaters with less than 150 minutes TOI
        return pd.concat([complete, non_players]).sort_values(['Player','Season','Team','ID'])

def apply_rosters(df,type,roster_path):
    #Apply roster information to stats dataframe

    #Nothing to add for teams
    if type == 'team':
        return df
    else:
        #Import rosters and player info
        rosters = pd.read_csv(roster_path)
        names = rosters[['player_id','player_name',
                            'headshot','position','handedness',
                            'height_in','weight_lbs',
                            'birth_date','birth_country']].drop_duplicates(subset=['player_id'],keep='last')

        #Add names
        complete = pd.merge(df,names,how='left',left_on=['ID'],right_on=['player_id'])

        #Rename if there are no missing names'
        complete = complete.rename(columns={'player_name':'Goalie' if type=='goalie' else 'Player',
                                            'headshot':'Headshot',
                                            'position':'Position',
                                            'handedness':'Handedness',
                                            'height_in':'Height (in)',
                                            'weight_lbs':'Weight (lbs)',
                                            'birth_date':'Birthday',
                                            'birth_country':'Nationality'}).drop(columns=['player_id'])

        #Add player age
        complete['Birthday'] = pd.to_datetime(complete['Birthday'])
        complete['Year'] = complete['Season'].astype(str).str[4:8].astype(int)
        complete['Age'] = complete['Year'] - complete['Birthday'].dt.year

        #Find player headshot
        complete['Headshot'] = 'https://assets.nhle.com/mugs/nhl/'+complete['Season'].astype(str)+'/'+complete['Team']+'/'+complete['ID'].astype(int).astype(str)+'.png'

        #Add WSBA ID
        complete['WSBA'] = complete['ID'].astype(str).str.replace('.0','')+complete['Season'].astype(str)+complete['Team']

        #Remove goaltenders from skater dataframes
        if type == 'skater':
            complete = complete.loc[complete['Position']!='G']
        elif type == 'game_score':
            complete = complete.loc[(complete['Position']!='G') | ((complete['Position']=='G')&(complete['P'].isna()))]

        #Return dataframe with stats info
        return complete
