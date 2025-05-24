import pandas as pd
import numpy as np
from .xg_model import *

## AGGREGATE FUNCTIONS ##

## GLOBAL VARIABLES ##
shot_types = ['wrist','deflected','tip-in','slap','backhand','snap','wrap-around','poke','bat','cradle','between-legs']
fenwick_events = ['missed-shot','shot-on-goal','goal']

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
            Fi=('event_type', lambda x: (x != "blocked-shot").sum()),
            xGi=('xG', 'sum'),
        ).reset_index().rename(columns={'event_player_1_id': 'ID', 'event_team_abbr': 'Team', 'season': 'Season', 'game_id':'Game'})
        )

        shot = shot.rename(columns={
            'Gi':f'{type.capitalize()}Gi',
            'Fi':f'{type.capitalize()}Fi',
            'xGi':f'{type.capitalize()}xGi',
        })
        indv = pd.merge(indv,shot,how='outer',on=clean_group)

    indv[['Gi','A1','A2','Penl','Draw','FW','FL']] = indv[['Gi','A1','A2','Penl','Draw','FW','FL']].fillna(0)

    indv['P1'] = indv['Gi']+indv['A1']
    indv['P'] = indv['P1']+indv['A2']
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
            xGF=('xGF', 'sum'),
            xGA=('xGA', 'sum'),
            CF=('CF','sum'),
            CA=('CA','sum'),
            OZF=('OZF','sum'),
            NZF=('NZF','sum'),
            DZF=('DZF','sum')
    ).reset_index()

    onice_stats['xGF/FF'] = onice_stats['xGF']/onice_stats['FF']
    onice_stats['GF/xGF'] = onice_stats['GF']/onice_stats['xGF']
    onice_stats['FshF%'] = onice_stats['GF']/onice_stats['FF']
    onice_stats['xGA/FA'] = onice_stats['xGA']/onice_stats['FA']
    onice_stats['GA/xGA'] = onice_stats['GA']/onice_stats['xGA']
    onice_stats['FshA%'] = onice_stats['GA']/onice_stats['FA']
    onice_stats['OZF%'] = onice_stats['OZF']/(onice_stats['OZF']+onice_stats['NZF']+onice_stats['DZF'])
    onice_stats['NZF%'] = onice_stats['NZF']/(onice_stats['OZF']+onice_stats['NZF']+onice_stats['DZF'])
    onice_stats['DZF%'] = onice_stats['DZF']/(onice_stats['OZF']+onice_stats['NZF']+onice_stats['DZF'])

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

    onice_stats['xGF/FF'] = onice_stats['xGF']/onice_stats['FF']
    onice_stats['GF/xGF'] = onice_stats['GF']/onice_stats['xGF']
    onice_stats['FshF%'] = onice_stats['GF']/onice_stats['FF']
    onice_stats['xGA/FA'] = onice_stats['xGA']/onice_stats['FA']
    onice_stats['GA/xGA'] = onice_stats['GA']/onice_stats['xGA']
    onice_stats['FshA%'] = onice_stats['GA']/onice_stats['FA']
    onice_stats['PM%'] = onice_stats['Take']/(onice_stats['Give']+onice_stats['Take'])
    onice_stats['HF%'] = onice_stats['HF']/(onice_stats['HF']+onice_stats['HA'])
    onice_stats['PENL%'] = onice_stats['Draw']/(onice_stats['Draw']+onice_stats['Penl'])

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

    onice_stats['xGF/FF'] = onice_stats['xGF']/onice_stats['FF']
    onice_stats['GF/xGF'] = onice_stats['GF']/onice_stats['xGF']
    onice_stats['FshF%'] = onice_stats['GF']/onice_stats['FF']
    onice_stats['xGA/FA'] = onice_stats['xGA']/onice_stats['FA']
    onice_stats['GA/xGA'] = onice_stats['GA']/onice_stats['xGA']
    onice_stats['FshA%'] = onice_stats['GA']/onice_stats['FA']
    onice_stats['GSAx'] = onice_stats['xGA']-onice_stats['GA']

    return onice_stats
