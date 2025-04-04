import requests as rs
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

def get_col():
    return ['game_id','game_date','away_team_abbr','home_team_abbr',
            'period','seconds_elapsed','event_team_status',
            'event_team_abbr','description','event_type',
            'event_player_1_name','event_player_2_name',
            'event_player_3_name','penalty_duration',
            'event_goalie','away_goalie','home_goalie',
            'away_score','home_score']

def ncaa_parse_json(game_id,data,scores):
    #Given game_id and raw json for plays and goals, return complete play-by-play for specified game

    #Game info
    away = data['meta']['teams'][1]['sixCharAbbr'] if data['meta']['teams'][0]['homeTeam'] else data['meta']['teams'][0]['sixCharAbbr']
    home = data['meta']['teams'][0]['sixCharAbbr'] if data['meta']['teams'][0]['homeTeam'] else data['meta']['teams'][1]['sixCharAbbr']
    date = data['updatedTimestamp']

    #Add goals
    goals = {}
    i = 0
    for period in scores['periods']:
        #Period will be key, time will be subkey, and score description will be the value

        period['period'] = period['title']
        i += 1
        for goal in period['summary']:
            #Period time adjustments (only 'seconds_elapsed' is included in the resulting data)
            goal['period_time_simple'] = goal['time'].replace(":","")
            goal['period_seconds_elapsed'] = np.where(len(goal['period_time_simple'])==3,
                                                ((int(goal['period_time_simple'][0])*60)+int(goal['period_time_simple'][-2:])),
                                                ((int(goal['period_time_simple'][0:2])*60)+int(goal['period_time_simple'][-2:])))
            goal['seconds_elapsed'] = goal['period_seconds_elapsed'] + (1200*(i-1))
            
            goals.update({goal['seconds_elapsed']:[goal['scoreText'],goal['scoreType']]})


    #Add plays from each period
    pbp = []
    for period in data['periods']:
        #Parse event data
        #Events include goalie-on events, all corsi events and penalties
        events = []
        for event in period['playStats']:
            #Each event description is either in visitorText or homeText
            desc = event['visitorText'] if event['visitorText'] != "" else event['homeText']
            event['description'] = desc

            #Period time adjustments (only 'seconds_elapsed' is included in the resulting data)
            event['period_time_simple'] = event['time'].replace(":","")
            event['period_seconds_elapsed'] = np.where(len(event['period_time_simple'])==3,
                                                ((int(event['period_time_simple'][0])*60)+int(event['period_time_simple'][-2:])),
                                                ((int(event['period_time_simple'][0:2])*60)+int(event['period_time_simple'][-2:])))
            event['seconds_elapsed'] = abs(event['period_seconds_elapsed']-1200)
            event['seconds_elapsed'] = event['seconds_elapsed'] + (1200*(int(period['periodNumber'])-1))

            #Find event type and determine event players
            if "at goalie" in desc:
                event['event_type'] = 'goalie-on'
                event['event_goalie'] = desc.split(" ")[0].upper() + " " + desc.split(" ")[1].upper()
                event[f'{'home' if event['visitorText']=="" else 'away'}_goalie'] = event['event_goalie'] 
            elif 'Shot by' in desc:
                #Determine type of shot
                #Names formatted as last, first
                regex = re.compile(r"\b[A-Z][a-zA-ZÀ-ÿ'’-]+, [A-Z][a-zA-ZÀ-ÿ'’-]+\b")
                names = regex.findall(desc)
                #Shooter
                first = names[0].split(", ")[1].upper()
                last = names[0].split(", ")[0].upper()
                event['event_player_1_name'] = f'{first} {last}'

                #For some reason, shots on goal are "missed" shots
                if 'BLOCKED' in desc:
                    event['event_type'] = 'blocked-shot'
                    #Blocker
                    first = names[1].split(", ")[1].upper()
                    last = names[1].split(", ")[0].upper()
                    event['event_player_2_name'] = f'{first} {last}'
                elif 'WIDE' in desc:
                    event['event_type'] = 'missed-shot'
                elif 'MISSED' in desc:
                    event['event_type'] = 'shot-on-goal'

                    #Goaltender
                    first = names[1].split(", ")[1].upper()
                    last = names[1].split(", ")[0].upper()
                    event['event_goalie'] = f'{first} {last}'

            elif 'Goal by' in desc:
                event['event_type'] = 'goal'
                #There are problems with assist-tracking in the play-by-play so the scoring summary is brought in to correct
                goal_desc = goals[event['seconds_elapsed']][0]
                type = goals[event['seconds_elapsed']][1]
                event['goal_desc'] = goal_desc
                event['goal_type'] = type

                #Names formatted as last, first
                regex = re.compile(r"[A-Z][a-zA-Z'’-]*(?:\s[A-Z][a-zA-Z'’-]*)*")
                names = regex.findall(goal_desc)
                
                #Involved players
                for i in range(len(names)):
                    event[f'event_player_{i+1}_name'] = names[i].upper()

            elif 'Penalty on' in desc:
                event['event_type'] = 'penalty'

                #Names formatted as last, first
                regex = re.compile(r"\b[A-Z][a-zA-Z'’-]+, [A-Z][a-zA-Z'’-]+\b")
                names = regex.findall(desc)
                
                #Penalty taker
                first = names[0].split(", ")[1].upper()
                last = names[0].split(", ")[0].upper()
                event['event_player_1_name'] = f'{first} {last}'
                
                #Penalty duration
                regex = re.compile(r"(\d+ minutes)")
                penl = regex.findall(desc)

                event['penalty_duration'] = penl[0][:2].strip("")
            else:
                event['event_type'] = ''

            
            events.append(event)

        plays = pd.json_normalize(events)
        plays['period'] = int(period['periodNumber'])

        if int(period['periodNumber']) == 0:
            continue

        plays['away_team_abbr'] = away
        plays['home_team_abbr'] = home

        plays['event_team_abbr'] = np.where(plays['visitorText']!="",away,
                                   np.where(plays['homeText']!="",home,""))
        plays['event_team_status'] = np.where(plays['visitorText']!="","away",
                                   np.where(plays['homeText']!="","home",""))
        pbp.append(plays)
    pbp = pd.concat(pbp)
    pbp = pbp.reset_index(drop=True)

    #Score is formatted as away-home
    pbp['score'][0] = '0-0'
    pbp['score'] = pbp['score'].replace(r'^\s*$', np.nan, regex=True).ffill()

    pbp[['away_score','home_score']] = pbp['score'].str.split("-",expand=True).astype(int)

    #Goaltenders
    goalie = ['away_goalie','home_goalie']
    for col in goalie:
        pbp[col] = pbp[col].ffill()

    #Add game info
    pbp['game_id'] = game_id
    pbp['game_date'] = date

    #Correct Goal Description
    pbp['description'] = np.where(pbp['event_type']=='goal',pbp['goal_desc'],pbp['description'])

    #Add event goaltender for goals (account for empty nets)
    pbp['event_goalie'] = np.where(pbp['goal_type'].str.contains("EN"),"",np.where(pbp['event_type']=='goal',
                                   np.where(pbp['event_team_status']=='home',
                                            pbp['away_goalie'],pbp['home_goalie'])
                                            ,pbp['event_goalie']))


    #Return: formatted play-by-play
    return pbp[get_col()]