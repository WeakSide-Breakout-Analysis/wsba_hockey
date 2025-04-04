import re
from bs4 import BeautifulSoup
import requests as rs
import json as json_lib
from tools.utils.shared import *
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

### SCRAPING FUNCTIONS ###
# Provided in this file are functions vital to the scraping functions in the WSBA Hockey Python package. #

### JSON GAME INFO ###
def get_game_roster(json):
    #Given raw json data, return game rosters
    roster = pd.json_normalize(json['rosterSpots'])

    alt_name_col = ['firstName.cs','firstName.de','firstName.es','firstName.fi','firstName.sk','firstName.sv']

    #Add alt-names pattern (appropriately replaces names in shift html)
    roster['fullName.default'] = (roster['firstName.default'] + " " + roster['lastName.default']).str.upper()
    for col in alt_name_col:
        try:
            roster[f'fullName.{re.sub('firstName.',"",col)}'] = (roster[col] + " " + roster['lastName.default']).str.upper()
        except:
            continue
    return roster

def get_game_info(json):
    #Given raw json data, return game information

    base = pd.json_normalize(json)
    game_id = base['id'][0]
    season = base['season'][0]
    season_type = base['gameType'][0]
    game_date = base['gameDate'][0]
    start_time = base['startTimeUTC'][0]
    venue = base['venue.default'][0]
    venue_location = base['venueLocation.default'][0]
    away_team_id = base['awayTeam.id'][0]
    away_team_abbr = base['awayTeam.abbrev'][0]
    home_team_id = base['homeTeam.id'][0]
    home_team_abbr = base['homeTeam.abbrev'][0]

    return {"game_id":game_id,
            "season":season,
            "season_type":season_type,
            "game_date":game_date,
            "start_time":start_time,
            'venue':venue,
            'venue_location':venue_location,
            'away_team_id':away_team_id,
            'away_team_abbr':away_team_abbr,
            'home_team_id':home_team_id,
            'home_team_abbr':home_team_abbr,
            'rosters':get_game_roster(json)}

## HTML PBP DATA ###
def strip_html_pbp(td,json):
    #Harry Shomer's Code (modified)

    #JSON Prep
    info = get_game_info(json)
    roster = info['rosters']

    away = info['away_team_id'] #Away team in the HTML is the seventh column 
    home = info['home_team_id'] #Home team in the HTML is the eighth column
    away_players = roster.loc[roster['teamId']==away][['fullName.default','playerId','sweaterNumber']]
    home_players = roster.loc[roster['teamId']==home][['fullName.default','playerId','sweaterNumber']]
    
    rosters = {"away":away_players.set_index("sweaterNumber")[['playerId','fullName.default']].to_dict(),
               "home":home_players.set_index("sweaterNumber")[['playerId','fullName.default']].to_dict()
               }
    
    #HTML Parsing
    for y in range(len(td)):
        # Get the 'br' tag for the time column...this get's us time remaining instead of elapsed and remaining combined
        if y == 3:
            td[y] = td[y].get_text()   # This gets us elapsed and remaining combined-< 3:0017:00
            index = td[y].find(':')
            td[y] = td[y][:index+3]
        elif (y == 6 or y == 7) and td[0] != '#':
            # 6 & 7-> These are the player 1 ice one's
            # The second statement controls for when it's just a header
            baz = td[y].find_all('td')
            bar = [baz[z] for z in range(len(baz)) if z % 4 != 0]  # Because of previous step we get repeats...delete some

            # The setup in the list is now: Name/Number->Position->Blank...and repeat
            # Now strip all the html
            players = []
            for i in range(len(bar)):
                if i % 3 == 0:
                    try:
                        #Using the supplied json we can bind player name and id to number and team
                        #Find number and team of player then lookup roster dictionary
                        
                        number = bar[i].get_text().strip('\n')  # Get number and strip leading/trailing newlines
                        if y == 6:
                            team = 'away'
                        else:
                            team = 'home'
                        
                        id = rosters[team]['playerId'][int(number)]
                        name = rosters[team]['fullName.default'][int(number)]
                        
                    except KeyError:
                        name = ''
                        number = ''
                        id = ''
                elif i % 3 == 1:
                    if name != '':
                        position = bar[i].get_text()
                        players.append([name, number, position, id])

            td[y] = players
        else:
            td[y] = td[y].get_text()

    return td


def clean_html_pbp(html,json):
    #Harry Shomer's Code (modified)
    soup = get_contents(html)

    # Create a list of lists (each length 8)...corresponds to 8 columns in html pbp
    td = [soup[i:i + 8] for i in range(0, len(soup), 8)]

    cleaned_html = [strip_html_pbp(x,json) for x in td]

    return cleaned_html

def get_html_roster(html,json,teams):
    #Given raw html and teams, return available roster data
    events = clean_html_pbp(html,json)

    #Roster dict
    roster_dict = {teams['away']:{},
                   teams['home']:{}}
    
    for event in events:
        if event[0] == "#":
            continue
        else:
            #Players are keys, value is a list with number, position, and description id
            for i in range(len(event[6])):
                player = event[6][i][0]
                num = event[6][i][1]
                pos = event[6][i][2]
                id = event[6][i][3]
                team = teams['away']
                
                #Accounting for players with three or more parts in their name
                if len(player.split())>2:
                    last = " ".join(player.split()[1:len(player.split())])
                else: 
                    last = player.split()[len(player.split())-1]

                desc_id = f'#{num} {last}'
                roster_dict[team].update({
                   desc_id:[num,pos,player,team,id]
                })      
            for i in range(len(event[7])):
                player = event[7][i][0]
                num = event[7][i][1]
                pos = event[7][i][2]
                id = event[7][i][3]
                team = teams['home']
                
                #Accounting for players with three or more parts in their name
                if len(player.split())>2:
                    last = " ".join(player.split()[1:len(player.split())])
                else: 
                    last = player.split()[len(player.split())-1]

                desc_id = f'#{num} {last}'
                roster_dict[team].update({
                   desc_id:[num,pos,player,team,id]
                })   

    return roster_dict

def get_json_coaches(game_id):
    #Given game id, return head coaches for away and home team
    
    #Retreive data
    json = rs.get(f'https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail').json()
    data = json['gameInfo']

    #Add coaches
    try:
        away = data['awayTeam']['headCoach']['default'].upper()
        home = data['homeTeam']['headCoach']['default'].upper()
        
        coaches = {'away':away,
                'home':home}
    except KeyError:
        return {}

    #Return: dict with coaches
    return coaches

def parse_html_event(event,roster,teams):
    #Given event from html events list and game roster, return event data

    events_dict = dict()
    if event[0] == "#" or event[4] in ['GOFF', 'EGT', 'PGSTR', 'PGEND', 'ANTHEM','SPC','PBOX']:
        return pd.DataFrame()
    else:
        #Event info
        events_dict['event_num'] = int(event[0])
        events_dict['period'] = int(event[1])
        events_dict['strength'] = re.sub(u'\xa0'," ",event[2])
        events_dict['period_time_elapsed'] = event[3]
        events_dict['seconds_elapsed'] = convert_to_seconds(event[3]) + (1200*(int(event[1])-1))
        events_dict['event_type'] = event[4]
        desc = re.sub(u'\xa0'," ",event[5])
        events_dict['description'] = desc

        events_dict['shot_type'] = desc.split(",")[1].lower().strip(" ") if event[4] in ['BLOCK','MISS','SHOT','GOAL'] else ""
        zone = [x for x in desc.split(',') if 'Zone' in x]
        if not zone:
            events_dict['zone_code'] = None
        elif zone[0].find("Off") != -1:
            events_dict['zone_code'] = 'O'
        elif zone[0].find("Neu") != -1:
            events_dict['zone_code'] = 'N'
        elif zone[0].find("Def") != -1:
            events_dict['zone_code'] = 'D'

        #Convert team names for compatiblity
        replace = [('LAK',"L.A"),('NJD',"N.J"),('SJS',"S.J"),('TBL',"T.B")]
        for name, repl in replace:
            teams['away'] = teams['away'].replace(repl,name)
            teams['home'] = teams['home'].replace(repl,name)
            desc = desc.replace(repl,name)
        
        event_team = desc[0:3] if desc[0:3] in [teams['away'],teams['home']] else   ""
        events_dict['event_team_abbr'] = event_team

        
        events_dict['away_team_abbr'] = teams['away']
        events_dict['home_team_abbr'] = teams['home']
        event_skaters = []

        away_skaters = 0
        away_goalie = 0
        #Away on-ice
        for i in range(len(event[6])):
            player = event[6][i][0]
            num = event[6][i][1]
            pos = event[6][i][2]
            id = event[6][i][3]
            
            if pos == 'G':
                events_dict['away_goalie'] = player
                events_dict['away_goalie_id'] = id
                away_goalie += 1
            else:
                events_dict[f'away_on_{i+1}'] = player
                events_dict[f'away_on_{i+1}_id'] = id
                away_skaters += 1

        home_skaters = 0
        home_goalie = 0
        #Home on-ice
        for i in range(len(event[7])):
            player = event[7][i][0]
            num = event[7][i][1]
            pos = event[7][i][2]    
            id = event[7][i][3]
            
            if pos == 'G':
                events_dict['home_goalie'] = player
                events_dict['home_goalie_id'] = id
                home_goalie += 1
            else:
                events_dict[f'home_on_{i+1}'] = player
                events_dict[f'home_on_{i+1}_id'] = id
                home_skaters += 1

        #Determine parsing route based on event (single player events are left)
        if event[4] in ['FAC','HIT','BLOCK','PENL']:
            #Regex to find team and player number involved (finds all for each event)
            #Code is modified from Harry Shomer in order to account for periods in a team abbreviation
            regex = re.compile(r'([A-Z]{2,3}|\b[A-Z]\.[A-Z])\s+#(\d+)')
            fac = regex.findall(desc)

            try: team_1,num_1 = fac[0]
            except: team_1 = ''
            try: team_2,num_2 = fac[1]
            except: team_2 = ''
            
            try: rost_1 = roster[team_1]
            except: rost_1 = {}
            try: rost_2 = roster[team_2]
            except: rost_2 = {}

            #Filter incorrectly parsed teams
            repl = []
            for team, num in fac:
                if team in [teams['home'],teams['away']]:
                    repl.append((team,num))

            fac = repl
                
            #Determine append order (really only applies to faceoffs)
            if len(fac) == 0:
                #No data
                ""
            else:
                if len(fac) == 1:
                    #Find event players using given roster
                    for desc_id,info in rost_1.items():
                        if desc_id in desc:
                            event_skaters.append([info[2],info[1],info[4]])
                else:
                    if team_1 == event_team:
                        for desc_id,info in rost_1.items():
                            if desc_id in desc:
                                event_skaters.append([info[2],info[1],info[4]])
                        for desc_id,info in rost_2.items():
                            if desc_id in desc:
                                event_skaters.append([info[2],info[1],info[4]])
                    else: 
                        for desc_id,info in rost_2.items():
                            if desc_id in desc:
                                event_skaters.append([info[2],info[1],info[4]])
                        for desc_id,info in rost_1.items():
                            if desc_id in desc:
                                event_skaters.append([info[2],info[1],info[4]])
        else:
            #Parse goal
            if event[4] == 'GOAL':
                regex = re.compile(r'#(\d+)\s+')
                goal = regex.findall(desc)

                goal_team = roster[event_team]
                #Search through individual element in goal (adds skaters in order from goal, first assist, second assist)
                for point in goal:
                    for info in goal_team.values():
                        if info[0] == point:
                            event_skaters.append([info[2],info[1],info[4]])
                            break
            else:
                #Parse single player or no player events
                combined = roster[teams['away']] | roster[teams['home']]
                for desc_id,info in combined.items():
                    if desc_id in desc:
                        event_skaters.append([info[2],info[1],info[4]])

        for i in range(len(event_skaters)):
            events_dict[f'event_player_{i+1}_name'] = event_skaters[i][0]
            events_dict[f'event_player_{i+1}_id'] = event_skaters[i][2]
            events_dict[f'event_player_{i+1}_pos'] = event_skaters[i][1]

        events_dict['away_skaters'] = away_skaters
        events_dict['home_skaters'] = home_skaters
        events_dict['away_goalie_in'] = away_goalie
        events_dict['home_goalie_in'] = home_goalie

        event_skaters = away_skaters if teams['away'] == event_team else home_skaters
        event_skaters_against = away_skaters if teams['home'] == event_team else home_skaters
        events_dict['strength_state'] = f'{event_skaters}v{event_skaters_against}'
        events_dict['event_skaters'] = np.where(event_team == teams['home'],home_skaters,away_skaters)

    #Return: dataframe of event in a single row
    return (pd.DataFrame([events_dict]))

def parse_html(game_id,html,json):
    #Given the game id, raw html document to a provided game, and json data, return parsed HTML play-by-play

    #Retreive cleaned html data (from Harry Shomer's hockey_scraper package) 
    events = clean_html_pbp(html,json)
    
    json_info = pd.json_normalize(json)
    teams = {
        'away':json_info['awayTeam.abbrev'][0],
        'home':json_info['homeTeam.abbrev'][0]
    }

    roster = get_html_roster(html,json,teams)
    event_log = []
    for event in events:
        event_log.append(parse_html_event(event,roster,teams))

    data = pd.concat(event_log)
    data['event_type'] = data['event_type'].replace({
     "PGSTR": "pre-game-start",
     "PGEND": "pre-game-end",
     'GSTR':"game-start",
     "ANTHEM":"anthem",
     "PSTR":"period-start",
     'FAC':"faceoff",
     "SHOT":"shot-on-goal",
     "BLOCK":"blocked-shot",
     "STOP":"stoppage",
     "MISS":"missed-shot",
     "HIT":"hit",
     "GOAL":"goal",
     "GIVE":"giveaway",
     "TAKE":"takeaway",
     "DELPEN":"delayed-penalty",
     "PENL":"penalty",
     "CHL":"challenge",
     "PEND":"period-end",
     "GEND":"game-end"
    })

    check_col = ['event_player_1_id','event_player_2_id','event_player_3_id',
                 'away_on_1','away_on_2','away_on_3','away_on_4','away_on_5','away_on_6',
                 'away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id',
                 'home_on_1','home_on_2','home_on_3','home_on_4','home_on_5','home_on_6',
                 'home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id']

    for col in check_col:
        try: data[col]
        except:
            data[col] = ""

    #Return: HTML play-by-play
    return data

### JSON PBP DATA ###
def parse_json(json):
    #Given json data from an NHL API call, return play-by-play data.

    events = pd.json_normalize(json['plays']).reset_index(drop=True)
    info = pd.json_normalize(json)
    roster =get_game_roster(json)

    #Return error if game is set in the future
    if info['gameState'][0] == 'FUT':
        raise ValueError(f"Game {info['id'][0]} has not occured yet.")

    away = info['awayTeam.id'][0]
    home = info['homeTeam.id'][0]
    teams = {
        away:info['awayTeam.abbrev'][0],
        home:info['homeTeam.abbrev'][0]
    }

    #Create player information dicts used to create event_player columns
    players = {}
    for id, player in zip(list(roster['playerId']),list(roster['fullName.default'])):
        players.update({id:player.upper()})

    #Test columns
    cols = ['eventId', 'timeInPeriod', 'timeRemaining', 'situationCode', 'homeTeamDefendingSide', 'typeCode', 'typeDescKey', 'sortOrder', 'periodDescriptor.number', 'periodDescriptor.periodType', 'periodDescriptor.maxRegulationPeriods', 'details.eventOwnerTeamId', 'details.losingPlayerId', 'details.winningPlayerId', 'details.xCoord', 'details.yCoord', 'details.zoneCode', 'pptReplayUrl', 'details.shotType', 'details.scoringPlayerId', 'details.scoringPlayerTotal', 'details.assist1PlayerId', 'details.assist1PlayerTotal', 'details.assist2PlayerId', 'details.assist2PlayerTotal', 'details.goalieInNetId', 'details.awayScore', 'details.homeScore', 'details.highlightClipSharingUrl', 'details.highlightClipSharingUrlFr', 'details.highlightClip', 'details.highlightClipFr', 'details.discreteClip', 'details.discreteClipFr', 'details.shootingPlayerId', 'details.awaySOG', 'details.homeSOG', 'details.playerId', 'details.hittingPlayerId', 'details.hitteePlayerId', 'details.reason', 'details.typeCode', 'details.descKey', 'details.duration', 'details.servedByPlayerId', 'details.secondaryReason', 'details.blockingPlayerId', 'details.committedByPlayerId', 'details.drawnByPlayerId', 'game_id', 'season', 'season_type', 'game_date']

    for col in cols:
        try:events[col]
        except:
            events[col]=""

    #Event_player_columns include players in a given set of events; the higher the number, the greater the importance the event player was to the play
    events['event_player_1_id'] = events['details.winningPlayerId'].combine_first(events['details.scoringPlayerId'])\
                                                                   .combine_first(events['details.shootingPlayerId'])\
                                                                   .combine_first(events['details.playerId'])\
                                                                   .combine_first(events['details.hittingPlayerId'])\
                                                                   .combine_first(events['details.committedByPlayerId'])
        
    events['event_player_2_id'] = events['details.losingPlayerId'].combine_first(events['details.assist1PlayerId'])\
                                                                    .combine_first(events['details.hitteePlayerId'])\
                                                                    .combine_first(events['details.drawnByPlayerId'])\
                                                                    .combine_first(events['details.blockingPlayerId'])

    events['event_player_3_id'] = events['details.assist2PlayerId']

    events['event_team_status'] = np.where(events['details.eventOwnerTeamId']==home,"home","away")

    #Coordinate adjustments:
    #The WSBA NHL Scraper includes three sets of coordinates per event:
    # x, y - Raw coordinates from JSON pbpp
    # x_fixed, y_fixed - Coordinates fixed to the right side of the ice (x is always greater than 0)
    # x_adj, y_adj - Adjusted coordinates configuring away events with negative x vlaues while home events are always positive
    
    #Some games (mostly preseason and all star games) do not include coordinates.  
    try:
        events['x_fixed'] = abs(events['details.xCoord'])
        events['y_fixed'] = np.where(events['details.xCoord']<0,-events['details.yCoord'],events['details.yCoord'])
        events['x_adj'] = np.where(events['event_team_status']=="home",events['x_fixed'],-events['x_fixed'])
        events['y_adj'] = np.where(events['event_team_status']=="home",events['y_fixed'],-events['y_fixed'])
        events['event_distance'] = np.sqrt(((89 - events['x_fixed'])**2) + (events['y_fixed']**2))
        events['event_angle'] = np.degrees(np.arctan2(abs(events['y_fixed']), abs(89 - events['x_fixed'])))
    except TypeError:
        print(f"No coordinates found for game {info['id'][0]}...")
    
        events['x_fixed'] = np.nan
        events['y_fixed'] = np.nan
        events['x_adj'] = np.nan
        events['y_adj'] = np.nan
        events['event_distance'] = np.nan
        events['event_angle'] = np.nan
    
    
    events['event_team_abbr'] = events['details.eventOwnerTeamId'].replace(teams)
    events['event_goalie'] = events['details.goalieInNetId'].replace(players)

    #Rename columns to follow WSBA naming conventions
    events = events.rename(columns={
        "eventId":"event_id",
        "periodDescriptor.number":"period",
        "periodDescriptor.periodType":"period_type",
        "timeInPeriod":"period_time_elasped",
        "timeRemaining":"period_time_remaining",
        "situationCode":"situation_code",
        "homeTeamDefendingSide":"home_team_defending_side",
        "typeCode":"event_type_code",
        "typeDescKey":"event_type",
        "details.shotType":"shot_type",
        "details.duration":"penalty_duration",
        "details.descKey":"penalty_description",
        "details.reason":"reason",
        "details.zoneCode":"zone_code",
        "details.xCoord":"x",
        "details.yCoord":"y",
        "details.goalieInNetId": "event_goalie_id",
        "details.awaySOG":"away_SOG",
        "details.homeSOG":"home_SOG"
    })

    #Period time adjustments (only 'seconds_elapsed' is included in the resulting data)
    events['period_time_simple'] = events['period_time_elasped'].str.replace(":","",regex=True)
    events['period_seconds_elapsed'] = np.where(events['period_time_simple'].str.len()==3,
                                           ((events['period_time_simple'].str[0].astype(int)*60)+events['period_time_simple'].str[-2:].astype(int)),
                                           ((events['period_time_simple'].str[0:2].astype(int)*60)+events['period_time_simple'].str[-2:].astype(int)))
    events['seconds_elapsed'] = ((events['period']-1)*1200)+events['period_seconds_elapsed']

    events = events.loc[(events['event_type']!="")]
    
    #Assign score and fenwick for each event
    fenwick_events = ['missed-shot','shot-on-goal','goal']
    ag = 0
    ags = []
    hg = 0
    hgs = []

    af = 0
    afs = []
    hf = 0
    hfs = []
    for event,team in zip(list(events['event_type']),list(events['event_team_status'])):
        if event in fenwick_events:
            if team == "home":
                hf += 1
                if event == 'goal':
                    hg += 1
            else:
                af += 1
                if event == 'goal':
                    ag += 1
       
        ags.append(ag)
        hgs.append(hg)
        afs.append(af)
        hfs.append(hf)

    events['away_score'] = ags
    events['home_score'] = hgs
    events['away_fenwick'] = afs
    events['home_fenwick'] = hfs

    #Return: dataframe with parsed game
    return events

def combine_pbp(game_id,html,json):
    #Given game id, html data, and json data, return complete play-by-play data for provided game

    html_pbp = parse_html(game_id,html,json)
    info = get_game_info(json)

    #Route data combining - json if season is after 2009-2010:
    if str(info['season']) in ['20052006','20062007','20072008','20082009','20092010']:
        #ESPN x HTML
        espn_pbp = parse_espn(str(info['game_date']),info['away_team_abbr'],info['home_team_abbr']).rename(columns={'coords_x':'x',"coords_y":'y'})
        merge_col = ['period','seconds_elapsed','event_type','event_team_abbr']

        df = pd.merge(html_pbp,espn_pbp,how='left',on=merge_col)

    else:
        #JSON x HTML
        json_pbp = parse_json(json)
        #Modify merge conditions and merge pbps
        merge_col = ['period','seconds_elapsed','event_type','event_team_abbr','event_player_1_id']
        html_pbp = html_pbp.drop(columns=['event_player_2_id','event_player_3_id','shot_type','zone_code'])

        df = pd.merge(html_pbp,json_pbp,how='left',on=merge_col)

    #Add game info
    info_col = ['season','season_type','game_id','game_date',"start_time","venue","venue_location",
        'away_team_abbr','home_team_abbr']
    
    for col in info_col:
        df[col] = info[col]

    #Fill period_type column and assign shifts a sub-500 event code
    df['period_type'] = np.where(df['period']<4,"REG",np.where(np.logical_and(df['period']==5,df['season_type']==2),"SO","OT"))
    try: df['event_type_code'] = np.where(df['event_type']!='change',df['event_type_code'],499)
    except:
        ""
    df = df.sort_values(['period','seconds_elapsed']).reset_index()

    df['event_team_status'] = np.where(df['event_team_abbr'].isna(),"",np.where(df['home_team_abbr']==df['event_team_abbr'],"home","away"))

    #Retrieve coaches
    coaches = get_json_coaches(game_id)
    if not coaches:
        df['away_coach'] = ""
        df['home_coach'] = ""
        df['event_coach'] = ""
    else:
        df['away_coach'] = coaches['away']
        df['home_coach'] = coaches['home']
        df['event_coach'] = np.where(df['event_team_abbr']==df['home_team_abbr'],coaches['home'],np.where(df['event_team_abbr']==df['away_team_abbr'],coaches['away'],""))
        
    col = [col for col in get_col() if col in df.columns.to_list()]
    #Return: complete play-by-play information for provided game
    return df[col]

### ESPN SCRAPING FUNCTIONS ###
def espn_game_id(date,away,home):
    #Given a date formatted as YYYY-MM-DD and teams, return game id from ESPN schedule
    date = date.replace("-","")

    #Retreive data
    api = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={date}"
    schedule = pd.json_normalize(rs.get(api).json()['events'])

    #Create team abbreviation columns
    schedule['away_team_abbr'] = schedule['shortName'].str[:3].str.strip(" ")
    schedule['home_team_abbr'] = schedule['shortName'].str[-3:].str.strip(" ")
    
    #Modify team abbreviations as necessary
    schedule = schedule.replace({
        "LA":"LAK",
        "NJ":"NJD",
        "SJ":"SJS",
        "TB":"TBL",
    })

    #Retreive game id
    game_id = schedule.loc[(schedule['away_team_abbr']==away)&
                           (schedule['home_team_abbr']==home),'id'].tolist()[0]

    #Return: ESPN game id
    return game_id

def parse_espn(date,away,home):
    #Given a date formatted as YYYY-MM-DD and teams, return game events
    game_id = espn_game_id(date,away,home)
    url = f'https://www.espn.com/nhl/playbyplay/_/gameId/{game_id}'
    
    #Code modified from Patrick Bacon

    #Retreive game events as json
    page = rs.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout = 500)
    soup = BeautifulSoup(page.content.decode('ISO-8859-1'), 'lxml', multi_valued_attributes = None)
    json = json_lib.loads(str(soup).split('"playGrps":')[1].split(',"tms"')[0])

    #DataFrame of time-related info for events
    clock_df = pd.DataFrame()

    for period in range(0, len(json)):
        clock_df = clock_df._append(pd.DataFrame(json[period]))

    clock_df = clock_df[~pd.isna(clock_df.clock)]

    # Needed to add .split(',"st":3')[0] for playoffs

    #DataFrame of coordinates for events
    coords_df = pd.DataFrame(json_lib.loads(str(soup).split('plays":')[1].split(',"st":1')[0].split(',"st":2')[0].split(',"st":3')[0]))

    clock_df = clock_df.assign(
        clock = clock_df.clock.apply(lambda x: x['displayValue'])
    )

    coords_df = coords_df.assign(
        coords_x = coords_df[~pd.isna(coords_df.coordinate)].coordinate.apply(lambda x: x['x']).astype(int),
        coords_y = coords_df[~pd.isna(coords_df.coordinate)].coordinate.apply(lambda y: y['y']).astype(int),
        event_player_1_name = coords_df[~pd.isna(coords_df.athlete)]['athlete'].apply(lambda x: x['name'])
    )

    #Combine
    espn_events = coords_df.merge(clock_df.loc[:, ['id', 'clock']])

    espn_events = espn_events.assign(
        period = espn_events['period'].apply(lambda x: x['number']),
        minutes = espn_events['clock'].str.split(':').apply(lambda x: x[0]).astype(int),
        seconds = espn_events['clock'].str.split(':').apply(lambda x: x[1]).astype(int),
        event_type = espn_events['type'].apply(lambda x: x['txt'])
    )

    espn_events = espn_events.assign(coords_x = np.where((pd.isna(espn_events.coords_x)) & (pd.isna(espn_events.coords_y)) &
                (espn_events.event_type=='Face Off'), 0, espn_events.coords_x
    ),
                      coords_y = np.where((pd.isna(espn_events.coords_x)) & (pd.isna(espn_events.coords_y)) &
                (espn_events.event_type=='Face Off'), 0, espn_events.coords_y))

    espn_events = espn_events[(~pd.isna(espn_events.coords_x)) & (~pd.isna(espn_events.coords_y)) & (~pd.isna(espn_events.event_player_1_name))]

    espn_events = espn_events.assign(
        coords_x = espn_events.coords_x.astype(int),
        coords_y = espn_events.coords_y.astype(int)
    )
    
    #Rename events
    #The turnover event includes just one player in the event information, meaning takeaways will have no coordinates for play-by-plays created by ESPN scraping
    espn_events['event_type'] = espn_events['event_type'].replace({
        "Face Off":'faceoff',
        "Hit":'hit',
        "Shot":'shot-on-goal',
        "Missed":'missed-shot',
        "Blocked":'blocked-shot',
        "Goal":'goal',
        "Turnover":'giveaway',
        "Delayed Penalty":'delayed-penalty',
        "Penalty":'penalty',
    })

    #Period time adjustments (only 'seconds_elapsed' is included in the resulting data)
    espn_events['period_time_simple'] = espn_events['clock'].str.replace(":","",regex=True)
    espn_events['period_seconds_elapsed'] = np.where(espn_events['period_time_simple'].str.len()==3,
                                            ((espn_events['period_time_simple'].str[0].astype(int)*60)+espn_events['period_time_simple'].str[-2:].astype(int)),
                                            ((espn_events['period_time_simple'].str[0:2].astype(int)*60)+espn_events['period_time_simple'].str[-2:].astype(int)))
    espn_events['seconds_elapsed'] = ((espn_events['period']-1)*1200)+espn_events['period_seconds_elapsed']

    espn_events = espn_events.rename(columns = {'text':'description'})

    #Add event team
    espn_events['event_team_abbr'] = espn_events['homeAway'].replace({
        "away":away,
        "home":home
    })

    #Some games (mostly preseason and all star games) do not include coordinates.  
    try:
        espn_events['x_fixed'] = abs(espn_events['coords_x'])
        espn_events['y_fixed'] = np.where(espn_events['coords_x']<0,-espn_events['coords_y'],espn_events['coords_y'])
        espn_events['x_adj'] = np.where(espn_events['homeAway']=="home",espn_events['x_fixed'],-espn_events['x_fixed'])
        espn_events['y_adj'] = np.where(espn_events['homeAway']=="home",espn_events['y_fixed'],-espn_events['y_fixed'])
        espn_events['event_distance'] = np.sqrt(((89 - espn_events['x_fixed'])**2) + (espn_events['y_fixed']**2))
        espn_events['event_angle'] = np.degrees(np.arctan2(abs(espn_events['y_fixed']), abs(89 - espn_events['x_fixed'])))
    except TypeError:
        print(f"No coordinates found for ESPN game...")
    
        espn_events['x_fixed'] = np.nan
        espn_events['y_fixed'] = np.nan
        espn_events['x_adj'] = np.nan
        espn_events['y_adj'] = np.nan
        espn_events['event_distance'] = np.nan
        espn_events['event_angle'] = np.nan

    #Assign score and fenwick for each event
    fenwick_events = ['missed-shot','shot-on-goal','goal']
    ag = 0
    ags = []
    hg = 0
    hgs = []

    af = 0
    afs = []
    hf = 0
    hfs = []
    for event,team in zip(list(espn_events['event_type']),list(espn_events['homeAway'])):
        if event in fenwick_events:
            if team == "home":
                hf += 1
                if event == 'goal':
                    hg += 1
            else:
                af += 1
                if event == 'goal':
                    ag += 1
       
        ags.append(ag)
        hgs.append(hg)
        afs.append(af)
        hfs.append(hf)

    espn_events['away_score'] = ags
    espn_events['home_score'] = hgs
    espn_events['away_fenwick'] = afs
    espn_events['home_fenwick'] = hfs
    #Return: play-by-play events in supplied game from ESPN
    return espn_events

### SHIFT SCRAPING FUNCTIONS ###
def analyze_shifts(shift, id, name, pos, team):
    #Collects teams in given shifts html (parsed by Beautiful Soup)
    #Modified version of Harry Shomer's analyze_shifts function in the hockey_scraper package
    shifts = dict()

    shifts['player_name'] = name.upper()
    shifts['player_id'] = id
    shifts['player_pos'] = pos
    shifts['period'] = '4' if shift[1] == 'OT' else '5' if shift[1] == 'SO' else shift[1]
    shifts['event_team_abbr'] = get_team(team.strip(' '))
    shifts['start'] = convert_to_seconds(shift[2].split('/')[0])
    shifts['duration'] = convert_to_seconds(shift[4].split('/')[0])

    # I've had problems with this one...if there are no digits the time is fucked up
    if re.compile(r'\d+').findall(shift[3].split('/')[0]):
        shifts['end'] = convert_to_seconds(shift[3].split('/')[0])
    else:
        shifts['end'] = shifts['start'] + shifts['duration']
    return shifts

def parse_shifts_html(html,json):
    #Parsing of shifts data for a single team in a provided game
    #Modified version of Harry Shomer's parse_shifts function in the hockey_scraper package

    #JSON Prep
    info = get_game_info(json)
    roster = info['rosters']

    away = info['away_team_id'] #Away team in the HTML is the seventh column 
    home = info['home_team_id'] #Home team in the HTML is the eighth column
    away_players = roster.loc[roster['teamId']==away][['playerId','fullName.default','positionCode','sweaterNumber']]
    home_players = roster.loc[roster['teamId']==home][['playerId','fullName.default','positionCode','sweaterNumber']]
    
    #Create roster dict
    rosters = {"away":away_players.set_index("sweaterNumber")[['playerId','fullName.default','positionCode']].to_dict(),
               "home":home_players.set_index("sweaterNumber")[['playerId','fullName.default','positionCode']].to_dict()
               }
    
    all_shifts = []
    #columns = ['game_id', 'player_name', 'player_id', 'period', 'team_abbr', 'start', 'end', 'duration']
    td, teams = get_soup(html)

    team = teams[0]
    home_team = teams[1]
    players = dict()
    status = 'home' if team == home_team else 'away'

    # Iterates through each player shifts table with the following data:
    # Shift #, Period, Start, End, and Duration.
    for t in td:
        t = t.get_text()
        if ',' in t:     # If a comma exists it is a player
            name = t
            
            name = name.split(',')
            number = int(name[0][:2].strip())
            id = rosters[status]['playerId'][number]
            players[id] = dict()

            #HTML shift functions assess one team at a time, which simplifies the lookup process with number to name and id
            
            players[id]['name'] = rosters[status]['fullName.default'][number]
            players[id]['pos'] = rosters[status]['positionCode'][number]

            players[id]['shifts'] = []
        else:
            players[id]['shifts'].extend([t])

    for key in players.keys():
        # Create lists of shifts-table columns for analysis
        players[key]['shifts'] = [players[key]['shifts'][i:i + 5] for i in range(0, len(players[key]['shifts']), 5)]

        name = players[key]['name']
        pos = players[key]['pos']

        # Parsing
        shifts = [analyze_shifts(shift, key, name, pos, team) for shift in players[key]['shifts']]
        all_shifts.extend(shifts)

    df = pd.DataFrame(all_shifts)

    shifts_raw = df[df['duration'] > 0]

    #Return: single-team individual shifts by player
    return shifts_raw

def parse_shift_events(html,json,home):
    #Given shift document and home team conditional, parse and convert document to shift events congruent to html play-by-play
    shift = parse_shifts_html(html,json)
    rosters = get_game_roster(json)

    # Identify shift starts for each shift event
    shifts_on = shift.groupby(['event_team_abbr', 'period', 'start']).agg(
        num_on=('player_name', 'size'),
        players_on=('player_name', lambda x: ', '.join(x)),
        ids_on=('player_id', lambda x: ', '.join(map(str,x))),
    ).reset_index()

    shifts_on = shifts_on.rename(columns={
        'start':"seconds_elapsed"
    })

    # Identify shift stops for each shift event
    shifts_off = shift.groupby(['event_team_abbr', 'period', 'end']).agg(
        num_off=('player_name', 'size'),
        players_off=('player_name', lambda x: ', '.join(x)),
        ids_off=('player_id', lambda x: ', '.join(map(str,x))),
    ).reset_index()

    shifts_off = shifts_off.rename(columns={
        'end':"seconds_elapsed"
    })

    # Merge and sort by time in game
    shifts = pd.merge(shifts_on, shifts_off, on=['event_team_abbr', 'period', 'seconds_elapsed'], how='outer')
    
    shifts['seconds_elapsed'] = shifts['seconds_elapsed'] + (1200*(shifts['period'].astype(int)-1))
    shifts['event_type'] = 'change'

    #Shift events similar to html (remove shootout shifts)
    shifts = shifts.loc[shifts['period'].astype(int)<5].sort_values(['period','seconds_elapsed'])

    #Generate on-ice columns
    skater_names = list(rosters.loc[rosters['positionCode']!="G",'playerId'].astype(str))
    goalie_names = list(rosters.loc[rosters['positionCode']=="G",'playerId'].astype(str))
    team = list(shift['event_team_abbr'])[0]

    skaters = pd.DataFrame()
    goalies = pd.DataFrame()
    for player in skater_names:
        #For each player in the game, determine when they began and ended shifts.  
        #With player names as columns, 1 represents a shift event a player was on the ice for while 0 represents off the ice
        on_ice = (np.cumsum(
            shifts.loc[(shifts['event_team_abbr'] == team), 'ids_on']
            .apply(str)
            .apply(lambda x: int(bool(re.search(player, x)))) -
            shifts.loc[(shifts['event_team_abbr'] == team), 'ids_off']
            .apply(str)
            .apply(lambda x: int(bool(re.search(player, x))))
        ))
        skaters[player] = on_ice
    
    skaters = skaters.fillna(0).astype(int)

    on_skaters = (skaters == 1).stack().reset_index()
    on_skaters = on_skaters[on_skaters[0]].groupby("level_0")["level_1"].apply(list).reset_index()
    
    max_players = 6
    for i in range(max_players):
        on_skaters[f"{'home' if home else 'away'}_on_{i+1}_id"] = on_skaters["level_1"].apply(lambda x: x[i] if i < len(x) else " ")
    
    on_skaters = on_skaters.drop(columns=["level_1"]).rename(columns={"level_0": "row"})
    
    #Repeat this process with goaltenders
    for player in goalie_names:
        on_ice = (np.cumsum(
            shifts.loc[(shifts['event_team_abbr'] == team), 'ids_on']
            .apply(str)
            .apply(lambda x: int(bool(re.search(player, x)))) -
            shifts.loc[(shifts['event_team_abbr'] == team), 'ids_off']
            .apply(str)
            .apply(lambda x: int(bool(re.search(player, x))))
        ))
        goalies[player] = on_ice
    
    goalies = goalies.fillna(0).astype(int)
    
    on_goalies = (goalies == 1).stack().reset_index()
    on_goalies = on_goalies[on_goalies[0]].groupby("level_0")["level_1"].apply(list).reset_index()
    
    max_players = 1
    for i in range(max_players):
        on_goalies[f"{'home' if home else 'away'}_goalie_id"] = on_goalies["level_1"].apply(lambda x: x[i] if i < len(x) else " ")
    
    on_goalies = on_goalies.drop(columns=["level_1"]).rename(columns={"level_0": "row"})
    
    #combine on-ice skaters and goaltenders for each shift event
    on_players = pd.merge(on_skaters,on_goalies,how='outer',on=['row'])

    shifts['row'] = shifts.index

    if home:
        shifts['home_team_abbr'] = team
    else:
        shifts['away_team_abbr'] = team
    #Return: shift events with newly added on-ice columns.  NAN values are replaced with string "REMOVE" as means to create proper on-ice columns for json pbp
    return pd.merge(shifts,on_players,how="outer",on=['row']).replace(np.nan,"")

def combine_shifts(away_html,home_html,json):
    #JSON Prep
    info = get_game_info(json)
    del info['rosters']

    roster = get_game_roster(json)
    #Quickly combine shifts data
    away = parse_shift_events(away_html,json,False)
    home = parse_shift_events(home_html,json,True)

    #Combine shifts
    data = pd.concat([away,home]).sort_values(['period','seconds_elapsed'])

    #Create info columns
    for col in info.keys():
        data[col] = info[col]

        #Create player information dicts to create on-ice names
    players = {}
    for id, player in zip(list(roster['playerId']),list(roster['fullName.default'])):
        players.update({str(id):player.upper()})

    for i in range(0,7):
        if i == 6:
            data['away_goalie'] = data['away_goalie_id'].replace(players)
            data['home_goalie'] = data['home_goalie_id'].replace(players)
        else:
            data[f'away_on_{i+1}'] = data[f'away_on_{i+1}_id'].replace(players)
            data[f'home_on_{i+1}'] = data[f'home_on_{i+1}_id'].replace(players)

    data = data.sort_values(['period','seconds_elapsed'])
    #Fill on-ice columns down
    on_ice_col = ['away_on_1','away_on_2','away_on_3','away_on_4','away_on_5','away_on_6',
                'away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id',
                'home_on_1','home_on_2','home_on_3','home_on_4','home_on_5','home_on_6',
                'home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id',
                'away_goalie','home_goalie','away_goalie_id','home_goalie_id']

    for col in on_ice_col:
        data[col] = data[col].ffill()

    #Create strength state information
    away_on = ['away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id',]
    home_on = ['home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id',]
    data['away_skaters'] = data[away_on].replace(r'^\s*$', np.nan, regex=True).notna().sum(axis=1)
    data['home_skaters'] = data[home_on].replace(r'^\s*$', np.nan, regex=True).notna().sum(axis=1)
    data['strength_state'] = np.where(data['event_team_abbr']==data['away_team_abbr'],data['away_skaters'].astype(str)+"v"+data['home_skaters'].astype(str),data['home_skaters'].astype(str)+"v"+data['away_skaters'].astype(str))

    #Return: full shifts data converted to play-by-play format
    col = [col for col in get_col() if col in data.columns.to_list()]
    return data[col]

### FINALIZE PBP ###
def get_col():
    return [
        'season','season_type','game_id','game_date',"start_time","venue","venue_location",
        'away_team_abbr','home_team_abbr','event_num','period','period_type',
        'seconds_elapsed',"situation_code","strength_state","home_team_defending_side",
        "event_type_code","event_type","description","penalty_duration",
        "event_team_abbr",'num_on', 'players_on','ids_on','num_off','players_off','ids_off','shift_type',
        "event_team_status",
        "event_player_1_name","event_player_2_name","event_player_3_name",
        "event_player_1_id","event_player_2_id","event_player_3_id",
        "event_player_1_pos","event_player_2_pos","event_player_3_pos",
        "event_goalie","event_goalie_id",
        "shot_type","zone_code","x","y","x_fixed","y_fixed","x_adj","y_adj",
        "event_skaters","away_skaters","home_skaters",
        "event_distance","event_angle","away_score","home_score", "away_fenwick", "home_fenwick",
        "away_on_1","away_on_2","away_on_3","away_on_4","away_on_5","away_on_6","away_goalie",
        "home_on_1","home_on_2","home_on_3","home_on_4","home_on_5","home_on_6","home_goalie",
        "away_on_1_id","away_on_2_id","away_on_3_id","away_on_4_id","away_on_5_id","away_on_6_id","away_goalie_id",
        "home_on_1_id","home_on_2_id","home_on_3_id","home_on_4_id","home_on_5_id","home_on_6_id","home_goalie_id",
        "event_coach","away_coach","home_coach"
    ]

def combine_data(game_id,html_pbp,away_shifts,home_shifts,json):
    #Given game_id, html_pbp, away and home shifts, and json pbp, return total game play-by-play data is provided with additional and corrected details
    #Create dfs
    pbp = combine_pbp(game_id,html_pbp,json)
    shifts = combine_shifts(away_shifts,home_shifts,json)

    #Combine data    
    df = pd.concat([pbp,shifts])

    #Create priority columns designed to order events that occur at the same time in a game
    even_pri = ['takeaway','giveaway','missed-shot','hit','shot-on-goal','blocked-shot']
    df['priority'] = np.where(df['event_type'].isin(even_pri),1,
                              np.where(df['event_type']=='goal',2,
                              np.where(df['event_type']=='stoppage',3,
                              np.where(df['event_type']=='delayed-penalty',4,
                              np.where(df['event_type']=='penalty',5,
                              np.where(df['event_type']=='period-end',6,
                              np.where(df['event_type']=='change',7,
                              np.where(df['event_type']=='game-end',8,
                              np.where(df['event_type']=='period-start',9,
                              np.where(df['event_type']=='faceoff',10,0))))))))))
                              
    df[['period','seconds_elapsed']] =  df[['period','seconds_elapsed']].astype(int)
    df = df.sort_values(['period','seconds_elapsed','priority'])
    
    #Recalibrate event_num column to accurately depict the order of all events, including changes
    df.reset_index(inplace=True,drop=True)
    df['event_num'] = df.index+1
    df['event_team_status'] = np.where(df['event_team_abbr'].isna(),"",np.where(df['home_team_abbr']==df['event_team_abbr'],"home","away"))
    df['event_type_last'] = df['event_type'].shift(1)
    df['event_type_last_2'] = df['event_type_last'].shift(1)
    df['event_type_next'] = df['event_type'].shift(-1)
    lag_events = ['stoppage','goal','period-end']
    lead_events = ['faceoff','period-end']
    period_end_secs = [0,1200,2400,3600,4800,6000,7200,8400,9600,10800]
    #Define shifts by "line-change" or "on-the-fly"
    df['shift_type'] = np.where(df['event_type']=='change',np.where(np.logical_or(np.logical_or(df['event_type_last'].isin(lag_events),df['event_type_last_2'].isin(lag_events),df['event_type_next'].isin(lead_events)),df['seconds_elapsed'].isin(period_end_secs)),"line-change","on-the-fly"),"")
    df['description'] = df['description'].combine_first(df['event_team_abbr']+" CHANGE: "+df['shift_type'])
    try:
        df['event_type_code'] = np.where(df['event_type']=='change',499,df['event_type_code'])
    except:
        ""

    #Forward fill as necessary
    cols = ['period_type','home_team_defending_side','away_score','away_fenwick','home_score','home_fenwick','away_coach','home_coach']
    for col in cols:
        try: df[col]
        except: df[col] = ""
        df[col] = df[col].ffill()

    #Return: complete play-by-play with all important data for each event in a provided game
    return df[[col for col in get_col() if col in df.columns.to_list()]].replace(r'^\s*$', np.nan, regex=True)