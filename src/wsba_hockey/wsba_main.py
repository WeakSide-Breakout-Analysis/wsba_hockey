import random
import os
import asyncio
import time
import requests as rs
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Union
from datetime import datetime, timedelta, date
from wsba_hockey.tools.scraping import *
from wsba_hockey.tools.xg_model import *
from wsba_hockey.tools.agg import *
from wsba_hockey.tools.plotting import *
from wsba_hockey.tools.columns import col_map

### WSBA HOCKEY ###
## Provided below are all integral functions in the WSBA Hockey Python package. ##

## GLOBAL VARIABLES ##
CONVERT_SEASONS = {2007: 20072008, 
                   2008: 20082009, 
                   2009: 20092010, 
                   2010: 20102011, 
                   2011: 20112012, 
                   2012: 20122013, 
                   2013: 20132014, 
                   2014: 20142015, 
                   2015: 20152016, 
                   2016: 20162017, 
                   2017: 20172018, 
                   2018: 20182019, 
                   2019: 20192020, 
                   2020: 20202021, 
                   2021: 20212022, 
                   2022: 20222023, 
                   2023: 20232024, 
                   2024: 20242025,
                   2025: 20252026}

SEASON_NAMES = {20072008: '2007-08', 
                20082009: '2008-09',
                20092010: '2009-10', 
                20102011: '2010-11',
                20112012: '2011-12', 
                20122013: '2012-13',
                20132014: '2013-14', 
                20142015: '2014-15',
                20152016: '2015-16', 
                20162017: '2016-17',
                20172018: '2017-18',
                20182019: '2018-19', 
                20192020: '2019-20',
                20202021: '2020-21', 
                20212022: '2021-22',
                20222023: '2022-23', 
                20232024: '2023-24',
                20242025: '2024-25',
                20252026: '2025-26'}

CONVERT_TEAM_ABBR = {'L.A':'LAK',
                     'N.J':'NJD',
                     'S.J':'SJS',
                     'T.B':'TBL',
                     'PHX':'ARI'}

PER_SIXTY = ['Fi','xGi','Gi','A1','A2','P1','P','Si','OZF','NZF','DZF','FF','FA','xGF','xGA','GF','GA','SF','SA','CF','CA','HF','HA','Give','Take','Penl','Penl2','Penl5','Draw','PIM','Block','GSAx']

#Some games in the API are specifically known to cause errors in scraping.
#This list is updated as frequently as necessary
KNOWN_PROBS = {
    2007020011:'Missing shifts data for game between Chicago and Minnesota.',
    2007021178:'Game between the Bruins and Sabres is missing data after the second period, for some reason.',
    2008020259:'HTML data is completely missing for this game.',
    2008020409:'HTML data is completely missing for this game.',
    2008021077:'HTML data is completely missing for this game.',
    2008030311:'Missing shifts data for game between Pittsburgh and Carolina',
    2009020081:'HTML pbp for this game between Pittsburgh and Carolina is missing all but the period start and first faceoff events, for some reason.',
    2009020658:'Missing shifts data for game between New York Islanders and Dallas.',
    2009020885:'Missing shifts data for game between Sharks and Blue Jackets.',
    2010020124:'Game between Capitals and Hurricanes is sporadically missing player on-ice data',
    2012020018:'HTML events contain mislabeled events.',
    2018021133:'Game between Lightning and Capitals has incorrectly labeled event teams (i.e. WSH TAKEAWAY - #71 CIRELLI (Cirelli is a Tampa Bay skater in this game)).',
}

SHOT_TYPES = ['wrist','deflected','tip-in','slap','backhand','snap','wrap-around','poke','bat','cradle','between-legs']

EVENTS = ['faceoff','hit','giveaway','takeaway','blocked-shot','missed-shot','shot-on-goal','goal','penalty']

DIR = os.path.dirname(os.path.realpath(__file__))
SCHEDULE_PATH = os.path.join(DIR,'tools\\schedule\\schedule.csv')
INFO_PATH = os.path.join(DIR,'tools\\teaminfo\\nhl_teaminfo.csv')
DEFAULT_ROSTER = os.path.join(DIR,'tools\\rosters\\nhl_rosters.csv')
GAME_SCORE = os.path.join(DIR,'tools\\game_score\\')

GS_SCORE_FEATURES = {
    'skater':["P",
            "PENL%",
            "PM%",
            "F%",
            "EV_xGC%",
            "EV_xGF",
            "EV_xGA",
            "PP_xGC%",
            "PP_xGF",
            "PP_xGA",
            "SH_xGC%",
            "SH_xGF",
            "SH_xGA"],

    'goalie':["xGF%",
            "GA/xGA"]
}

#Load column names for standardization
COL_MAP = col_map()

DRAFT_CAT = {
    0: 'All Prospects',
    1: 'North American Skaters',
    2: 'International Skater',
    3: 'North American Goalies',
    4: 'International Goalies'
}

## SCRAPE FUNCTIONS ##
def nhl_scrape_game(game_ids:int | list[int], split_shifts:bool = False, remove:list[str] = [], verbose:bool = False, sources:bool = False, errors:bool = False):
    """
    Given a set of game_ids (NHL API), return complete play-by-play information as requested.

    Args:
        game_ids (int or List[int] or ['random', int, int, int]):
            List of NHL game IDs to scrape or use ['random', n, start_year, end_year] to fetch n random games.
        split_shifts (bool, optional):
            If True, returns a dict with separate 'pbp' and 'shifts' DataFrames. Default is False.
        remove (List[str], optional):
            List of event types to remove from the result. Default is an empty list.
        verbose (bool, optional):
            If True, generates extra event features (such as those required to calculate xG). Default is False.
        sources (bool, optional):
            If True, saves raw HTML, JSON, SHIFTS, and single-game full play-by-play to a separate folder in the working directory. Default is False.
        errors (bool, optional):
            If True, includes a list of game IDs that failed to scrape in the return. Default is False.

    Returns:
        pd.DataFrame:
            If split_shifts is False, returns a single DataFrame of play-by-play data.
        dict[str, pd.DataFrame]:
            If split_shifts is True, returns a dictionary with keys:
            - 'pbp': play-by-play events
            - 'shifts': shift change events
            - 'errors' (optional): list of game IDs that failed if errors=True
    """
    
    #Wrap game_id in a list if only a single game_id is provided
    game_ids = [game_ids] if type(game_ids) != list else game_ids

    pbps = []
    if game_ids[0] == 'random':
        #Randomize selection of game_ids
        #Some ids returned may be invalid (for example, 2020022000)
        num = game_ids[1]
        start = game_ids[2] if len(game_ids) > 1 else 2007
        end = game_ids[3] if len(game_ids) > 2 else (date.today().year)-1

        game_ids = []
        i = 0
        print("Finding valid, random game ids...")
        while i is not num:
            print(f"\rGame IDs found in range {start}-{end}: {i}/{num}",end="")
            rand_year = random.randint(start,end)
            rand_season_type = random.randint(2,3)
            rand_game = random.randint(1,1312)

            #Ensure id validity (and that number of scraped games is equal to specified value)
            rand_id = f'{rand_year}{rand_season_type:02d}{rand_game:04d}'
            try: 
                #If game exists and has at least begun, then scraping can occur.
                rand_data = rs.get(f"https://api-web.nhle.com/v1/gamecenter/{rand_id}/play-by-play").json()
                if rand_data['gameState'] == 'FUT':
                    continue
                else:
                    i += 1
                    game_ids.append(rand_id)
            except: 
                continue
        
        print(f"\rGame IDs found in range {start}-{end}: {i}/{num}")
            
    #Scrape each game
    #Track Errors
    error_ids = []
    prog = 0
    for game_id in game_ids:
        print(f'Scraping data from game {game_id}...',end='')
        start = time.perf_counter()

        try:
            #Retrieve data
            info = get_game_info(game_id)
            data = asyncio.run(combine_data(info, sources))
                
            #Append data to list
            pbps.append(data)

            end = time.perf_counter()
            secs = end - start
            prog += 1
            
            #Export if sources is true
            if sources:
                dirs = f'sources/{info['season']}/'

                if not os.path.exists(dirs):
                    os.makedirs(dirs)

                data.to_csv(f'{dirs}{info['game_id']}.csv',index=False)

            print(f" finished in {secs:.2f} seconds. {prog}/{len(game_ids)} ({(prog/len(game_ids))*100:.2f}%)")
        except Exception as e:
            #Games such as the all-star game and pre-season games will incur this error
            #Other games have known problems
            if game_id in KNOWN_PROBS.keys():
                print(f"\nGame {game_id} has a known problem: {KNOWN_PROBS[game_id]}")
            else:
                print(f"\nUnable to scrape game {game_id}.  Exception: {e}")
            
            #Track error
            error_ids.append(game_id)
            
    #Add all pbps together
    if not pbps:
        print("\rNo data returned.")
        return pd.DataFrame()
    df = pd.concat(pbps)

    #If verbose is true features required to calculate xG are added to dataframe
    if verbose:
        df = prep_xG_data(df)
    else:
        ""

    #Print final message
    if error_ids:
        print(f'\rScrape of provided games finished.\nThe following games failed to scrape: {error_ids}')
    else:
        print('\rScrape of provided games finished.')
    
    #Split pbp and shift events if necessary
    #Return: complete play-by-play with data removed or split as necessary
    
    if split_shifts:
        remove.append('change')
        
        #Return: dict with pbp and shifts seperated
        pbp_dict = {"pbp":df.loc[~df['event_type'].isin(remove)],
            "shifts":df.loc[df['event_type']=='change']
            }
        
        if errors:
            pbp_dict.update({'errors':error_ids})

        return pbp_dict
    else:
        #Return: all events that are not set for removal by the provided list
        pbp = df.loc[~df['event_type'].isin(remove)]

        if errors:
            pbp_dict = {'pbp':pbp,
                        'errors':error_ids}
            
            return pbp_dict
        else:
            return pbp

def nhl_scrape_schedule(season:int | Literal['now'] = 'now', start:str = '', end:str = ''):
    """
    Given season and an optional date range, retrieve NHL schedule data.

    Args:
        season (int or str, optional): 
            The NHL season formatted such as "20242025" or "now".  Default is "now".
        start (str, optional): 
            The date string (MM-DD) to start the schedule scrape at. Default is a blank string.
        end (str, optional): 
            The date string (MM-DD) to end the schedule scrape at. Default is a blank string.

    Returns:
        pd.DataFrame: 
            A DataFrame containing the schedule data for the specified season and date range.
    """

    api = "https://api-web.nhle.com/v1/score/"
    form = '%Y-%m-%d'

    #If the season argument is now (live schedule) then skip this step
    if season == 'now':
        #Set start and end to filler values to ensure only one date is scraped (the phrase 'now' will be appened pre-scrape)
        start = end = datetime.now()
    else:
        #If either start or end are blank then find start and endpoints for specified season
        if start == '' or end == '':
            season_data = rs.get('https://api.nhle.com/stats/rest/en/season').json()['data']
            season_data = [s for s in season_data if s['id'] == season][0]
            start = season_data['startDate'][0:10]
            end = season_data['endDate'][0:10]
        else:
            #Determine how to approach scraping; if month in season is after the new year the year must be adjusted
            new_year = ["01","02","03","04","05","06"]
            if start[:2] in new_year:
                start = f'{int(str(season)[:4])+1}-{start}'
                end = f'{str(season)[:-4]}-{end}'
            else:
                start = f'{int(str(season)[:4])}-{start}'
                end = f'{str(season)[:-4]}-{end}'

        #Create datetime values from dates
        start = datetime.strptime(start,form)
        end = datetime.strptime(end,form)

    game = []

    day = (end-start).days+1
    if day < 0:
        #Handles dates which are over a year apart
        day = 365 + day
    for i in range(day):
        now = season == 'now'
        inc = start+timedelta(days=i)
        
        date_string = 'now' if now else str(inc)[:10]

        #For each day, call NHL api and retreive info on all games of selected game
        print(f'Scraping games {'as of' if now else 'on'} {date_string}...')
        
        get = rs.get(f'{api}{date_string}').json()
        gameWeek = pd.json_normalize(get['games']).drop(columns=['goals'],errors='ignore')
        
        #Return nothing if there's nothing
        if gameWeek.empty:
            game.append(gameWeek)
        else:
            gameWeek['game_date'] = get['currentDate']
            gameWeek['game_title'] = gameWeek['awayTeam.abbrev'] + " @ " + gameWeek['homeTeam.abbrev'] + " - " + gameWeek['game_date']
            gameWeek['start_time_est'] = pd.to_datetime(gameWeek['startTimeUTC']).dt.tz_convert('US/Eastern').dt.strftime("%I:%M %p")

        game.append(gameWeek)
        
    #Concatenate all games and standardize column naming
    df = pd.concat(game).rename(columns=COL_MAP['schedule'],errors='ignore')
    df = df.loc[:, ~df.columns.duplicated()]

    #Set logo links to dark variants (if any data exists)
    try:
        for team in ['away','home']:
            df[f'{team}_team_logo'] = df[f'{team}_team_logo'].str.replace('light','dark')
    except KeyError:
        print('No games found for range of dates provided.')

    #Return: specificed schedule data
    return df[[col for col in COL_MAP['schedule'].values() if col in df.columns]]

def nhl_scrape_season(season:int, split_shifts:bool = False, season_types:list[int] = [2,3], remove:list[str] = [], start:str = '', end:str = '', local:bool=False, local_path:str = SCHEDULE_PATH, verbose:bool = False, sources:bool = False, errors:bool = False):
    """
    Given season, scrape all play-by-play occuring within the season.

    Args:
        season (int): 
            The NHL season formatted such as "20242025".
        split_shifts (bool, optional):
            If True, returns a dict with separate 'pbp' and 'shifts' DataFrames. Default is False.
        season_types (List[int], optional):
            List of season_types to include in scraping process.  Default is all regular season and playoff games which are 2 and 3 respectively.
        remove (List[str], optional):
            List of event types to remove from the result. Default is an empty list.
        start (str, optional): 
            The date string (MM-DD) to start the schedule scrape at. Default is a blank string.
        end (str, optional): 
            The date string (MM-DD) to end the schedule scrape at. Default is a blank string.
        local (bool, optional):
            If True, use local file to retreive schedule data.
        local_path (bool, optional):
            If True, specifies the path with schedule data necessary to scrape a season's games (only relevant if local = True).
        verbose (bool, optional):
            If True, generates extra event features (such as those required to calculate xG). Default is False.
        sources (bool, optional):
            If True, saves raw HTML, JSON, SHIFTS, and single-game full play-by-play to a separate folder in the working directory. Default is False.
        errors (bool, optional):
            If True, includes a list of game IDs that failed to scrape in the return. Default is False.

    Returns:
        pd.DataFrame:
            If split_shifts is False, returns a single DataFrame of play-by-play data.
        dict[str, pd.DataFrame]:
            If split_shifts is True, returns a dictionary with keys:
            - 'pbp': play-by-play events
            - 'shifts': shift change events
            - 'errors' (optional): list of game IDs that failed if errors=True
    """
     
    #Determine whether to use schedule data in repository or to scrape
    local_failed = False

    if local:
        try:
            load = pd.read_csv(local_path)
            load['game_date'] = pd.to_datetime(load['game_date'])

            if start == '' or end == '':
                season_data = rs.get('https://api.nhle.com/stats/rest/en/season').json()['data']
                season_data = [s for s in season_data if s['id'] == season][0]
                
                season_start = season_data['startDate'][0:10]
                season_end = season_data['endDate'][0:10]

            else:   
                season_start = f'{(str(season)[0:4] if int(start[0:2])>=9 else str(season)[4:8])}-{start[0:2]}-{start[3:5]}'
                season_end =  f'{(str(season)[0:4] if int(end[0:2])>=9 else str(season)[4:8])}-{end[0:2]}-{end[3:5]}'

            form = '%Y-%m-%d'

            #Create datetime values from dates
            start_date = datetime.strptime(season_start,form)
            end_date = datetime.strptime(season_end,form)

            load = load.loc[(load['season']==season)&
                            (load['season_type'].isin(season_types))&
                            (load['game_date']>=start_date)&(load['game_date']<=end_date)&
                            (load['game_schedule_state']=='OK')&
                            (load['game_state']!='FUT')
                            ]
            
            game_ids = load['game_id'].to_list()
        except KeyError:
            #If loading games locally fails then force a scrape
            local_failed = True
            print('Loading games locally has failed.  Loading schedule data with a scrape...')
    else:
        local_failed = True

    if local_failed:
        load = nhl_scrape_schedule(season,start,end)
        load = load.loc[(load['season']==season)&
                        (load['season_type'].isin(season_types))&
                        (load['game_schedule_state']=='OK')&
                        (load['game_state']!='FUT')
                        ]
        
        game_ids = load['game_id'].to_list()

    #If no games found, terminate the process
    if not game_ids:
        print('No games found for dates in season...')
        return ""
    
    print(f"Scraping games from {str(season)[0:4]}-{str(season)[4:8]} season...")
    start = time.perf_counter()

    #Perform scrape
    if split_shifts:
        data = nhl_scrape_game(game_ids,split_shifts=True,remove=remove,verbose=verbose,sources=sources,errors=errors)
    else:
        data = nhl_scrape_game(game_ids,remove=remove,verbose=verbose,sources=sources,errors=errors)
    
    end = time.perf_counter()
    secs = end - start
    
    print(f'Finished season scrape in {(secs/60)/60:.2f} hours.')
    #Return: Complete pbp and shifts data for specified season as well as dataframe of game_ids which failed to return data
    return data

def nhl_scrape_seasons_info(seasons:list[int] = []):
    """
    Returns info related to NHL seasons (by default, all seasons are included)
    Args:
        seasons (List[int], optional): 
            The NHL season formatted such as "20242025".

    Returns:
        pd.DataFrame: 
            A DataFrame containing the information for requested seasons.
    """

    print(f'Scraping info for seasons: {seasons}')
    
    #Load two different data sources: general season info and standings data related to season
    api = "https://api.nhle.com/stats/rest/en/season"
    info = "https://api-web.nhle.com/v1/standings-season"
    data = rs.get(api).json()['data']
    data_2 = rs.get(info).json()['seasons']

    df = pd.json_normalize(data)
    df_2 = pd.json_normalize(data_2)

    #Remove common columns
    df_2 = df_2.drop(columns=['conferencesInUse', 'divisionsInUse', 'pointForOTlossInUse','rowInUse','tiesInUse','wildcardInUse'])
    
    df = pd.merge(df,df_2,how='outer',on=['id']).rename(columns=COL_MAP['season_info'])
    
    df = df[[col for col in COL_MAP['season_info'].values() if col in df.columns]]

    if len(seasons) > 0:
        return df.loc[df['season'].isin(seasons)].sort_values(by=['season'])
    else:
        return df.sort_values(by=['season'])

def nhl_scrape_standings(arg:int | list[int] | Literal['now'] = 'now', season_type:int = 2):
    """
    Returns standings or playoff bracket
    Args:
        arg (int or list[int] or str, optional):
            Date formatted as 'YYYY-MM-DD' to scrape standings, NHL season such as "20242025", list of NHL seasons, or 'now' for current standings. Default is 'now'.
        season_type (int, optional):
            Part of season to scrape.  If 3 (playoffs) then scrape the playoff bracket for the season implied by arg. When arg = 'now' this is defaulted to the most recent playoff year.  Any dates passed through are parsed as seasons. Default is 2.

    Returns:
        pd.DataFrame: 
            A DataFrame containing the standings information (or playoff bracket).
    """

    current_year = datetime.now().year

    if season_type == 3:
        if arg == "now":
            arg = [current_year]
        elif type(arg) == int:
            #Find year from season
            arg = [str(arg)[4:8]]
        elif type(arg) == list:
            #Find year from seasons
            arg = [str(s)[4:8] for s in arg]
        else:
            #Find year from season from date
            arg = [int(arg[0:4])+1 if (9 < int(arg[5:7]) < 13) else int(arg[0:4])]

        print(f"Scraping playoff bracket for season{'s' if len(arg)>1 else ''}: {arg}")
        
        dfs = []
        for season in arg:
            api = f"https://api-web.nhle.com/v1/playoff-bracket/{season}"

            data = rs.get(api).json()['series']
            dfs.append(pd.json_normalize(data))

        #Combine and standardize columns
        df = pd.concat(dfs).rename(columns=COL_MAP['standings'])

        #Return: playoff bracket
        return df[[col for col in COL_MAP['standings'].values() if col in df.columns]]

    else:
        if arg == "now":
            print("Scraping standings as of now...")
            arg = [arg]
        elif arg in nhl_scrape_seasons():
            print(f'Scraping standings for season: {arg}')
            arg = [arg]
        elif type(arg) == list:
            print(f'Scraping standings for seasons: {arg}')
        else:
            print(f"Scraping standings for date: {arg}")
            arg = [arg]

        dfs = []
        for search in arg:
            #If the end is an int then its a season otherwise it is either 'now' or a date as a string
            if type(search) == int:
                #Check if the season date is during the requested season - if so then use this date to find the current standings for the requested season
                season_data = rs.get('https://api.nhle.com/stats/rest/en/season').json()['data']
                season_data = [s for s in season_data if s['id'] == search][0]
                
                season_start = season_data['startDate']
                season_end = season_data['regularSeasonEndDate']

                today = datetime.now().strftime("%Y-%m-%d")
                
                if season_start <= today <= season_end:
                    end = today[0:10]
                else:
                    end = season_end[0:10]
            else:
                end = search
                
            api = f"https://api-web.nhle.com/v1/standings/{end}"

            data = rs.get(api).json()['standings']
            dfs.append(pd.json_normalize(data))

        #Standardize columns
        df = pd.concat(dfs).rename(columns=COL_MAP['standings'])

        #Return: standings data
        return df[[col for col in COL_MAP['standings'].values() if col in df.columns]]

def nhl_scrape_roster(season: int, teams: list[str] | None = None):
    """
    Returns rosters for a selection teams in a given season.

    Args:
        season (int):
            The NHL season formatted such as "20242025".

        teams (str, optional):
            List of teams(three letter abbreviation) to scrape.

    Returns:
        pd.DataFrame: 
            A DataFrame containing the rosters for all teams in the specified season.
    """

    print(f'Scrpaing rosters for the {season} season...')
    teaminfo = pd.read_csv(INFO_PATH)

    if not teams:
        teams = teaminfo['team_abbr'].drop_duplicates()

    rosts = []
    for team in teams:
        try:
            print(f'Scraping {team} roster...')
            api = f'https://api-web.nhle.com/v1/roster/{team}/{season}'
            
            data = rs.get(api).json()
            forwards = pd.json_normalize(data['forwards'])
            forwards['heading_position'] = "F"
            dmen = pd.json_normalize(data['defensemen'])
            dmen['heading_position'] = "D"
            goalies = pd.json_normalize(data['goalies'])
            goalies['heading_position'] = "G"

            roster = pd.concat([forwards,dmen,goalies]).reset_index(drop=True)
            roster['player_name'] = (roster['firstName.default']+" "+roster['lastName.default']).str.upper()
            roster['season'] = season
            roster['team_abbr'] = team

            rosts.append(roster)
        except:
            print(f'No roster found for {team}...')
            rosts.append(pd.DataFrame())

    #Combine rosters
    df = pd.concat(rosts)

    #Standardize columns
    df = df.rename(columns=COL_MAP['roster'])

    #Return: roster data for provided season
    return df[[col for col in COL_MAP['roster'].values() if col in df.columns]]

def nhl_scrape_prospects(team:str):
    """
    Returns prospects for specified team

    Args:
        team (str):
            Three character team abbreviation such as 'BOS'

    Returns:
        pd.DataFrame: 
            A DataFrame containing the prospect data for the specified team.
    """

    api = f'https://api-web.nhle.com/v1/prospects/{team}'

    data = rs.get(api).json()

    print(f'Scraping {team} prospects...')

    #Iterate through positions
    players = [pd.json_normalize(data[pos]) for pos in ['forwards','defensemen','goalies']]

    prospects = pd.concat(players)
    #Add name columns
    prospects['player_name'] = (prospects['firstName.default']+" "+prospects['lastName.default']).str.upper()

    #Standardize columns
    prospects = prospects.rename(columns=COL_MAP['prospects'])
    
    #Return: team prospects
    return prospects[[col for col in COL_MAP['prospects'].values() if col in prospects.columns]]

def nhl_scrape_team_info(country:bool = False):
    """
    Returns team or country information from the NHL API.

    Args:
        country (bool, optional):
            If True, returns country information instead of NHL team information.

    Returns:
        pd.DataFrame: 
            A DataFrame containing team or country information from the NHL API.
    """

    print(f'Scraping {'country' if country else 'team'} information...')
    api = f'https://api.nhle.com/stats/rest/en/{'country' if country else 'team'}'
    
    data =  pd.json_normalize(rs.get(api).json()['data'])

    #Add logos if necessary
    if not country:
        data['logo_light'] = 'https://assets.nhle.com/logos/nhl/svg/'+data['triCode']+'_light.svg'
        data['logo_dark'] = 'https://assets.nhle.com/logos/nhl/svg/'+data['triCode']+'_dark.svg'

    #Standardize columns
    data = data.rename(columns=COL_MAP['team_info'])

    #Return: team or country info 
    return data[[col for col in COL_MAP['team_info'].values() if col in data.columns]].sort_values(by=(['country_abbr','country_name'] if country else ['team_abbr','team_name']))

def nhl_scrape_player_info(player_ids:list[int]):
    """
    Returns player data for specified players.

    Args:
        player_ids (list[int]):
            List of NHL API player IDs to retrieve information for.

    Returns:
        pd.DataFrame: 
            A DataFrame containing player data for specified players.
    """

    print(f'Retreiving player information for {player_ids}...')

    #Wrap game_id in a list if only a single game_id is provided
    player_ids = [player_ids] if type(player_ids) != list else player_ids

    infos = []
    for player_id in player_ids:
        player_id = int(player_id)
        api = f'https://api-web.nhle.com/v1/player/{player_id}/landing'

        data = pd.json_normalize(rs.get(api).json())
        #Add name column
        data['player_name'] = (data['firstName.default'] + " " + data['lastName.default']).str.upper()

        #Append
        infos.append(data)

    if infos:
        df = pd.concat(infos)
        
        #Standardize columns
        df = df.rename(columns=COL_MAP['player_info'])

        #Return: player data
        return df[[col for col in COL_MAP['player_info'].values() if col in df.columns]]
    else:
        return pd.DataFrame()

def nhl_scrape_draft_rankings(arg:str | Literal['now'] = 'now', category:int = 0):
    """
    Returns draft rankings
    Args:
        arg (str, optional):
            Date formatted as 'YYYY-MM-DD' to scrape draft rankings for specific date or 'now' for current draft rankings. Default is 'now'.
        category (int, optional):
            Category number for prospects.  When arg = 'now' this does not apply.

            - Category 1 is North American Skaters.
            - Category 2 is International Skaters.
            - Category 3 is North American Goalies.
            - Category 4 is International Goalies

            Default is 0 (all prospects).
            
    Returns:
        pd.DataFrame: 
            A DataFrame containing draft rankings.
    """

    print(f'Scraping draft rankings for {arg}...\nCategory: {DRAFT_CAT[category]}...')

    #Player category only applies when requesting a specific season
    api = f"https://api-web.nhle.com/v1/draft/rankings/{arg}/{category}" if category > 0 else f"https://api-web.nhle.com/v1/draft/rankings/{arg}"
    data = pd.json_normalize(rs.get(api).json()['rankings'])

    #Add player name columns
    data['player_name'] = (data['firstName']+" "+data['lastName']).str.upper()

    #Fix positions
    data['positionCode'] = data['positionCode'].replace({
        'LW':'L',
        'RW':'R'
    })

    #Standardize columns
    data = data.rename(columns=COL_MAP['draft_rankings'])

    #Return: prospect rankings
    return data[[col for col in COL_MAP['draft_rankings'].values() if col in data.columns]]

def nhl_scrape_game_info(game_ids:list[int]):
    """
    Given a set of game_ids (NHL API), return information for each game.

    Args:
        game_ids (List[int] or ['random', int, int, int]):
            List of NHL game IDs to scrape or use ['random', n, start_year, end_year] to fetch n random games.
    
    Returns:
        pd.DataFrame:
            An DataFrame containing information for each game.    
    """

    #Wrap game_id in a list if only a single game_id is provided
    game_ids = [game_ids] if type(game_ids) != list else game_ids

    print(f'Finding game information for games: {game_ids}')

    link = 'https://api-web.nhle.com/v1/gamecenter'

    #Scrape information
    df = pd.concat([pd.json_normalize(rs.get(f'{link}/{game_id}/landing').json()) for game_id in game_ids])

    #Add extra info
    df['game_date'] = df['gameDate']
    df['game_title'] = df['awayTeam.abbrev'] + " @ " + df['homeTeam.abbrev'] + " - " + df['game_date']
    df['start_time_est'] = pd.to_datetime(df['startTimeUTC']).dt.tz_convert('US/Eastern').dt.strftime("%I:%M %p")

    #Standardize columns
    df = df.rename(columns=COL_MAP['schedule'])

    #Return: game information
    return df[[col for col in COL_MAP['schedule'].values() if col in df.columns]]

def nhl_scrape_edge(season: int, type: list[tuple[Literal['skater','goalie','team'], list[str | int]]], season_type:int = 2):
    """
    Returns NHL Edge stats and data for a selection of skaters, goalies, or teams in a given season.

    Args:
        season (int):
            The NHL season formatted such as "20242025".

        type (list[tuple[Literal['skater', 'goalie', 'team'], list[str | int]]]):  
            A list of selection tuples defining what data to scrape.
            Each tuple contains:
                - Entity type: `'skater'`, `'goalie'`, or `'team'`
                - Identifiers for the entity:
                    * Skater / goalie: NHL player IDs (int or str)
                    * Team: team abbreviations (str)

            Examples:
                ['skater', [8478402, 8480012]]
                ['goalie', [8471214]]
                ['team', ['BOS', 'TOR']]

        season_type (int, optional):  
            Season type identifier.
            Defaults to 2 (regular season).
            Common values:
                - 2: Regular season
                - 3: Playoffs

    Returns:
        pd.DataFrame:
            A DataFrame containing NHL EDGE metrics for the requested
            skaters, goalies, and/or teams for the specified season.
    """
    
    print(f'Scrpaing edge data for the {season} season...')

    #First value in type specifies the edge data to scrape
    scrape = type[0]

    #NHL edge endpoint for teams uses their team ID rather than their three-letter abbreviation
    if scrape == 'team':
        data = nhl_scrape_team_info()
        teams = data.set_index('team_abbr')['team_id'].to_dict()

        entries = [teams[team] for team in type[1]]
    else:
        entries = type[1]

    dfs = []
    for entry in entries:
        try:
            print(f'Scraping NHL Edge data for {scrape} {entry}...')
            api = f'https://api-web.nhle.com/v1/edge/{scrape}-detail/{entry}/{season}/{season_type}'
            
            data = rs.get(api).json()
            edge = pd.json_normalize(data)

            edge['season'] = season

            if scrape != 'team':
                edge['player_name'] = (edge['player.firstName.default']+" "+edge['player.lastName.default']).str.upper()

            dfs.append(edge)
        except:
            print(f'No NHL Edge data found for {scrape} {entry}...')
            dfs.append(pd.DataFrame())

    #Combine edge data
    df = pd.concat(dfs)

    #Standardize columns
    df = df.rename(columns=COL_MAP['edge'])

    #Return: dataframe including NHL Edge data for the specified type and the entries included
    return df[[col for col in COL_MAP['edge'].values() if col in df.columns]]

def nhl_scrape_seasons(analytic: bool = False):
    """
    Returns list of NHL seasons

    Args:
        analytic (bool, optional):
            Filters list of seasons to those only included in the WSBA Hockey package (2007-2008 and beyond) if True.  Default is False.

    Returns:
        pd.DataFrame:
            A DataFrame containing a list of all NHL seasons.
    """

    data = rs.get('https://api-web.nhle.com/v1/season').json()

    if analytic:
        data = [season for season in data if season > 20062007]

    return data

def nhl_apply_xG(pbp: pd.DataFrame):
    """
    Given play-by-play data, return this data with xG-related columns
    Args:
        pbp (pd.DataFrame):
            A DataFrame containing play-by-play data generated within the WBSA Hockey package.
    Returns:
        pd.DataFrame: 
            A DataFrame containing input play-by-play data with xG column.
    """

    print(f'Applying WSBA xG to model with seasons: {pbp['season'].drop_duplicates().to_list()}')

    #Apply xG model
    pbp = wsba_xG(pbp)
    
    return pbp

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
    #These are represented by their own metrics in which Goals = (Fenwick*(League Average Fenwick SH%)) + ((xGoals/Fenwick - League Average Fenwick SH%)*Fenwick) + (Goals - xGoals)
    def goal_comp(fenwick,xg_fen,xg,g,fsh):
        rate = fenwick * fsh
        qual = (xg_fen-fsh)*fenwick
        fini = g-xg

        return rate+qual+fini

    if type == 'goalie':
        pos = agg
        for group in [('OOFF','F'),('ODEF','A')]:
            #Have to set this columns for compatibility with df.apply
                pos['fsh'] = pos[f'Fsh{group[1]}%']
                pos['fenwick'] =  pos[f'F{group[1]}/60']
                pos['xg'] = pos[f'xG{group[1]}/60']
                pos['g'] = pos[f'G{group[1]}/60']
                pos['xg_fen'] = pos[f'xG{group[1]}/F{group[1]}']
                pos['finishing'] = pos[f'G{group[1]}/xG{group[1]}']

                #Find average for position in frame
                avg_fen = pos['fenwick'].mean()
                avg_xg = pos['xg'].mean()
                avg_g = pos['g'].mean()
                avg_fsh = avg_g/avg_fen
                avg_xg_fen = avg_xg/avg_fen

                #Calculate composite percentiles
                pos[f'{group[0]}-SR'] = pos['fenwick'].rank(pct=True)
                pos[f'{group[0]}-SQ'] = pos['xg_fen'].rank(pct=True)
                pos[f'{group[0]}-FN'] = pos['finishing'].rank(pct=True)

                #Calculate shot rate, shot quality, and finishing impacts
                pos[f'{group[0]}-SRI'] = pos['g'] - pos.apply(lambda x: goal_comp(avg_fen,x.xg_fen,x.xg,x.g,avg_fsh),axis=1)
                pos[f'{group[0]}-SQI'] = pos['g'] - pos.apply(lambda x: goal_comp(x.fenwick,avg_xg_fen,x.xg,x.g,avg_fsh),axis=1)
                pos[f'{group[0]}-FNI'] = pos['g'] - pos.apply(lambda x: goal_comp(x.fenwick,x.xg_fen,avg_xg,avg_g,avg_fsh),axis=1)

                #Convert impacts to totals
                #Calculate shot rate, shot quality, and finishing impacts
                pos[f'{group[0]}-SRI-T'] = (pos[f'{group[0]}-SRI']/60)*pos['TOI']
                pos[f'{group[0]}-SQI-T'] = (pos[f'{group[0]}-SQI']/60)*pos['TOI']
                pos[f'{group[0]}-FNI-T'] = (pos[f'{group[0]}-FNI']/60)*pos['TOI']
       
       #Rank per 60 stats
        for stat in ['FF','FA','xGF','xGA','GF','GA','CF','CA','GSAx']:
            pos[f'{stat}/60-P'] = pos[f'{stat}/60'].rank(pct=True)

        #Flip percentiles for against stats
        for stat in ['FA','xGA','GA','CA']:
            pos[f'{stat}/60-P'] = 1-pos[f'{stat}/60-P']

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

        #Flip against metric percentiles
        pos['ODEF-SR'] = 1-pos['ODEF-SR']
        pos['ODEF-SQ'] = 1-pos['ODEF-SQ']
        pos['ODEF-FN'] = 1-pos['ODEF-FN']

        #Extraneous Values
        pos['EGF'] = pos['OOFF-SRI']+pos['OOFF-SQI']+pos['OOFF-FNI']
        pos['ExGF'] = pos['OOFF-SRI']+pos['OOFF-SQI']
        pos['EGA'] = pos['ODEF-SRI']+pos['ODEF-SQI']+pos['ODEF-FNI']
        pos['ExGA'] = pos['ODEF-SRI']+pos['ODEF-SQI']

        #...and their percentiles
        pos['EGF-P'] = pos['EGF'].rank(pct=True)
        pos['ExGF-P'] = pos['ExGF'].rank(pct=True)
        pos['EGA-P'] = pos['EGA'].rank(pct=True)
        pos['ExGA-P'] = pos['ExGA'].rank(pct=True)

        pos['EGA-P'] = 1-pos['EGA']
        pos['ExGA-P'] = 1-pos['ExGA']

        #...and then their totals
        pos['EGF-T'] = (pos['EGF']/60)*pos['TOI']
        pos['ExGF-T'] = (pos['ExGF']/60)*pos['TOI']
        pos['EGA-T'] = (pos['EGA']/60)*pos['TOI']
        pos['ExGA-T'] = (pos['ExGA']/60)*pos['TOI']

        #Goal Composites...
        pos['Team-Adjusted-EGI'] = pos['ODEF-FNI']-pos['ExGA']
        pos['GISAx'] = pos['ExGA']-pos['EGA']
        pos['NetGI'] = pos['EGF'] - pos['EGA']
        pos['NetxGI'] = pos['ExGF'] - pos['ExGA']

        #...and their percentiles
        pos['Team-Adjusted-EGI-P'] = pos['Team-Adjusted-EGI'].rank(pct=True)
        pos['GISAx-P'] = pos['GISAx'].rank(pct=True)
        pos['NetGI-P'] = pos['NetGI'].rank(pct=True)
        pos['NetxGI-P'] = pos['NetxGI'].rank(pct=True)

        #...and then their totals
        pos['Team-Adjusted-EGI-T'] = (pos['Team-Adjusted-EGI']/60)*pos['TOI']
        pos['GISAx-T'] = (pos['GISAx']/60)*pos['TOI']
        pos['NetGI-T'] = (pos['NetGI']/60)*pos['TOI']
        pos['NetxGI-T'] = (pos['NetxGI']/60)*pos['TOI']

        #Return: team stats with shooting impacts
        return pos.drop(columns=['fsh','fenwick','xg_fen','xg','g','finishing']).sort_values(['Goalie','Season','Team'])

    elif type =='team':
        pos = agg
        for group in [('OOFF','F'),('ODEF','A')]:
            #Have to set this columns for compatibility with df.apply
                pos['fsh'] = pos[f'Fsh{group[1]}%']
                pos['fenwick'] =  pos[f'F{group[1]}/60']
                pos['xg'] = pos[f'xG{group[1]}/60']
                pos['g'] = pos[f'G{group[1]}/60']
                pos['xg_fen'] = pos[f'xG{group[1]}/F{group[1]}']
                pos['finishing'] = pos[f'G{group[1]}/xG{group[1]}']

                #Find average for position in frame
                avg_fen = pos['fenwick'].mean()
                avg_xg = pos['xg'].mean()
                avg_g = pos['g'].mean()
                avg_fsh = avg_g/avg_fen
                avg_xg_fen = avg_xg/avg_fen

                #Calculate composite percentiles
                pos[f'{group[0]}-SR'] = pos['fenwick'].rank(pct=True)
                pos[f'{group[0]}-SQ'] = pos['xg_fen'].rank(pct=True)
                pos[f'{group[0]}-FN'] = pos['finishing'].rank(pct=True)

                #Calculate shot rate, shot quality, and finishing impacts
                pos[f'{group[0]}-SRI'] = pos['g'] - pos.apply(lambda x: goal_comp(avg_fen,x.xg_fen,x.xg,x.g,avg_fsh),axis=1)
                pos[f'{group[0]}-SQI'] = pos['g'] - pos.apply(lambda x: goal_comp(x.fenwick,avg_xg_fen,x.xg,x.g,avg_fsh),axis=1)
                pos[f'{group[0]}-FNI'] = pos['g'] - pos.apply(lambda x: goal_comp(x.fenwick,x.xg_fen,avg_xg,avg_g,avg_fsh),axis=1)

                #Convert impacts to totals
                #Calculate shot rate, shot quality, and finishing impacts
                pos[f'{group[0]}-SRI-T'] = (pos[f'{group[0]}-SRI']/60)*pos['TOI']
                pos[f'{group[0]}-SQI-T'] = (pos[f'{group[0]}-SQI']/60)*pos['TOI']
                pos[f'{group[0]}-FNI-T'] = (pos[f'{group[0]}-FNI']/60)*pos['TOI']
       
       #Rank per 60 stats
        for stat in PER_SIXTY[11:len(PER_SIXTY)]:
            pos[f'{stat}/60-P'] = pos[f'{stat}/60'].rank(pct=True)

        #Flip percentiles for against stats
        for stat in ['FA','xGA','GA','CA','HA','Give','Penl','Penl2','Penl5']:
            pos[f'{stat}/60-P'] = 1-pos[f'{stat}/60-P']

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

        #Flip against metric percentiles
        pos['ODEF-SR'] = 1-pos['ODEF-SR']
        pos['ODEF-SQ'] = 1-pos['ODEF-SQ']
        pos['ODEF-FN'] = 1-pos['ODEF-FN']

        pos['EGF'] = pos['OOFF-SRI']+pos['OOFF-SQI']+pos['OOFF-FNI']
        pos['ExGF'] = pos['OOFF-SRI']+pos['OOFF-SQI']
        pos['EGA'] = pos['ODEF-SRI']+pos['ODEF-SQI']+pos['ODEF-FNI']
        pos['ExGA'] = pos['ODEF-SRI']+pos['ODEF-SQI']

        #...and their percentiles
        pos['EGF-P'] = pos['EGF'].rank(pct=True)
        pos['ExGF-P'] = pos['ExGF'].rank(pct=True)
        pos['EGA-P'] = pos['EGA'].rank(pct=True)
        pos['ExGA-P'] = pos['ExGA'].rank(pct=True)

        pos['EGA-P'] = 1-pos['EGA']
        pos['ExGA-P'] = 1-pos['ExGA']

        #...and then their totals
        pos['EGF-T'] = (pos['EGF']/60)*pos['TOI']
        pos['ExGF-T'] = (pos['ExGF']/60)*pos['TOI']
        pos['EGA-T'] = (pos['EGA']/60)*pos['TOI']
        pos['ExGA-T'] = (pos['ExGA']/60)*pos['TOI']

        #Return: team stats with shooting impacts
        return pos.drop(columns=['fsh','fenwick','xg_fen','xg','g','finishing']).sort_values(['Season','Team'])

    else:
        #Remove skaters with less than 150 minutes of TOI then split between forwards and dmen
        #These are added back in after the fact
        forwards = agg.loc[(agg['Position']!='D')&(agg['TOI']>=150)]
        defensemen = agg.loc[(agg['Position']=='D')&(agg['TOI']>=150)]
        non_players = agg.loc[agg['TOI']<150]

        #Loop through both positions, all groupings (INDV, OOFF, and ODEF) generating impacts
        for pos in [forwards,defensemen]:
            for group in [('INDV','i'),('OOFF','F'),('ODEF','A')]:
                #Have to set this columns for compatibility with df.apply
                pos['fsh'] = pos[f'Fsh{group[1]}%']
                pos['fenwick'] =  pos[f'F{group[1]}/60']
                pos['xg'] = pos[f'xG{group[1]}/60']
                pos['g'] = pos[f'G{group[1]}/60']
                pos['xg_fen'] = pos[f'xG{group[1]}/F{group[1]}']
                pos['finishing'] = pos[f'G{group[1]}/xG{group[1]}']

                #Find average for position in frame
                avg_fen = pos['fenwick'].mean()
                avg_xg = pos['xg'].mean()
                avg_g = pos['g'].mean()
                avg_fsh = avg_g/avg_fen
                avg_xg_fen = avg_xg/avg_fen

                #Calculate composite percentiles
                pos[f'{group[0]}-SR'] = pos['fenwick'].rank(pct=True)
                pos[f'{group[0]}-SQ'] = pos['xg_fen'].rank(pct=True)
                pos[f'{group[0]}-FN'] = pos['finishing'].rank(pct=True)

                #Calculate shot rate, shot quality, and finishing impacts
                pos[f'{group[0]}-SRI'] = pos['g'] - pos.apply(lambda x: goal_comp(avg_fen,x.xg_fen,x.xg,x.g,avg_fsh),axis=1)
                pos[f'{group[0]}-SQI'] = pos['g'] - pos.apply(lambda x: goal_comp(x.fenwick,avg_xg_fen,x.xg,x.g,avg_fsh),axis=1)
                pos[f'{group[0]}-FNI'] = pos['g'] - pos.apply(lambda x: goal_comp(x.fenwick,x.xg_fen,avg_xg,avg_g,avg_fsh),axis=1)

                #Convert impacts to totals
                #Calculate shot rate, shot quality, and finishing impacts
                pos[f'{group[0]}-SRI-T'] = (pos[f'{group[0]}-SRI']/60)*pos['TOI']
                pos[f'{group[0]}-SQI-T'] = (pos[f'{group[0]}-SQI']/60)*pos['TOI']
                pos[f'{group[0]}-FNI-T'] = (pos[f'{group[0]}-FNI']/60)*pos['TOI']

            #Calculate On-Ice Involvement Percentiles
            pos['Fi/F'] = pos['FC%'].rank(pct=True)
            pos['xGi/F'] = pos['xGC%'].rank(pct=True)
            pos['Pi/F'] = pos['GI%'].rank(pct=True)
            pos['Gi/F'] = pos['GC%'].rank(pct=True)
            pos['RushFi/60'] = (pos['Rush']/pos['TOI'])*60
            pos['RushxGi/60'] = (pos['Rush xG']/pos['TOI'])*60
            pos['RushesxGi'] = pos['RushxGi/60'].rank(pct=True)
            pos['RushesFi'] = pos['RushFi/60'].rank(pct=True)

            #Rank per 60 stats
            for stat in PER_SIXTY:
                pos[f'{stat}/60-P'] = pos[f'{stat}/60'].rank(pct=True)

            #Flip percentiles for against stats
            for stat in ['FA','xGA','GA','CA','HA','Give','Penl','Penl2','Penl5']:
                pos[f'{stat}/60-P'] = 1-pos[f'{stat}/60-P']

        #Add positions back together
        complete = pd.concat([forwards,defensemen])

        #Flip against metric percentiles
        complete['ODEF-SR'] = 1-complete['ODEF-SR']
        complete['ODEF-SQ'] = 1-complete['ODEF-SQ']
        complete['ODEF-FN'] = 1-complete['ODEF-FN']

        #Extraneous Values
        complete['EGi'] = complete['INDV-SRI']+complete['INDV-SQI']+complete['INDV-FNI']
        complete['ExGi'] = complete['INDV-SRI']+complete['INDV-SQI']
        complete['EGF'] = complete['OOFF-SRI']+complete['OOFF-SQI']+complete['OOFF-FNI']
        complete['ExGF'] = complete['OOFF-SRI']+complete['OOFF-SQI']
        complete['EGA'] = complete['ODEF-SRI']+complete['ODEF-SQI']+complete['ODEF-FNI']
        complete['ExGA'] = complete['ODEF-SRI']+complete['ODEF-SQI']

        #...and their percentiles
        complete['EGi-P'] = complete['EGi'].rank(pct=True)
        complete['ExGi-P'] = complete['ExGi'].rank(pct=True)
        complete['EGF-P'] = complete['EGF'].rank(pct=True)
        complete['ExGF-P'] = complete['ExGF'].rank(pct=True)
        complete['EGA-P'] = complete['EGA'].rank(pct=True)
        complete['ExGA-P'] = complete['ExGA'].rank(pct=True)

        complete['EGA-P'] = 1-complete['EGA']
        complete['ExGA-P'] = 1-complete['ExGA']

        #...and then their totals
        complete['EGi-T'] = (complete['EGi']/60)*complete['TOI']
        complete['ExGi-T'] = (complete['ExGi']/60)*complete['TOI']
        complete['EGF-T'] = (complete['EGF']/60)*complete['TOI']
        complete['ExGF-T'] = (complete['ExGF']/60)*complete['TOI']
        complete['EGA-T'] = (complete['EGA']/60)*complete['TOI']
        complete['ExGA-T'] = (complete['ExGA']/60)*complete['TOI']

        #Goal Composites...
        complete['LiEG'] = complete['EGF'] - complete['EGi']
        complete['LiExG'] = complete['ExGF'] - complete['ExGi']
        complete['LiGIn'] = complete['LiEG']*complete['AC%']
        complete['LixGIn'] = complete['LiExG']*complete['AC%']
        complete['ALiGIn'] = complete['LiGIn']-complete['LixGIn']
        complete['CompGI'] = complete['EGi'] + complete['LiGIn'] 
        complete['LiRelGI'] = complete['CompGI'] - (complete['EGF']-complete['CompGI'])
        complete['NetGI'] = complete['EGF'] - complete['EGA']
        complete['NetxGI'] = complete['ExGF'] - complete['ExGA']

        #...and their percentiles
        complete['LiEG-P'] = complete['LiEG'].rank(pct=True)
        complete['LiExG-P'] = complete['LiExG'].rank(pct=True)
        complete['LiGIn-P'] = complete['LiGIn'].rank(pct=True)
        complete['LixGIn-P'] = complete['LixGIn'].rank(pct=True)
        complete['ALiGIn-P'] = complete['ALiGIn'].rank(pct=True)
        complete['CompGI-P'] = complete['CompGI'].rank(pct=True)
        complete['LiRelGI-P'] = complete['LiRelGI'].rank(pct=True)
        complete['NetGI-P'] = complete['NetGI'].rank(pct=True)
        complete['NetxGI-P'] = complete['NetxGI'].rank(pct=True)

        #..and then their totals
        complete['LiEG-T'] = (complete['LiEG']/60)*complete['TOI']
        complete['LiExG-T'] = (complete['LiExG']/60)*complete['TOI']
        complete['LiGIn-T'] = (complete['LiGIn']/60)*complete['TOI']
        complete['LixGIn-T'] = (complete['LixGIn']/60)*complete['TOI']
        complete['ALiGIn-T'] = (complete['ALiGIn']/60)*complete['TOI']
        complete['CompGI-T'] = (complete['CompGI']/60)*complete['TOI']
        complete['LiRelGI-T'] = (complete['LiRelGI']/60)*complete['TOI']
        complete['NetGI-T'] = (complete['NetGI']/60)*complete['TOI']
        complete['NetxGI-T'] = (complete['NetxGI']/60)*complete['TOI']

        #Add back skaters with less than 150 minutes TOI
        df = pd.concat([complete,non_players]).drop(columns=['fsh','fenwick','xg_fen','xg','g','finishing']).sort_values(['Player','Season','Team','ID'])
        #Return: skater stats with shooting impacts
        return df

def nhl_calculate_stats(pbp:pd.DataFrame, type:Literal['skater','goalie','team','game_score'], game_strength:Union[Literal['all'], str, list[str]] = 'all', season_types:int | list[int] = [2,3], split_game:bool = False, roster_path:str = DEFAULT_ROSTER, shot_impact:bool = False, simple_col:bool = False):
    """
    Given play-by-play data, seasonal information, game strength, rosters, and an xG model,
    return aggregated statistics at the skater, goalie, or team level.

    Args:
        pbp (pd.DataFrame):
            A DataFrame containing play-by-play event data.
        type (Literal['skater', 'goalie', 'team', 'game_score']):
            Type of statistics to calculate. Must be one of 'skater', 'goalie', 'team', or 'game_score' (specific combination of skaters and goaltenders by game).
        season (int): 
            The NHL season formatted such as "20242025".
        game_strength (int or list[str], optional):
            List of game strength states to include (e.g., ['5v5','5v4','4v5']).  Default is 'all'.
        season_types (int or List[int], optional):
            List of season_types to include in scraping process.  Default is all regular season and playoff games which are 2 and 3 respectively.
        split_game (bool, optional):
            If True, aggregates stats separately for each game; otherwise, stats are aggregated across all games.  Value is ignored when type == 'game_score'.  Default is False.
        roster_path (str, optional):
            File path to the roster data used for mapping players and teams.
        shot_impact (bool, optional):
            If True, applies shot impact metrics to the stats DataFrame.  Default is False.
        simple_col (bool, optional):
            If True, retains the column names (abbreviated and non-standard) used when developing the package.  Default is False.
            
    Returns:
        pd.DataFrame:
            A DataFrame containing the aggregated statistics according to the selected parameters.
    """
    
    print(f"Calculating statistics for all games in the provided play-by-play data at {game_strength} for {type}s...\nSeasons included: {pbp['season'].drop_duplicates().to_list()}...")
    start = time.perf_counter()

    #Check if xG column exists and apply model if it does not
    try:
        pbp['xG']
    except KeyError: 
        pbp = wsba_xG(pbp)

    #If single values provided for columns typically in a list then place them into a list
    if isinstance(season_types, int):
        season_types = [season_types]
    if isinstance(game_strength, str) and game_strength != 'all':
        game_strength = [game_strength]

    #Apply season_type filter
    pbp = pbp.loc[(pbp['season_type'].isin(season_types))]

    #Convert all columns with player ids to float in order to avoid merging errors
    for col in get_col():
        if "_id" in col:
            try: pbp[col] = pbp[col].astype(float)
            except KeyError: continue

    #Split by game if specified
    if split_game:
        second_group = ['season','game_id']
    else:
        second_group = ['season']

    #Split calculation
    if type == 'game_score':
        #Create game score features for all positions
        skater = calc_game_score_features(pbp,'skater')
        goalie = calc_game_score_features(pbp,'goalie')

        #Generate game score with corresponding model
        dfs = []
        for label, df in [('skater',skater),('goalie',goalie)]:
            with open(os.path.join(GAME_SCORE,f'wsba_gs_{label}.json')) as f:
                model_data = json.load(f)

            #Extract model data
            coefficients = np.array(model_data["coefficients"])
            scaler_mean = np.array(model_data["scaler_mean"])
            scaler_scale = np.array(model_data["scaler_scale"])

            #Prepare features
            features = df[GS_SCORE_FEATURES[label]]

            #Scale features
            features_scaled = (features - scaler_mean) / scaler_scale

            #Display impact value of each variable
            for col, coef in zip(features.columns, coefficients):
                if col in df:
                    df[f'{col} Score'] = features_scaled[col]*coef

            #Calculate game score
            df["GS"] = df[[f'{col} Score' for col in GS_SCORE_FEATURES[label]]].sum(axis=1)

            dfs.append(df)
        
        #Combine game_score
        complete = pd.concat(dfs)

        #Remove entries with no ID listed
        complete = complete.loc[complete['ID'].notna()]

        #Import rosters and player info
        rosters = pd.read_csv(roster_path)
        names = rosters[['player_id','player_name',
                            'headshot','position','handedness',
                            'height_in','weight_lbs',
                            'birth_date','birth_country']].drop_duplicates(subset=['player_id','player_name'],keep='last')

        #Add names
        complete = pd.merge(complete,names,how='left',left_on='ID',right_on='player_id')

        #Rename if there are no missing names
        complete = complete.rename(columns={'player_name':'Player',
                                            'headshot':'Headshot',
                                            'position':'Position',
                                            'handedness':'Handedness',
                                            'height_in':'Height (in)',
                                            'weight_lbs':'Weight (lbs)',
                                            'birth_date':'Birthday',
                                            'birth_country':'Nationality'})
        
        #WSBA
        complete['WSBA'] = complete['ID'].astype(str).str.replace('.0','')+complete['Team']+complete['Season'].astype(str)

        #Add player age
        complete['Birthday'] = pd.to_datetime(complete['Birthday'])
        complete['season_year'] = complete['Season'].astype(str).str[4:8].astype(int)
        complete['Age'] = complete['season_year'] - complete['Birthday'].dt.year

        #Find player headshot
        complete['Headshot'] = 'https://assets.nhle.com/mugs/nhl/'+complete['Season'].astype(str)+'/'+complete['Team']+'/'+complete['ID'].astype(int).astype(str)+'.png'

        #Calculate game score composites
        for strength in ['EV','PP','SH']:
            complete[f'{strength} Score'] = complete[[col for col in complete.columns if f'{strength}_' in col and 'Score' in col and 'xGC%' not in col]].sum(axis=1)
        complete['Production Score'] = complete['P Score']
        complete['Play-Driving Score'] = complete[[col for col in complete.columns if f'_xGC%' in col and 'Score' in col]].sum(axis=1)
        complete['Offensive Score'] = complete['Production Score']+complete['Play-Driving Score']+complete[[col for col in complete.columns if f'_xGF' in col and 'Score' in col]].sum(axis=1)
        complete['Defensive Score'] = complete[[col for col in complete.columns if f'_xGA' in col and 'Score' in col]].sum(axis=1)
        complete['Penalties Score'] = complete['PENL% Score'] 
        complete['Puck Management Score'] = complete['PM% Score']
        complete['Faceoffs Score'] = complete['F% Score']
        complete['Misc Score'] = complete['PENL% Score']+complete['PM% Score']+complete['F% Score']
        complete['Workload Score'] = complete['xGF% Score']
        complete['Goaltending Score'] = complete['GA/xGA Score']

        complete = complete[[
            "Player","ID","Season","Team","Game",'WSBA',
            'Headshot','Position','Handedness',
            'Height (in)','Weight (lbs)',
            'Birthday','Age','Nationality',
            ]+GS_SCORE_FEATURES['skater']+GS_SCORE_FEATURES['goalie']+
            [f'{col} Score' for col in GS_SCORE_FEATURES['skater']]+[f'{col} Score' for col in GS_SCORE_FEATURES['goalie']]+
            ['EV Score','PP Score','SH Score','Production Score','Play-Driving Score',
             'Offensive Score','Defensive Score','Workload Score','Goaltending Score',
             'Penalties Score','Puck Management Score','Faceoffs Score',
             'Misc Score','GS']
        ]

        #Remove goalies that appear with skaters
        complete = complete.loc[(complete['Position']!='G') | ((complete['Position']=='G')&(complete['P'].isna()))]

        #Convert season name
        complete['Season'] = complete['Season'].replace(SEASON_NAMES)

        end = time.perf_counter()
        length = end-start
        print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')

        return complete if simple_col else complete.rename(columns=COL_MAP['stats'], errors='ignore')

    if type == 'goalie':
        complete = calc_goalie(pbp,game_strength,second_group)

        #Set TOI to minute
        complete['TOI'] = complete['TOI']/60

        #Add per 60 stats
        for stat in ['FF','FA','xGF','xGA','GF','GA','SF','SA','CF','CA','GSAx']:
            complete[f'{stat}/60'] = (complete[stat]/complete['TOI'])*60
            
        complete['GF%'] = complete['GF']/(complete['GF']+complete['GA'])
        complete['SF%'] = complete['SF']/(complete['SF']+complete['SA'])
        complete['xGF%'] = complete['xGF']/(complete['xGF']+complete['xGA'])
        complete['FF%'] = complete['FF']/(complete['FF']+complete['FA'])
        complete['CF%'] = complete['CF']/(complete['CF']+complete['CA'])

        #Remove entries with no ID listed
        complete = complete.loc[complete['ID'].notna()]

        #Import rosters and player info
        rosters = pd.read_csv(roster_path)
        names = rosters[['player_id','player_name',
                            'headshot','position','handedness',
                            'height_in','weight_lbs',
                            'birth_date','birth_country']].drop_duplicates(subset=['player_id','player_name'],keep='last')

        #Add names
        complete = pd.merge(complete,names,how='left',left_on='ID',right_on='player_id')

        #Rename if there are no missing names
        complete = complete.rename(columns={'player_name':'Goalie',
                                            'headshot':'Headshot',
                                            'position':'Position',
                                            'handedness':'Handedness',
                                            'height_in':'Height (in)',
                                            'weight_lbs':'Weight (lbs)',
                                            'birth_date':'Birthday',
                                            'birth_country':'Nationality'})
        
        #WSBA
        complete['WSBA'] = complete['ID'].astype(str).str.replace('.0','')+complete['Team']+complete['Season'].astype(str)

        #Add player age
        complete['Birthday'] = pd.to_datetime(complete['Birthday'])
        complete['season_year'] = complete['Season'].astype(str).str[4:8].astype(int)
        complete['Age'] = complete['season_year'] - complete['Birthday'].dt.year

        #Find player headshot
        complete['Headshot'] = 'https://assets.nhle.com/mugs/nhl/'+complete['Season'].astype(str)+'/'+complete['Team']+'/'+complete['ID'].astype(int).astype(str)+'.png'

        #Convert season name
        complete['Season'] = complete['Season'].replace(SEASON_NAMES)

        head = ['Goalie','ID','Game'] if 'Game' in complete.columns else ['Goalie','ID']
        complete = complete[head+[
            "Season","Team",'WSBA',
            'Headshot','Position','Handedness',
            'Height (in)','Weight (lbs)',
            'Birthday','Age','Nationality',
            'GP','TOI',
            "GF","SF","FF","xGF","xGF/FF","GF/xGF","ShF%","FshF%",
            "GA","SA","FA","xGA","xGA/FA","GA/xGA","ShA%","FshA%",
            'CF','CA',
            'GSAx',
            'RushF','RushA','RushFxG','RushAxG','RushFG','RushAG'
        ]+[f'{stat}/60' for stat in ['FF','FA','xGF','xGA','GF','GA','SF','SA','CF','CA','GSAx']]]

        #Apply shot impacts if necessary
        if shot_impact:
            complete = shooting_impacts(complete,'goalie')
    
        end = time.perf_counter()
        length = end-start
        print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')

        return complete if simple_col else complete.rename(columns=COL_MAP['stats'], errors='ignore')
        
    elif type == 'team':
        complete = calc_team(pbp,game_strength,second_group)

        #WSBA
        complete['WSBA'] = complete['Team']+complete['Season'].astype(str)

        #Set TOI to minute
        complete['TOI'] = complete['TOI']/60

        #Add per 60 stats
        for stat in PER_SIXTY[11:len(PER_SIXTY)]:
            complete[f'{stat}/60'] = (complete[stat]/complete['TOI'])*60
            
        complete['GF%'] = complete['GF']/(complete['GF']+complete['GA'])
        complete['SF%'] = complete['SF']/(complete['SF']+complete['SA'])
        complete['xGF%'] = complete['xGF']/(complete['xGF']+complete['xGA'])
        complete['FF%'] = complete['FF']/(complete['FF']+complete['FA'])
        complete['CF%'] = complete['CF']/(complete['CF']+complete['CA'])
        
        #Convert season name
        complete['Season'] = complete['Season'].replace(SEASON_NAMES)

        head = ['Team','Game'] if 'Game' in complete.columns else ['Team']
        complete = complete[head+[
            'Season','WSBA',
            'GP','TOI',
            "GF","SF","FF","xGF","xGF/FF","GF/xGF","ShF%","FshF%",
            "GA","SA","FA","xGA","xGA/FA","GA/xGA","ShA%","FshA%",
            'CF','CA',
            'GF%','SF%','FF%','xGF%','CF%',
            'HF','HA','HF%',
            'Penl','Penl2','Penl5','PIM','Draw','PENL%',
            'Give','Take','PM%',
            'Block',
            'RushF','RushA','RushFxG','RushAxG','RushFG','RushAG',
            'GSAx'
        ]+[f'{stat}/60' for stat in PER_SIXTY[11:len(PER_SIXTY)]]]
        #Apply shot impacts if necessary
        if shot_impact:
            complete = shooting_impacts(complete,'team')
        
        end = time.perf_counter()
        length = end-start
        print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')

        return complete if simple_col else complete.rename(columns=COL_MAP['stats'], errors='ignore')
    else:
        indv_stats = calc_indv(pbp,game_strength,second_group)
        onice_stats = calc_onice(pbp,game_strength,second_group)

        #IDs sometimes set as objects
        indv_stats['ID'] = indv_stats['ID'].astype(float)
        onice_stats['ID'] = onice_stats['ID'].astype(float)

        #Merge and add columns for extra stats
        complete = pd.merge(indv_stats,onice_stats,how="outer",on=['ID','Team','Season']+(['Game'] if 'game_id' in second_group else []))
        complete['GC%'] = complete['Gi']/complete['GF']
        complete['AC%'] = (complete['A1']+complete['A2'])/complete['GF']
        complete['GI%'] = (complete['Gi']+complete['A1']+complete['A2'])/complete['GF']
        complete['FC%'] = complete['Fi']/complete['FF']
        complete['xGC%'] = complete['xGi']/complete['xGF']
        complete['GF%'] = complete['GF']/(complete['GF']+complete['GA'])
        complete['SF%'] = complete['SF']/(complete['SF']+complete['SA'])
        complete['xGF%'] = complete['xGF']/(complete['xGF']+complete['xGA'])
        complete['FF%'] = complete['FF']/(complete['FF']+complete['FA'])
        complete['CF%'] = complete['CF']/(complete['CF']+complete['CA'])

        #Set TOI to minute
        complete['TOI'] = complete['TOI']/60

        #Add per 60 stats
        for stat in PER_SIXTY:
            complete[f'{stat}/60'] = (complete[stat]/complete['TOI'])*60

        #Shot Type Metrics
        type_metrics = []
        for type in shot_types:
            for stat in PER_SIXTY[:3]:
                type_metrics.append(f'{type.capitalize()}{stat}')

        #Remove entries with no ID listed
        complete = complete.loc[complete['ID'].notna()]

        #Import rosters and player info
        rosters = pd.read_csv(roster_path)
        names = rosters[['player_id','player_name',
                            'headshot','position','handedness',
                            'height_in','weight_lbs',
                            'birth_date','birth_country']].drop_duplicates(subset=['player_id','player_name'],keep='last')

        #Add names
        complete = pd.merge(complete,names,how='left',left_on='ID',right_on='player_id')

        #Rename if there are no missing names
        complete = complete.rename(columns={'player_name':'Player',
                                            'headshot':'Headshot',
                                            'position':'Position',
                                            'handedness':'Handedness',
                                            'height_in':'Height (in)',
                                            'weight_lbs':'Weight (lbs)',
                                            'birth_date':'Birthday',
                                            'birth_country':'Nationality'})

        #Add player age
        complete['Birthday'] = pd.to_datetime(complete['Birthday'])
        complete['season_year'] = complete['Season'].astype(str).str[4:8].astype(int)
        complete['Age'] = complete['season_year'] - complete['Birthday'].dt.year

        #Find player headshot
        complete['Headshot'] = 'https://assets.nhle.com/mugs/nhl/'+complete['Season'].astype(str)+'/'+complete['Team']+'/'+complete['ID'].astype(int).astype(str)+'.png'

        #Remove goalies that occasionally appear in a set
        complete = complete.loc[complete['Position']!='G']

        #Add WSBA ID
        complete['WSBA'] = complete['ID'].astype(str).str.replace('.0','')+complete['Season'].astype(str)+complete['Team']

        head = ['Player','ID','Game'] if 'Game' in complete.columns else ['Player','ID']
        complete = complete[head+[
            "Season","Team",'WSBA',
            'Headshot','Position','Handedness',
            'Height (in)','Weight (lbs)',
            'Birthday','Age','Nationality',
            'GP','TOI',
            "Gi","A1","A2",'P1','P','Si','Shi%',
            'Give','Take','PM%','HF','HA','HF%',
            "Fi","xGi",'xGi/Fi',"Gi/xGi","Fshi%",
            "GF","SF","FF","xGF","xGF/FF","GF/xGF","ShF%","FshF%",
            "GA","SA","FA","xGA","xGA/FA","GA/xGA","ShA%","FshA%",
            'Ci','CF','CA','CF%',
            'FF%','xGF%','GF%',
            'Rush',"Rush xG",'Rush G',"GC%","AC%","GI%","FC%","xGC%",
            'F','FW','FL','F%',
            'Penl','Penl2','Penl5',
            'Draw','PIM','PENL%',
            'Block',
            'OZF','NZF','DZF',
            'OZF%','NZF%','DZF%',
            'GSAx'
        ]+[f'{stat}/60' for stat in PER_SIXTY]+type_metrics].fillna(0).sort_values(['Player','Season','Team','ID'])
        
        #Apply shot impacts if necessary (Note: this will remove skaters with fewer than 150 minutes of TOI due to the shot impact TOI rule)
        if shot_impact:
            complete = shooting_impacts(complete,'skater')
            
        #Convert season name
        complete['Season'] = complete['Season'].replace(SEASON_NAMES)

        end = time.perf_counter()
        length = end-start
        print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')

        return complete if simple_col else complete.rename(columns=COL_MAP['stats'], errors='ignore')

def nhl_plot_skaters_shots(pbp:pd.DataFrame, skater_dict:dict[str | int, list[int, str]], strengths:Union[Literal['all'], list[str]] = 'all', strengths_title:str | None = None, marker_dict:dict = event_markers, situation:Literal['indv','for','against'] = ['indv'], title:bool = True, legend:bool = False):
    """
    Return a dictionary of shot plots for the specified skaters.

    Args:
        pbp (pd.DataFrame):
            A DataFrame containing play-by-play event data to be visualized.
        skater_dict (dict[str, list[str]]):
            Dictionary of skaters to plot, where each key is a player name and the value is a list 
            with season and team info (e.g., {'Patrice Bergeron': [20212022, 'BOS']} or {8470638: [20212022, 'BOS']}).
        strengths (str or list[str], optional):
            List of game strength states to include (e.g., ['5v5','5v4','4v5']).
        strengths_title (str or None, optional):
            Specify a title to describe the strengths states included in the plot.  Default is None (strengths shown will be a full list of the included strengths in the plot).
        marker_dict (dict[str, dict], optional):
            Dictionary of event types mapped to marker styles used in plotting.
        situation (Literal['indv', 'for', 'against'], optional):
            Determines which shot events to include for the player:
            - 'indv': only the player's own shots,
            - 'for': shots taken by the player's team while they are on ice,
            - 'against': shots taken by the opposing team while the player is on ice.
        title (bool, optional):
            Whether to include a plot title.
        legend (bool, optional):
            Whether to include a legend on the plots.

    Returns:
        Dict[str or int, Dict[int, Dict[str, matplotlib.figure.Figure]]]:
            A dictionary mapping each skater’s name or id to their corresponding season, team, then matplotlib heatmap figure.
    """

    print(f'Plotting the following skater shots: {skater_dict}...')

    roster = pd.read_csv(DEFAULT_ROSTER)

    #Iterate through skaters, adding plots to dict
    skater_plots = {}

    for skater in skater_dict.keys():
        skater_name = skater.title() if isinstance(skater, str) else roster.loc[roster['player_id']==skater,'player_name'].iloc[0].title()
        skater_info = skater_dict[skater]

        title = f'{skater_name} Fenwick Shots for {skater_info[1]} in {str(skater_info[0])[2:4]}-{str(skater_info[0])[6:8]}' if title else ''
        #Key is formatted as IDSEASONTEAM (i.e. 847063820212022BOS)
        skater_plots.update({skater:{skater_info[0]:{skater_info[1]:plot_skater_shots(pbp,skater,skater_info[0],skater_info[1],strengths,strengths_title,title,marker_dict,situation,legend)}}})

    #Return: list of plotted skater shot charts
    return skater_plots

def nhl_plot_heatmap(pbp:pd.DataFrame, player_dict:dict[str | int | Literal[8], list[int, str]], strengths:Union[Literal['all'], list[str]] = 'all', strengths_title:str | None = None, title:bool = True):
    """
    Return a dictionary of heatmaps for the specified players or teams.

    Args:
        pbp (pd.DataFrame):
            A DataFrame containing play-by-play event data to be visualized.
        player_dict (dict[str, list[str]]):
            Dictionary of players to plot, where each key is a player name and the value is a list 
            with season and team info (e.g., {'Patrice Bergeron': [20212022, 'BOS']} or {8470638: [20212022, 'BOS']}).  
            Setting the key to the int value 8 will generate a heatmap for the full team.
        strengths (str or list[str], optional):
            List of game strength states to include (e.g., ['5v5','5v4','4v5']).
        strengths_title (str or None, optional):
            Specify a title to describe the strengths states included in the plot.  Default is None (strengths shown will be a full list of the included strengths in the plot).
        title (bool, optional):
            Whether to include a plot title.

    Returns:
        Dict[str or int, Dict[int, Dict[str, matplotlib.figure.Figure]]]:
            A dictionary mapping each skater’s name or id to their corresponding season, team, then matplotlib heatmap figure.  The phrase 'Team' takes the place for team heatmaps.
    """

    print(f'Plotting full-ice heatmap for the following players or teams: {player_dict}...')

    roster = pd.read_csv(DEFAULT_ROSTER)

    #Iterate through players, adding plots to dict
    player_plots = {}

    for player in player_dict.keys():
        player_info = player_dict[player]

        if player == 8:
            player = None
            player_key = 'Team'
            title_header = f'{player_info[1]} Team Heatmap'
        else:
            player_key = player
            player_name = player.title() if isinstance(player, str) else roster.loc[roster['player_id']==player,'player_name'].iloc[0].title()
            title_header = f'{player_name} Heatmap for {player_info[1]}'

        title = f'{title_header} in {str(player_info[0])[2:4]}-{str(player_info[0])[6:8]}' if title else ''
        player_plots.update({player_key:{player_info[0]:{player_info[1]:gen_heatmap(pbp,player,player_info[0],player_info[1],strengths,'xG',strengths_title,title)}}})

    #Return: list of plotted player shot charts
    return player_plots

def nhl_plot_games(pbp:pd.DataFrame, events:list[str], strengths:Union[Literal['all'], list[str]] = 'all', game_ids: Union[Literal['all'], list[int]] = 'all', marker_dict:dict = event_markers, team_colors:dict = {'away':'primary','home':'primary'}, legend:bool =False):
    """
    Returns a dictionary of event plots for the specified games.

    Args:
        pbp (pd.DataFrame):
            A DataFrame containing play-by-play event data.
        events (list[str]):
            List of event types to include in the plot (e.g., ['shot-on-goal', 'goal']).
        strengths (str or list[str], optional):
            List of game strength states to include (e.g., ['5v5','5v4','4v5']).
        game_ids (str or list[int], optional):
            List of game IDs to plot. If set to 'all', plots will be generated for all games in the DataFrame.
        marker_dict (dict[str, dict], optional):
            Dictionary mapping event types to marker styles and/or colors used in plotting.
        legend (bool, optional):
            Whether to include a legend on the plots.

    Returns:
        dict[int, matplotlib.figure.Figure]:
            A dictionary mapping each game ID to its corresponding matplotlib event plot figure.
    """

    #Find games to scrape
    if game_ids == 'all':
        game_ids = pbp['game_id'].drop_duplicates().to_list()

    print(f'Plotting the following games: {game_ids}...')

    game_plots = {}
    #Iterate through games, adding plot to dict
    for game in game_ids:
        game_plots.update({game:plot_game_events(pbp,game,events,strengths,marker_dict,team_colors,legend)})

    #Return: list of plotted game events
    return game_plots

def nhl_plot_game_score(pbp:pd.DataFrame, game_ids: Union[Literal['all'], list[int]] = 'all'):
    """
    Returns a dictionary of game score bar charts for the specified games.

    Args:
        pbp (pd.DataFrame):
            A DataFrame containing play-by-play event data.
        game_ids (str or list[int], optional):
            List of game IDs to plot. If set to 'all', plots will be generated for all games in the DataFrame.

    Returns:
        dict[int, dict[str, matplotlib.figure.Figure]]:
            A dictionary mapping each game ID to its corresponding set of game score charts.
            Each game is separated into two keys, one for each team (three-letter abbreviation such as 'BOS').
    """

    #Find games to scrape
    if game_ids == 'all':
        game_ids = pbp['game_id'].drop_duplicates().to_list()

    print(f'Charting game score for the following games: {game_ids}...')

    game_plots = {}
    #Iterate through games, adding plot to dict
    for game in game_ids:
        pbp = pbp.loc[pbp['game_id']==game]
        stats = nhl_calculate_stats(pbp,'game_score').sort_values('game_score',ascending=False)

        game_plots.update({game:plot_game_score(stats)})

    #Return: list of charted games
    return game_plots

def repo_load_rosters(seasons:list[int] = []):
    """
    Returns roster data from repository

    Args:
        seasons (list[int], optional):
            A DataFrame containing play-by-play event data.

    Returns:
        pd.DataFrame:
            A DataFrame containing roster data for supplied seasons.
    """

    data = pd.read_csv(DEFAULT_ROSTER)
    if not seasons:
        data = data.loc[data['season'].isin(seasons)]

    return data

def repo_load_schedule(seasons:list[int] = []):
    """
    Returns schedule data from repository

    Args:
        seasons (list[int], optional):
            A DataFrame containing play-by-play event data.

    Returns:
        pd.DataFrame:
            A DataFrame containing the schedule data for the specified season and date range.    
    """

    data = pd.read_csv(SCHEDULE_PATH)
    if len(seasons)>0:
        data = data.loc[data['season'].isin(seasons)]

    return data

def repo_load_teaminfo():
    """
    Returns team data from repository

    Args:

    Returns:
        pd.DataFrame:
            A DataFrame containing general team information.
    """

    return pd.read_csv(INFO_PATH)

## CLASSES ##
class NHL_Database:
    """
    A class for managing and analyzing NHL play-by-play data.

    This class supports game scraping, filtering, stat calculation, and plotting.
    It initializes with either a provided list of game IDs or a default/random set.

    Attributes:
        name (str):
            Designated name of the database.
        pbp (pd.DataFrame): 
            Combined play-by-play data for selected games.
        games (list[int]): 
            Unique game IDs currently in the dataset.
        stats (dict[str, dict[str, pd.DataFrame]]): 
            Dictionary storing calculated stats by type and name.
        plots (dict[int, matplotlib.figure.Figure] |  dict[str or int, dict[int, dict[str, matplotlib.figure.Figure]]]): 
            Dictionary storing plot outputs keyed by game or event.

    Args:
        game_ids (list[int], optional): 
            List of game IDs to scrape initially.
        pbp (pd.DataFrame, optional): 
            Existing PBP DataFrame to load instead of scraping.
    """

    def __init__(self, name:str, game_ids:list[int] = [], pbp:pd.DataFrame = pd.DataFrame()):
        """
        Initialize the WSBA_Database with scraped or preloaded PBP data.

        If no `pbp` is provided and `game_ids` is empty, a random set of games will be scraped.

        Args:
            name (str):
                Name of database.
            game_ids (list[int], optional): 
                List of NHL game IDs to scrape in initialization.
            pbp (pd.DataFrame, optional): 
                Existing play-by-play data to initialization.

        Returns:
            pd.DataFrame: 
                The initialized play-by-play dataset.
        """

        print(f'Initializing database "{name}"...')
        self.name = name

        if game_ids:
            self.pbp = nhl_apply_xG(nhl_scrape_game(game_ids))
        else:
            self.pbp = nhl_apply_xG(nhl_scrape_game(['random',3,2007,2024])) if pbp.empty else pbp

        self.games = self.pbp['game_id'].drop_duplicates().to_list()
        self.stats = {}
        self.game_plots = {}
        self.plots = {}

    def add_games(self, game_ids:list[int]):
        """
        Add additional games to the existing play-by-play dataset.

        Args:
            game_ids (list[int]): 
                List of game IDs to scrape and append.

        Returns:
            pd.DataFrame: 
                The updated play-by-play dataset.
        """

        print('Adding games...')
        self.pbp = pd.concat([self.pbp,nhl_apply_xG(wsba.nhl_scrape_game(game_ids))])

        return self.pbp
    
    def select_games(self, game_ids:list[int]):
        """
        Return a filtered subset of the PBP data for specific games.

        Args:
            game_ids (list[int]): 
                List of game IDs to include.

        Returns:
            pd.DataFrame: 
                Filtered PBP data matching the selected games.
        """
         
        print('Selecting games...')

        df = self.pbp
        return df.loc[df['game_id'].isin(game_ids)]

    def add_stats(self, name:str, type:Literal['skater','goalie','team'], game_strength:Union[Literal['all'], str, list[str]] = 'all', season_types:int | list[int] = 2, split_game:bool = False, roster_path:str = DEFAULT_ROSTER, shot_impact:bool = False, simple_col:bool = False):
        """
        Calculate and store statistics for the given play-by-play data.

        Args:
            name (str): 
                Key name to store the results under.
            type (Literal['skater', 'goalie', 'team']):
                Type of statistics to calculate. Must be one of 'skater', 'goalie', or 'team'.
            season (int): 
                The NHL season formatted such as "20242025".
            game_strength (int or list[str]):
                List of game strength states to include (e.g., ['5v5','5v4','4v5']).
            season_types (int or List[int], optional):
                List of season_types to include in scraping process.  Default is all regular season and playoff games which are 2 and 3 respectively.
            split_game (bool, optional):
                If True, aggregates stats separately for each game; otherwise, stats are aggregated across all games.  Default is False.
            roster_path (str, optional):
                File path to the roster data used for mapping players and teams.
            shot_impact (bool, optional):
                If True, applies shot impact metrics to the stats DataFrame.  Default is False.
            simple_col (bool, optional):
                If True, retains the column names (abbreviated and non-standard) used when developing the package.  Default is False.

        Returns:
            pd.DataFrame: 
                The calculated statistics.
        """

        df =  wsba.nhl_calculate_stats(self.pbp, type, game_strength, season_types, split_game, roster_path, shot_impact, simple_col)
        self.stats.update({type:{name:df}})

        return df
    
    def get_players(self):
        """
        Return list of player IDs in the database.

        Returns:
            List: 
                List of player IDs.
        """

        return pd.unique(self.pbp[[
            'away_on_1_id','away_on_2_id','away_on_3_id','away_on_4_id','away_on_5_id','away_on_6_id','away_goalie_id',
            'home_on_1_id','home_on_2_id','home_on_3_id','home_on_4_id','home_on_5_id','home_on_6_id','home_goalie_id'
        ]].values.ravel()).tolist()
    
    def get_teams(self):
        """
        Return list of teams in the database.

        Returns:
            List: 
                List of teams IDs.
        """

        return pd.unique(self.pbp[['away_team_abbr','home_team_abbr']].values.ravel()).tolist()

    def get_seasons(self):
        """
        Return list of seasons in the database.

        Returns:
            List: 
                List of seasons IDs.
        """

        return pd.unique(self.pbp['season']).tolist()


    def add_game_plots(self, events:list[str], strengths:Union[Literal['all'], list[str]], game_ids: Union[Literal['all'], list[int]] = 'all', marker_dict:dict = event_markers, team_colors:dict = {'away':'primary','home':'primary'}, legend:bool = False):
        """
        Generate visualizations of game events based on play-by-play data.

        Args:
            events (list[str]):
                List of event types to include in the plot (e.g., ['shot-on-goal', 'goal']).
            strengths (str or list[str]):
                List of game strength states to include (e.g., ['5v5','5v4','4v5']).
            game_ids (str or list[int]):
                List of game IDs to plot. If set to 'all', plots will be generated for all games in the DataFrame.
            marker_dict (dict[str, dict]):
                Dictionary mapping event types to marker styles and/or colors used in plotting.
            team_colors (dict[str, str]):
                Dictionary mapping team venue (home or away) to its primary or secondary color.
            legend (bool):
                Whether to include a legend on the plots.

        Returns:
            dict[int, matplotlib.figure.Figure]:
                A dictionary mapping each game ID to its corresponding matplotlib event plot figure.
        """
        
        self.game_plots.update(nhl_plot_games(self.pbp, events, strengths, game_ids, marker_dict, team_colors, legend))

        return self.game_plots    
    
    def add_plots(self, plot:Literal['shot','heatmap'], player_dict:dict[str | int | Literal[8], list[int, str]], strengths:Union[Literal['all'], list[str]] = 'all', strengths_title:str | None = None, marker_dict:dict = event_markers, situation:Literal['indv','for','against'] = 'indv', title:bool = True, legend:bool = False):
        """
        Generate visualizations for players or teams based on play-by-play data.

        Args:
            plot (str):
                Type of plot to generate (shot plot or heatmap)
            player_dict (dict[str, list[str]]):
                Dictionary of players to plot, where each key is a player name and the value is a list 
                with season and team info (e.g., {'Patrice Bergeron': [20212022, 'BOS']} or {8470638: [20212022, 'BOS']}).  
                Setting the key to the int value 8 will generate a heatmap for the full team.

                If generating a shot plot, only skaters can be plotted.
            strengths (str or list[str], optional):
                List of game strength states to include (e.g., ['5v5','5v4','4v5']).
            strengths_title (str or None, optional):
                Specify a title to describe the strengths states included in the plot.  Default is None (strengths shown will be a full list of the included strengths in the plot).
            marker_dict (dict[str, dict]):
                Dictionary mapping event types to marker styles and/or colors used in plotting.  Only applies when plot is equal to 'shot'.
            situation (Literal['indv', 'for', 'against'], optional):
                Determines which shot events to include for the player:
                - 'indv': only the player's own shots,
                - 'for': shots taken by the player's team while they are on ice,
                - 'against': shots taken by the opposing team while the player is on ice.

                Only applies when plot is equal to 'shot'.
            title (bool, optional):
                Whether to include a plot title.
            legend (bool):
                Whether to include a legend on the plots.  Only applies when plot is equal to 'shot'.

        Returns:
            Dict[str or int, Dict[int, Dict[str, matplotlib.figure.Figure]]]:
                A dictionary mapping each skater’s name or id to their corresponding season, team, then matplotlib heatmap figure.  The phrase 'Team' takes the place for team heatmaps.
        """
        
        data = nhl_plot_skaters_shots(self.pbp,player_dict,strengths,strengths_title,marker_dict,situation,title,legend) if plot == 'shot' else nhl_plot_heatmap(self.pbp,player_dict,strengths,strengths_title,title)

        self.plots.update(data)

        return self.plots    
    
    def export_data(self, path:str = ''):
        """
        Export the data within the object to a specified directory.

        The method writes:
        - The full play-by-play DataFrame to a CSV file.
        - All calculated statistics by type and name to CSV files in subfolders.
        - All stored plots to PNG files.

        If no path is provided, exports to a folder named after the database (`self.name/`).

        Args:
            path (str, optional): 
                Root folder to export data into. Defaults to `self.name/`.
        """

        print(f'Exporting data in database "{self.name}"...')
        start = time.perf_counter()

        # Use default path if none provided
        path = f'{self.name}/' if path == '' else os.path.join(path,f'{self.name}')
        os.makedirs(path, exist_ok=True)

        # Export master PBP
        self.pbp.to_csv(os.path.join(path, 'pbp.csv'), index=False)

        # Export stats
        for stat_type in self.stats.keys():
            for name, df in self.stats[stat_type].items():
                stat_path = os.path.join(path, 'stats', stat_type)
                os.makedirs(stat_path, exist_ok=True)
                df.to_csv(os.path.join(stat_path, f'{name}.csv'), index=False)

        # Export game plots
        plot_path = os.path.join(path, 'game_plots')
        os.makedirs(plot_path, exist_ok=True)
        for game_id, plot in self.game_plots.items():
            plot.savefig(os.path.join(plot_path, f'{game_id}.png'), bbox_inches='tight')

        # Export plots
        plot_path = os.path.join(path, 'plots')
        os.makedirs(plot_path, exist_ok=True)
        for eid, seasons in self.plots.items():
            os.makedirs(f'{plot_path}/{eid}', exist_ok=True)
            for season, teams in seasons.items():
                os.makedirs(f'{plot_path}/{eid}/{season}', exist_ok=True)
                for team, plot in teams.items():
                    os.makedirs(f'{plot_path}/{eid}/{season}/{team}', exist_ok=True)
                    plot.savefig(os.path.join(plot_path, f'{eid}/{season}/{team}/plot.png'), bbox_inches='tight')

        # Completion message
        end = time.perf_counter()
        length = end - start
        print(f"...finished in {length:.2f} {'seconds' if length < 60 else 'minutes'}.")