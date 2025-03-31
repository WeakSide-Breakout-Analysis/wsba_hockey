import requests as rs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
from tools.scraping import *
from tools.xg_model import *
from tools.agg import *

### WSBA HOCKEY ###
## Provided below are all integral functions in the WSBA Hockey Python package. ##

## SCRAPE FUNCTIONS ##
def nhl_scrape_game(game_ids,split_shifts = False, remove = ['period-start','period-end','challenge','stoppage'],verbose=False):
    #Given a set of game_ids (NHL API), return complete play-by-play information as requested
    # param 'game_ids' - NHL game ids (or list formatted as ['random', num_of_games, start_year, end_year])
    # param 'split_shifts' - boolean which splits pbp and shift events if true
    # param 'remove' - list of events to remove from final dataframe
    # param 'xg' - xG model to apply to pbp for aggregation
    # param 'verbose' - boolean which adds additional event info if true

    pbps = []
    if game_ids[0] == 'random':
        #Randomize selection of game_ids
        #Some ids returned may be invalid (for example, 2020021300)
        num = game_ids[1]
        start = game_ids[2]
        end = game_ids[3]

        game_ids = []
        i = 0
        print("Finding valid, random game ids...")
        while i is not num:
            print(f"\rGame IDs found: {i}/{num}",end="")
            rand_year = random.randint(start,end)
            rand_season_type = random.randint(2,3)
            rand_game = random.randint(1,1312)

            #Ensure id validity (and that number of scraped games is equal to specified value)
            rand_id = f'{rand_year}{rand_season_type:02d}{rand_game:04d}'
            try: 
                rs.get(f"https://api-web.nhle.com/v1/gamecenter/{rand_id}/play-by-play").json()
                i += 1
                game_ids.append(rand_id)
            except: 
                continue
        
        print(f"\rGame IDs found: {i}/{num}")
            

    for game_id in game_ids:
        print("Scraping data from game " + str(game_id) + "...",end="")
        start = time.perf_counter()

        game_id = str(game_id)
        season = str(game_id[:4])+str(int(game_id[:4])+1)

        api = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        html = f"https://www.nhl.com/scores/htmlreports/{season}/PL{game_id[-6:]}.HTM"
        home_log = f"https://www.nhl.com/scores/htmlreports/{season}/TH{game_id[-6:]}.HTM"
        away_log = f"https://www.nhl.com/scores/htmlreports/{season}/TV{game_id[-6:]}.HTM"

        try: 
            #Retrieve raw data
            json = rs.get(api).json()
            html = rs.get(html).content
            home_shift = rs.get(home_log).content
            away_shift = rs.get(away_log).content

            #Parse JSONs and HTMLs
            data = combine_data(game_id,html,away_shift,home_shift,json)
                
            #Append data to list
            no_data = False
            pbps.append(data)

            end = time.perf_counter()
            secs = end - start
            print(f" finished in {secs} seconds.")

        except:
            #Games such as the all-star game and pre-season games will incur this error
            print(f"\nUnable to scrape game {game_id}.  Ensure the ID is properly inputted and formatted.")
            no_data = True
            pbps.append(pd.DataFrame())
      
    #Add all pbps together
    df = pd.concat(pbps)

    if verbose:
        df = prep_xG_data(df)
    else:
        ""
    #Split pbp and shift events if necessary
    #Return: complete play-by-play with data removed or split as necessary
    if no_data:
        return pd.DataFrame()

    
    if split_shifts == True:
        if len(remove) == 0:
            remove = ['change']
        
        #Return: dict with pbp and shifts seperated
        return {"pbp":df.loc[~df['event_type'].isin(remove)].dropna(axis=1,how='all'),
            "shifts":df.loc[df['event_type']=='change'].dropna(axis=1,how='all')
            }
    else:
        #Return: all events that are not set for removal by the provided list
        return df.loc[~df['event_type'].isin(remove)]

def nhl_scrape_schedule(season,start = "09-01", end = "08-01"):
    #Given a season, return schedule data
    # param 'season' - NHL season to scrape
    # param 'start' - Start date in season
    # param 'end' - End date in season

    api = "https://api-web.nhle.com/v1/schedule/"

    #Determine how to approach scraping; if month in season is after the new year the year must be adjusted
    new_year = ["01","02","03","04","05","06"]
    if start[:2] in new_year:
        start = str(int(season[:4])+1)+"-"+start
        end = str(season[:-4])+"-"+end
    else:
        start = str(season[:4])+"-"+start
        end = str(season[:-4])+"-"+end

    form = '%Y-%m-%d'

    #Create datetime values from dates
    start = datetime.strptime(start,form)
    end = datetime.strptime(end,form)

    game = []

    day = (end-start).days+1
    if day < 0:
        #Handles dates which are over a year apart
        day = 365 + day
    for i in range(day):
        #For each day, call NHL api and retreive id, season, season_type (1,2,3), and gamecenter link
        inc = start+timedelta(days=i)
        print("Scraping games on " + str(inc)[:10]+"...")
        
        get = rs.get(api+str(inc)[:10]).json()
        gameWeek = list(pd.json_normalize(get['gameWeek'])['games'])[0]

        for i in range(0,len(gameWeek)):
            game.append(pd.DataFrame({
                "id": [gameWeek[i]['id']],
                "season": [gameWeek[i]['season']],
                "season_type":[gameWeek[i]['gameType']],
                "away_team_abbr":[gameWeek[i]['awayTeam']['abbrev']],
                "home_team_abbr":[gameWeek[i]['homeTeam']['abbrev']],
                "gamecenter_link":[gameWeek[i]['gameCenterLink']]
                }))
    
    #Concatenate all games
    df = pd.concat(game)
    
    #Return: specificed schedule data
    return df

def nhl_scrape_season(season,split_shifts = False, season_types = [2,3], remove = ['period-start','period-end','game-end','challenge','stoppage'], start = "09-01", end = "08-01", local=False, local_path = "schedule/schedule.csv", verbose = False):
    #Given season, scrape all play-by-play occuring within the season
    # param 'season' - NHL season to scrape
    # param 'split_shifts' - boolean which splits pbp and shift events if true
    # param 'remove' - list of events to remove from final dataframe
    # param 'start' - Start date in season
    # param 'end' - End date in season
    # param 'local' - boolean indicating whether to use local file to scrape game_ids
    # param 'local_path' - path of local file
    # param 'verbose' - boolean which adds additional event info if true

    #Determine whether to use schedule data in repository or to scrape
    if local == True:
        load = pd.read_csv(local_path)
        load = load.loc[(load['season'].astype(str)==season)&(load['season_type'].isin(season_types))]
        game_ids = list(load['id'].astype(str))
    else:
        load = nhl_scrape_schedule(season,start,end)
        load = load.loc[(load['season'].astype(str)==season)&(load['season_type'].isin(season_types))]
        game_ids = list(load['id'].astype(str))

    df = []
    df_s = []

    print(f"Scraping games from {season[0:4]}-{season[4:8]} season...")
    start = time.perf_counter()
    for game_id in game_ids: 
        try:
            if split_shifts == True:
                data = nhl_scrape_game([game_id],split_shifts=True,remove=remove,verbose=verbose)
                df.append(data['pbp'])
                df_s.append(data['shifts'])
            else:
                data = nhl_scrape_game([game_id],remove=remove,verbose=verbose)
                df.append(data)

        except: 
            #Errors should be rare; testing of eight full-season scraped produced just one missing regular season game due to error
            continue

    #Missing data handled as a KeyError
    try: pbp = pd.concat(df)
    except: 
        raise KeyError("No data is available to return.")
        
    if split_shifts == True:
        try: shifts = pd.concat(df_s)
        except: raise KeyError("No data is available to return.")
    else:
        ""
    
    end = time.perf_counter()
    secs = end - start
    print(f'Finished season scrape in {(secs/60)/60} hours.')
    #Return: Complete pbp and shifts data for specified season as well as dataframe of game_ids which failed to return data
    if split_shifts == True:
        return {"pbp":pbp,
            'shifts':shifts}
    else:
        return pbp

def nhl_scrape_seasons_info(seasons = []):
    #Returns info related to NHL seasons (by default, all seasons are included)
    # param 'season' - list of seasons to include

    print("Scraping info for seasons: " + str(seasons))
    api = "https://api.nhle.com/stats/rest/en/season"
    info = "https://api-web.nhle.com/v1/standings-season"
    data = rs.get(api).json()['data']
    data_2 = rs.get(info).json()['seasons']

    df = pd.json_normalize(data)
    df_2 = pd.json_normalize(data_2)

    df = pd.merge(df,df_2,how='outer',on=['id'])
    
    if len(seasons) > 0:
        return df.loc[df['id'].astype(str).isin(seasons)].sort_values(by=['id'])
    else:
        return df.sort_values(by=['id'])

def nhl_scrape_standings(arg = "now",season_type = 2):
    #Returns standings
    # param 'arg' - by default, this is "now" returning active NHL standings.  May also be a specific date formatted as YYYY-MM-DD
    # param 'season_type' - by default, this scrapes the regular season standings.  If set to 3, it returns the playoff bracket for the specified season

    #arg param is ignored when set to "now" if season_type param is 3
    if season_type == 3:
        if arg == "now":
            arg = "2024"

        print("Scraping playoff bracket for season: "+arg)
        api = "https://api-web.nhle.com/v1/playoff-bracket/"+arg
    
        data = rs.get(api).json()['series']

        return pd.json_normalize(data)

    else:
        if arg == "now":
            print("Scraping standings as of now...")

        print("Scraping standings for season: "+arg)
        api = "https://api-web.nhle.com/v1/standings/"+arg
    
        data = rs.get(api).json()['standings']

        return pd.json_normalize(data)

def nhl_scrape_roster(season):
    #Given a nhl season, return rosters for all participating teams
    # param 'season' - NHL season to scrape
    print("Scrpaing rosters for the "+ season + "season...")
    teaminfo = pd.read_csv("teaminfo/nhl_teaminfo.csv")

    rosts = []
    for team in list(teaminfo['Team']):
        try:
            print("Scraping " + team + " roster...")
            api = "https://api-web.nhle.com/v1/roster/"+team+"/"+season
            
            data = rs.get(api).json()
            forwards = pd.json_normalize(data['forwards'])
            forwards['headingPosition'] = "F"
            dmen = pd.json_normalize(data['defensemen'])
            dmen['headingPosition'] = "D"
            goalies = pd.json_normalize(data['goalies'])
            goalies['headingPosition'] = "G"

            roster = pd.concat([forwards,dmen,goalies]).reset_index(drop=True)
            roster['fullName'] = (roster['firstName.default']+" "+roster['lastName.default']).str.upper()
            roster['season'] = str(season)
            roster['team_abbr'] = team

            rosts.append(roster)
        except:
            print("No roster found for " + team + "...")

    return pd.concat(rosts)

def nhl_scrape_player_info(roster):
    #Given compiled roster information from the nhl_scrape_roster function, return a list of all players (seperated into team and season) and associated information
    # param 'roster' - dataframe of roster information from the nhl_scrape_roster function

    data = roster

    print("Creating player info for provided roster data...")

    alt_name_col = ['firstName.cs',	'firstName.de',	'firstName.es',	'firstName.fi',	'firstName.sk',	'firstName.sv']
    for i in range(len(alt_name_col)):
        try: data['fullName.'+str(i+1)] = np.where(data[alt_name_col[i]].notna(),(data[alt_name_col[i]].astype(str)+" "+data['lastName.default'].astype(str)).str.upper(),np.nan)
        except: continue

    name_col = ['fullName',	'fullName.1',	'fullName.2',	'fullName.3',	'fullName.4',	'fullName.5', 'fullName.6']

    for name in name_col:
        try: data[name]
        except:
            data[name] = np.nan

    infos = []
    for name in name_col:
        infos.append(data[[name,"id","season","team_abbr","headshot",
                              "sweaterNumber","headingPosition",
                              "positionCode",'shootsCatches',
                              'heightInInches','weightInPounds',
                              'birthDate','birthCountry']].rename(columns={
                                                              name:'Player',
                                                              'id':"API",
                                                              "season":"Season",
                                                              "team_abbr":"Team",
                                                              'headshot':'Headshot',
                                                              'sweaterNumber':"Number",
                                                              'headingPosition':"Primary Position",
                                                              'positionCode':'Position',
                                                              'shootsCatches':'Handedness',
                                                              'heightInInches':'Height',
                                                              'weightInPounds':'Weight',
                                                              'birthDate':'Birthday',
                                                              'birthCountry':'Nationality'}))
    players = pd.concat(infos)
    players['Season'] = players['Season'].astype(str)
    players['Player'] = players['Player'].replace(r'^\s*$', np.nan, regex=True)

    return players.loc[players['Player'].notna()].sort_values(by=['Player','Season','Team'])

def nhl_scrape_team_info():
    #Return team information

    print('Scraping team information...')
    api = 'https://api.nhle.com/stats/rest/en/team'
    
    #Return: team information
    return pd.json_normalize(rs.get(api).json()['data'])

def nhl_calculate_stats(pbp,season,season_types,game_strength,roster_path="rosters/nhl_rosters.csv",xg="moneypuck"):
    #Given play-by-play, seasonal information, game_strength, rosters, and xG model, return aggregated stats
    # param 'pbp' - Scraped play-by-play
    # param 'season' - season or timeframe of events in play-by-play
    # param 'season_type' - list of season types (preseason, regular season, or playoffs) to include in aggregation
    # param 'game_strength' - list of game_strengths to include in aggregation
    # param 'roster_path' - path to roster file
    # param 'xg' - xG model to apply to pbp for aggregation

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
    info = calc_toi(pbp)

    #IDs sometimes set as objects
    indv_stats['ID'] = indv_stats['ID'].astype(float)
    onice_stats['ID'] = onice_stats['ID'].astype(float)
    info['ID'] = info['ID'].astype(float)

    complete = pd.merge(indv_stats,onice_stats,how="outer",on=['ID','Team'])
    complete = pd.merge(complete,info,how="outer",on=['ID','Team'])
    complete['Season'] = season
    complete['GC%'] = complete['G']/complete['GF']
    complete['AC%'] = (complete['A1']+complete['A2'])/complete['GF']
    complete['GI%'] = (complete['G']+complete['A1']+complete['A2'])/complete['GF']
    complete['FC%'] = complete['iFF']/complete['FF']
    complete['xGC%'] = complete['ixG']/complete['xGF']
    #complete['RC%'] = complete['Rush']/complete['iFF']
    #complete['AVG_Rush_POW'] = complete['Rush_POW']/complete['Rush']
    
    #Import rosters and player info
    rosters = pd.read_csv(roster_path)
    names = rosters[['id','fullName']].drop_duplicates()

    complete = pd.merge(complete,names,how='left',left_on='ID',right_on='id').rename(columns={'fullName':'Player'})
    return complete[[
        'Player','ID',"Season","Team","GP","TOI",
        "G","A1","A2","iFF","ixG",'ixG/iFF',"G/ixG","iFsh%",
        "GF","FF","xGF","xGF/FF","GF/xGF","FshF%",
        "GA","FA","xGA","xGA/FA","GA/xGA","FshA%",
        "GC%","AC%","GI%","FC%","xGC%"
    ]].fillna(0).sort_values(['Player','Season','Team','ID'])

def repo_load_rosters(seasons = []):
    #Returns roster data from repository
    # param 'seasons' - list of seasons to include

    data = pd.read_csv("rosters/nhl_rosters.csv")
    if len(seasons)>0:
        data = data.loc[data['season'].isin(seasons)]

    return data

def repo_load_schedule(seasons = []):
    #Returns schedule data from repository
    # param 'seasons' - list of seasons to include

    data = pd.read_csv("schedule/schedule.csv")
    if len(seasons)>0:
        data = data.loc[data['season'].isin(seasons)]

    return data

def repo_load_teaminfo():
    #Returns team data from repository

    return pd.read_csv("teaminfo/nhl_teaminfo.csv")
