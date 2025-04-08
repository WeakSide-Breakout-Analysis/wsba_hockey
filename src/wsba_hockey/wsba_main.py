import requests as rs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import time
import random
from tools.scraping import *
from tools.xg_model import *
from tools.agg import *
from tools.ncaa_scraping import *

### WSBA HOCKEY ###
## Provided below are all integral functions in the WSBA Hockey Python package. ##

## GLOBAL VARIABLES ##
seasons = [
    '20072008',
    '20082009',
    '20092010',
    '20102011',
    '20112012',
    '20122013',
    '20132014',
    '20142015',
    '20152016',
    '20162017',
    '20172018',
    '20182019',
    '20192020',
    '20202021',
    '20212022',
    '20222023',
    '20232024',
    '20242025'
]

#Some games in the API are specifically known to cause errors in scraping.
#This list is updated as frequently as necessary
known_probs ={
    '2007020011':'Missing shifts data for game between Chicago and Minnesota.',
    '2007021178':'Game between the Bruins and Sabres is missing data after the second period, for some reason.',
    '2008020259':'HTML data is completely missing for this game.',
    '2008020409':'HTML data is completely missing for this game.',
    '2008021077':'HTML data is completely missing for this game.',
    '2009020081':'HTML pbp for this game between Pittsburgh and Carolina is missing all but the period start and first faceoff events, for some reason.',
    '2009020658':'Missing shifts data for game between New York Islanders and Dallas.',
    '2009020885':'Missing shifts data for game between Sharks and Blue Jackets.',
    '2010020124':'Game between Capitals and Hurricanes is sporadically missing player on-ice data',
    '2013020971':'On March 10th, 2014, Stars forward Rich Peverley suffered from a cardiac episode midgame and as a result, the remainder of the game was postponed.  \nThe game resumed on April 9th, and the only goal scorer in the game, Blue Jackets forward Nathan Horton, did not appear in the resumed game due to injury.  Interestingly, Horton would never play in the NHL again.',
    '2019020876':'Due to the frightening collapse of Blues defensemen Jay Bouwmeester, a game on February 2nd, 2020 between the Ducks and Blues was postponed.  \nWhen the game resumed, Ducks defensemen Hampus Lindholm, who assisted on a goal in the inital game, did not play in the resumed match.'
}

## SCRAPE FUNCTIONS ##
def nhl_scrape_game(game_ids,split_shifts = False, remove = ['period-start','period-end','challenge','stoppage'],verbose = False, errors = False):
    #Given a set of game_ids (NHL API), return complete play-by-play information as requested
    # param 'game_ids' - NHL game ids (or list formatted as ['random', num_of_games, start_year, end_year])
    # param 'split_shifts' - boolean which splits pbp and shift events if true
    # param 'remove' - list of events to remove from final dataframe
    # param 'xg' - xG model to apply to pbp for aggregation
    # param 'verbose' - boolean which adds additional event info if true
    # param 'errors' - boolean returning game ids which did not scrape if true

    pbps = []
    if game_ids[0] == 'random':
        #Randomize selection of game_ids
        #Some ids returned may be invalid (for example, 2020021300)
        num = game_ids[1]
        try: 
            start = game_ids[2]
        except:
            start = 2007
        try:
            end = game_ids[3]
        except:
            end = (date.today().year)-1

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
                rs.get(f"https://api-web.nhle.com/v1/gamecenter/{rand_id}/play-by-play").json()
                i += 1
                game_ids.append(rand_id)
            except: 
                continue
        
        print(f"\rGame IDs found in range {start}-{end}: {i}/{num}")
            
    #Scrape each game
    #Track Errors
    error_ids = []
    for game_id in game_ids:
        print("Scraping data from game " + str(game_id) + "...",end="")
        start = time.perf_counter()

        try:
            #Retrieve data
            info = get_game_info(game_id)
            data = combine_data(info)
                
            #Append data to list
            pbps.append(data)

            end = time.perf_counter()
            secs = end - start
            print(f" finished in {secs:.2f} seconds.")

        except:
            #Games such as the all-star game and pre-season games will incur this error
            #Other games have known problems
            if game_id in known_probs.keys():
                print(f"\nGame {game_id} has a known problem: {known_probs[game_id]}")
            else:
                print(f"\nUnable to scrape game {game_id}.  Ensure the ID is properly inputted and formatted.")
            
            #Track error
            error_ids.append(game_id)
     
    #Add all pbps together
    if len(pbps) == 0:
        print("\rNo data returned.")
        return pd.DataFrame()
    df = pd.concat(pbps)

    #If verbose is true features required to calculate xG are added to dataframe
    if verbose:
        df = prep_xG_data(df)
    else:
        ""

    #Print final message
    if len(error_ids) > 0:
        print(f'\rScrape of provided games finished.\nThe following games failed to scrape: {error_ids}')
    else:
        print('\rScrape of provided games finished.')
    
    #Split pbp and shift events if necessary
    #Return: complete play-by-play with data removed or split as necessary
    
    if split_shifts == True:
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

def nhl_scrape_season(season,split_shifts = False, season_types = [2,3], remove = ['period-start','period-end','game-end','challenge','stoppage'], start = "09-01", end = "08-01", local=False, local_path = "schedule/schedule.csv", verbose = False, errors = False):
    #Given season, scrape all play-by-play occuring within the season
    # param 'season' - NHL season to scrape
    # param 'split_shifts' - boolean which splits pbp and shift events if true
    # param 'remove' - list of events to remove from final dataframe
    # param 'start' - Start date in season
    # param 'end' - End date in season
    # param 'local' - boolean indicating whether to use local file to scrape game_ids
    # param 'local_path' - path of local file
    # param 'verbose' - boolean which adds additional event info if true
    # param 'errors' - boolean returning game ids which did not scrape if true

    #Determine whether to use schedule data in repository or to scrape
    if local == True:
        load = pd.read_csv(local_path)
        load = load.loc[(load['season'].astype(str)==season)&(load['season_type'].isin(season_types))]
        game_ids = list(load['id'].astype(str))
    else:
        load = nhl_scrape_schedule(season,start,end)
        load = load.loc[(load['season'].astype(str)==season)&(load['season_type'].isin(season_types))]
        game_ids = list(load['id'].astype(str))

    #If no games found, terminate the process
    if not game_ids:
        print('No games found for dates in season...')
        return ""
    
    print(f"Scraping games from {season[0:4]}-{season[4:8]} season...")
    start = time.perf_counter()

    #Perform scrape
    if split_shifts == True:
        data = nhl_scrape_game(game_ids,split_shifts=True,remove=remove,verbose=verbose,errors=errors)
    else:
        data = nhl_scrape_game(game_ids,remove=remove,verbose=verbose,errors=errors)
    
    end = time.perf_counter()
    secs = end - start
    
    print(f'Finished season scrape in {(secs/60)/60:.2f} hours.')
    #Return: Complete pbp and shifts data for specified season as well as dataframe of game_ids which failed to return data
    if split_shifts == True:
        pbp_dict = {'pbp':data['pbp'],
            'shifts':data['shifts']}
        
        if errors:
            pbp_dict.update({'errors':data['errors']})
        return pbp_dict
    else:
        pbp = data['pbp']

        if errors:
            pbp_dict = {'pbp':pbp,
                        'errors':data['errors']}
            return pbp_dict
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

def nhl_scrape_prospects(team):
    #Given team abbreviation, retreive current team prospects

    api = f'https://api-web.nhle.com/v1/prospects/{team}'

    data = rs.get(api).json()
    
    #Iterate through positions
    players = [pd.json_normalize(data[pos]) for pos in ['forwards','defensemen','goalies']]

    prospects = pd.concat(players)
    #Add name columns
    prospects['fullName'] = (prospects['firstName.default']+" "+prospects['lastName.default']).str.upper()

    #Return: team prospects
    return prospects

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

def nhl_scrape_draft_rankings(arg = 'now', category = ''):
    #Given url argument for timeframe and prospect category, return draft rankings
    #Category 1 is North American Skaters
    #Category 2 is International Skaters
    #Category 3 is North American Goalie
    #Category 4 is International Goalie

    #Player category only applies when requesting a specific season
    api = f"https://api-web.nhle.com/v1/draft/rankings/{arg}/{category}" if category != "" else f"https://api-web.nhle.com/v1/draft/rankings/{arg}"
    data = pd.json_normalize(rs.get(api).json()['rankings'])

    #Add player name columns
    data['fullName'] = (data['firstName']+" "+data['lastName']).str.upper()

    #Return: prospect rankings
    return data

''' In Repair
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
'''

def ncaa_scrape_game(game_ids,remove=[]):
    #Given list of NCAA game id, return parsed play-by-play data
    # param 'game_ids' - NHL game ids (or list formatted as ['random', num_of_games, start_year, end_year])
    # param 'remove' - list of events to remove from final dataframe
    
    pbps = []
    for game_id in game_ids:
        api = f"https://ncaa-api.henrygd.me/game/{game_id}/play-by-play"
        scoring = f"https://ncaa-api.henrygd.me/game/{game_id}/scoring-summary"

        print(f'Scraping data from game {game_id}...',end="")
        start = time.perf_counter()
        try:
            #Retreive data
            data = rs.get(api).json()
            scores = rs.get(scoring).json()

            #Append data to list
            no_data = False
            pbps.append(ncaa_parse_json(game_id,data,scores))

            end = time.perf_counter()
            secs = end - start
            print(f" finished in {secs:.2f} seconds.")

        except:
            #Games such as the all-star game and pre-season games will incur this error
            print(f"\nUnable to scrape game {game_id}.  Ensure the ID is properly inputted and formatted.")
            no_data = True
            pbps.append(pd.DataFrame())

    #Add all pbps together
    df = pd.concat(pbps)

    #Return: complete play-by-play with data removed or split as necessary
    if no_data:
        return pd.DataFrame()
    else:
        return df.loc[~df['event_type'].isin(remove)]

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

def repo_load_pbp(seasons = []):
    #Returns play-by-play data from repository
    # param 'seasons' - list of seasons to include

    #Add parquet to total
    print(f'Loading play-by-play from the following seasons: {seasons}...')
    dfs = [pd.read_parquet(f"https://github.com/owensingh38/wsba_hockey/raw/refs/heads/main/src/wsba_hockey/pbp/parquet/nhl_pbp_{season}.parquet") for season in seasons]

    return pd.concat(dfs)
