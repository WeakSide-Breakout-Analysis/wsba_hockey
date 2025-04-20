import requests as rs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import time
import random
from .tools.scraping import *
from .tools.xg_model import *
from .tools.agg import *
from .tools.plotting import *

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

convert_seasons = {'2007': '20072008', 
                   '2008': '20082009', 
                   '2009': '20092010', 
                   '2010': '20102011', 
                   '2011': '20112012', 
                   '2012': '20122013', 
                   '2013': '20132014', 
                   '2014': '20142015', 
                   '2015': '20152016', 
                   '2016': '20162017', 
                   '2017': '20172018', 
                   '2018': '20182019', 
                   '2019': '20192020', 
                   '2020': '20202021', 
                   '2021': '20212022', 
                   '2022': '20222023', 
                   '2023': '20232024', 
                   '2024': '20242025'}

convert_team_abbr = {'L.A':'LAK',
                     'N.J':'NJD',
                     'S.J':'SJS',
                     'T.B':'TBL',
                     'PHX':'ARI'}

per_sixty = ['Fi','xGi','Gi','A1','A2','P1','P','FF','FA','xGF','xGA','GF','GA']

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
    '2018021133':'Game between Lightning and Capitals has incorrectly labeled event teams (i.e. WSH TAKEAWAY - #71 CIRELLI (Cirelli is a Tampa Bay skater in this game)).',
    '2019020876':'Due to the frightening collapse of Blues defensemen Jay Bouwmeester, a game on February 2nd, 2020 between the Ducks and Blues was postponed.  \nWhen the game resumed, Ducks defensemen Hampus Lindholm, who assisted on a goal in the inital game, did not play in the resumed match.'
}

name_change = {
    "":"",
}

shot_types = ['wrist','deflected','tip-in','slap','backhand','snap','wrap-around','poke','bat','cradle','between-legs']

new = 2024

standings_end = {
    '20072008':'04-06',
    '20082009':'04-12',
    '20092010':'04-11',
    '20102011':'04-10',
    '20112012':'04-07',
    '20122013':'04-28',
    '20132014':'04-13',
    '20142015':'04-11',
    '20152016':'04-10',
    '20162017':'04-09',
    '20172018':'04-08',
    '20182019':'04-06',
    '20192020':'03-11',
    '20202021':'05-19',
    '20212022':'04-01',
    '20222023':'04-14',
    '20232024':'04-18',
    '20242025':'04-17'
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
        pbp = data
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

def nhl_scrape_standings(arg = "now", season_type = 2):
    #Returns standings
    # param 'arg' - by default, this is "now" returning active NHL standings.  May also be a specific date formatted as YYYY-MM-DD, a season (scrapes the last standings date for the season) or a year (for playoffs).
    # param 'season_type' - by default, this scrapes the regular season standings.  If set to 3, it returns the playoff bracket for the specified season

    #arg param is ignored when set to "now" if season_type param is 3
    if season_type == 3:
        if arg == "now":
            arg = new

        print(f"Scraping playoff bracket for date: {arg}")
        api = f"https://api-web.nhle.com/v1/playoff-bracket/{arg}"

        data = rs.get(api).json()['series']

        return pd.json_normalize(data)

    else:
        if arg == "now":
            print("Scraping standings as of now...")
        elif arg in seasons:
            print(f'Scraping standings for season: {arg}')
        else:
            print(f"Scraping standings for date: {arg}")

        api = f"https://api-web.nhle.com/v1/standings/{arg[4:8]}-{standings_end[arg]}"
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

def nhl_scrape_team_info(country = False):
    #Given option to return franchise or country, return team information

    print('Scraping team information...')
    api = f'https://api.nhle.com/stats/rest/en/{'country' if country else 'team'}'
    
    data =  pd.json_normalize(rs.get(api).json()['data'])

    #Add logos if necessary
    if not country:
        data['logo_light'] = 'https://assets.nhle.com/logos/nhl/svg/'+data['triCode']+'_light.svg'
        data['logo_dark'] = 'https://assets.nhle.com/logos/nhl/svg/'+data['triCode']+'_dark.svg'

    return data.sort_values(by=(['country3Code','countryCode','iocCode','countryName'] if country else ['fullName','triCode','id']))

def nhl_scrape_player_data(player_id):
    #Given player id, return player information
    api = f'https://api-web.nhle.com/v1/player/{player_id}/landing'

    data = pd.json_normalize(rs.get(api).json())

    #Add name column
    data['fullName'] = (data['firstName.default'] + " " + data['lastName.default']).str.upper()

    #Return: player data
    return data

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

def nhl_shooting_impacts(agg,team=False):
    #Given stats table generated from the nhl_calculate_stats function, return table with shot impacts
    #Only 5v5 is supported as of now

    #param 'agg' - stats table
    #param 'team' - boolean determining if team stats are calculated instead of skater stats

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

    if team:
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
       
        #Add extra metrics
        pos['RushF/60'] = (pos['RushF']/pos['TOI'])*60
        pos['RushA/60'] = (pos['RushA']/pos['TOI'])*60
        pos['Rushes FF'] = pos['RushF/60'].rank(pct=True)
        pos['Rushes FA'] = pos['RushA/60'].rank(pct=True)
        pos['RushFxG/60'] = (pos['RushFxG']/pos['TOI'])*60
        pos['RushAxG/60'] = (pos['RushAxG']/pos['TOI'])*60
        pos['Rushes xGF'] = pos['RushFxG/60'].rank(pct=True)
        pos['Rushes xGA'] = pos['RushAxG/60'].rank(pct=True)
        pos['RushFG/60'] = (pos['RushFG']/pos['TOI'])*60
        pos['RushAG/60'] = (pos['RushAG']/pos['TOI'])*60
        pos['Rushes GF'] = pos['RushFG/60'].rank(pct=True)
        pos['Rushes GA'] = pos['RushAG/60'].rank(pct=True)

        #Flip against metric percentiles
        pos['ODEF-SR'] = 1-pos['ODEF-SR']
        pos['ODEF-SQ'] = 1-pos['ODEF-SQ']
        pos['ODEF-FN'] = 1-pos['ODEF-FN']

        #Return: team stats with shooting impacts
        return pos.drop(columns=['fsh','fenwick','xg_fen','xg','g','finishing']).sort_values(['Season','Team'])


    else:
        #Remove skaters with less than 150 minutes of TOI then split between forwards and dmen
        agg = agg.loc[agg['TOI']>=150]
        forwards = agg.loc[agg['Position']!='D']
        defensemen = agg.loc[agg['Position']=='D']

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

            #Calculate On-Ice Involvement Percentiles
            pos['Fenwick'] = pos['FC%'].rank(pct=True)
            pos['xG'] = pos['xGC%'].rank(pct=True)
            pos['Goal Factor'] = pos['GI%'].rank(pct=True)
            pos['Goal Scoring'] = pos['GC%'].rank(pct=True)
            pos['Rush/60'] = (pos['Rush']/pos['TOI'])*60
            pos['RushxG/60'] = (pos['Rush xG']/pos['TOI'])*60
            pos['Rushes xG'] = pos['RushxG/60'].rank(pct=True)
            pos['Rushes FF'] = pos['Rush/60'].rank(pct=True)

        #Add positions back together
        complete = pd.concat([forwards,defensemen])

        #Flip against metric percentiles
        complete['ODEF-SR'] = 1-complete['ODEF-SR']
        complete['ODEF-SQ'] = 1-complete['ODEF-SQ']
        complete['ODEF-FN'] = 1-complete['ODEF-FN']

        #Extraneous Values
        complete['Extraneous Gi'] = complete['INDV-SRI']+complete['INDV-SQI']+complete['INDV-FNI']
        complete['Extraneous xGi'] = complete['INDV-SRI']+complete['INDV-SQI']
        complete['Extraneous GF'] = complete['OOFF-SRI']+complete['OOFF-SQI']+complete['OOFF-FNI']
        complete['Extraneous xGF'] = complete['OOFF-SRI']+complete['OOFF-SQI']
        complete['Extraneous GA'] = complete['ODEF-SRI']+complete['ODEF-SQI']+complete['ODEF-FNI']
        complete['Extraneous xGA'] = complete['ODEF-SRI']+complete['ODEF-SQI']

        #Goal Composites
        complete['Linemate Extraneous Goals'] = complete['Extraneous GF'] - complete['Extraneous Gi']
        complete['Linemate Goal Induction'] = complete['Linemate Extraneous Goals']*complete['AC%']
        complete['Composite Goal Impact'] = complete['Extraneous Gi'] + complete['Linemate Goal Induction'] 
        complete['Linemate Rel. Goal Impact'] = complete['Composite Goal Impact'] - (complete['Extraneous GF']-complete['Composite Goal Impact'])
        complete['Net Goal Impact'] = complete['Extraneous GF'] - complete['Extraneous GA']
        complete['Net xGoal Impact'] = complete['Extraneous xGF'] - complete['Extraneous xGA']

        #Return: skater stats with shooting impacts
        return complete.drop(columns=['fsh','fenwick','xg_fen','xg','g','finishing']).sort_values(['Player','Season','Team','ID'])

def nhl_calculate_stats(pbp,type,season_types,game_strength,roster_path="rosters/nhl_rosters.csv",xg="moneypuck",shot_impact=False):
    #Given play-by-play, seasonal information, game_strength, rosters, and xG model, return aggregated stats
    # param 'pbp' - play-by-play dataframe
    # param 'type' - type of stats to calculate ('skater', 'goaltender', or 'team')
    # param 'season' - season or timeframe of events in play-by-play
    # param 'season_type' - list of season types (preseason, regular season, or playoffs) to include in aggregation
    # param 'game_strength' - list of game_strengths to include in aggregation
    # param 'roster_path' - path to roster file
    # param 'xg' - xG model to apply to pbp for aggregation
    # param 'shot_impact' - boolean determining if the shot impact model will be applied to the dataset

    print(f"Calculating statistics for all games in the provided play-by-play data...\nSeasons included: {pbp['season'].drop_duplicates().to_list()}...")
    start = time.perf_counter()

    #Add extra data and apply team changes
    pbp = prep_xG_data(pbp).replace(convert_team_abbr)

    #Check if xG column exists and apply model if it does not
    try:
        pbp['xG']
    except KeyError:
        if xg == 'wsba':
            pbp = wsba_xG(pbp)
        else:
            pbp = moneypuck_xG(pbp)

    #Filter by season types and remove shootouts
    pbp = pbp.loc[(pbp['season_type'].isin(season_types)) & (pbp['period'] < 5)]
    
    # Filter by game strength if not "all"
    if game_strength != "all":
        pbp = pbp.loc[pbp['strength_state'].isin(game_strength)]

    #Split calculation
    if type == 'team':
        complete = calc_team(pbp)

        #Set TOI to minute
        complete['TOI'] = complete['TOI']/60

        #Add per 60 stats
        for stat in per_sixty[7:13]:
            complete[f'{stat}/60'] = (complete[stat]/complete['TOI'])*60

        end = time.perf_counter()
        length = end-start
        print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')
        #Apply shot impacts if necessary (Note: this will remove skaters with fewer than 150 minutes of TOI due to the shot impact TOI rule)
        if shot_impact:
            return nhl_shooting_impacts(complete,True)
        else:
            return complete
    else:
        indv_stats = calc_indv(pbp)
        onice_stats = calc_onice(pbp)

        #IDs sometimes set as objects
        indv_stats['ID'] = indv_stats['ID'].astype(float)
        onice_stats['ID'] = onice_stats['ID'].astype(float)

        #Merge and add columns for extra stats
        complete = pd.merge(indv_stats,onice_stats,how="outer",on=['ID','Team','Season'])
        complete['GC%'] = complete['Gi']/complete['GF']
        complete['AC%'] = (complete['A1']+complete['A2'])/complete['GF']
        complete['GI%'] = (complete['Gi']+complete['A1']+complete['A2'])/complete['GF']
        complete['FC%'] = complete['Fi']/complete['FF']
        complete['xGC%'] = complete['xGi']/complete['xGF']

        #Remove entries with no ID listed
        complete = complete.loc[complete['ID'].notna()]

        #Import rosters and player info
        rosters = pd.read_csv(roster_path)
        names = rosters[['id','fullName',
                            'headshot','positionCode','shootsCatches',
                            'heightInInches','weightInPounds',
                            'birthDate','birthCountry']].drop_duplicates(subset=['id','fullName'],keep='last')

        #Add names
        complete = pd.merge(complete,names,how='left',left_on='ID',right_on='id')

        #Rename if there are no missing names
        complete = complete.rename(columns={'fullName':'Player',
                                            'headshot':'Headshot',
                                            'positionCode':'Position',
                                            'shootsCatches':'Handedness',
                                            'heightInInches':'Height (in)',
                                            'weightInPounds':'Weight (lbs)',
                                            'birthDate':'Birthday',
                                            'birthCountry':'Nationality'})

        #Set TOI to minute
        complete['TOI'] = complete['TOI']/60

        #Add per 60 stats
        for stat in per_sixty:
            complete[f'{stat}/60'] = (complete[stat]/complete['TOI'])*60

        #Add player age
        complete['Birthday'] = pd.to_datetime(complete['Birthday'])
        complete['season_year'] = complete['Season'].astype(str).str[4:8].astype(int)
        complete['Age'] = complete['season_year'] - complete['Birthday'].dt.year

        #Find player headshot
        complete['Headshot'] = 'https://assets.nhle.com/mugs/nhl/'+complete['Season'].astype(str)+'/'+complete['Team']+'/'+complete['ID'].astype(int).astype(str)+'.png'

        end = time.perf_counter()
        length = end-start
        #Remove goalies that occasionally appear in a set
        complete = complete.loc[complete['Position']!='G']
        #Add WSBA ID
        complete['WSBA'] = complete['Player']+complete['Season'].astype(str)+complete['Team']

        #Shot Type Metrics
        type_metrics = []
        for type in shot_types:
            for stat in per_sixty[:3]:
                type_metrics.append(f'{type.capitalize()}{stat}')

        complete = complete[[
            'Player','ID',
            "Season","Team",'WSBA',
            'Headshot','Position','Handedness',
            'Height (in)','Weight (lbs)',
            'Birthday','Age','Nationality',
            'GP','TOI',
            "Gi","A1","A2",'P1','P',
            "Fi","xGi",'xGi/Fi',"Gi/xGi","Fshi%",
            "GF","FF","xGF","xGF/FF","GF/xGF","FshF%",
            "GA","FA","xGA","xGA/FA","GA/xGA","FshA%",
            'Rush',"Rush xG",'Rush G',"GC%","AC%","GI%","FC%","xGC%",
        ]+[f'{stat}/60' for stat in per_sixty]+type_metrics].fillna(0).sort_values(['Player','Season','Team','ID'])
        
        print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')
        #Apply shot impacts if necessary (Note: this will remove skaters with fewer than 150 minutes of TOI due to the shot impact TOI rule)
        if shot_impact:
            return nhl_shooting_impacts(complete,False)
        else:
            return complete

def nhl_plot_skaters_shots(pbp,skater_dict,strengths,marker_dict=event_markers,onice = 'indv',title = True,legend=False,xg='moneypuck'):
    #Returns list of plots for specified skaters
    # param 'pbp' - pbp to plot data
    # param 'skater_dict' - skaters to plot shots for (format: {'Patrice Bergeron':['20242025','BOS']})
    # param 'strengths' - strengths to include in plotting
    # param 'marker_dict' - dict with markers to use for events
    # param 'onice' - can set which shots to include in plotting for the specified skater ('indv', 'for', 'against')
    # param 'title' - bool including title when true
    # param 'legend' - bool which includes legend if true
    # param 'xg' - xG model to apply to pbp for plotting

    print(f'Plotting the following skater shots: {skater_dict}...')

    #Iterate through games, adding plot to list
    skater_plots = []
    for skater in skater_dict.keys():
        skater_info = skater_dict[skater]
        title = f'{skater} Fenwick Shots for {skater_info[1]} in {skater_info[0][2:4]}-{skater_info[0][6:8]}' if title else ''
        skater_plots.append(plot_skater_shots(pbp,skater,skater_info[0],skater_info[1],strengths,title,marker_dict,onice,legend,xg))

    #Return: list of plotted skater shot charts
    return skater_plots

def nhl_plot_games(pbp,events,strengths,game_ids='all',marker_dict=event_markers,legend=False,xg='moneypuck'):
    #Returns list of plots for specified games
    # param 'pbp' - pbp to plot data
    # param 'events' - type of events to plot
    # param 'strengths' - strengths to include in plotting
    # param 'game_ids' - games to plot (list if not set to 'all')
    # param 'marker_dict' - dict with colors to use for events
    # param 'legend' - bool which includes legend if true
    # param 'xg' - xG model to apply to pbp for plotting

    #Find games to scrape
    if game_ids == 'all':
        game_ids = pbp['game_id'].drop_duplicates().to_list()

    print(f'Plotting the following games: {game_ids}...')

    #Iterate through games, adding plot to list
    game_plots = [plot_game_events(pbp,game,events,strengths,marker_dict,legend,xg) for game in game_ids]

    #Return: list of plotted game events
    return game_plots

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

def repo_load_seasons():
    #List of available seasons to scrape

    return seasons

def admin_convert_to_parquet(seasons):
    for season in seasons:
        load = pd.read_csv(f'pbp/csv/nhl_pbp_{season}.csv')

        load.to_parquet(f'pbp/parquet/nhl_pbp_{season}.parquet',index=False)
