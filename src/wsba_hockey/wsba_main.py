import requests as rs
import pandas as pd
import time
import random
from datetime import datetime, timedelta, date
from tools.scraping import *
from tools.xg_model import *
from tools.agg import *
from tools.plotting import *

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

per_sixty = ['Fi','xGi','Gi','A1','A2','P1','P','OZF','NZF','DZF','FF','FA','xGF','xGA','GF','GA','CF','CA','HF','HA','Give','Take','Penl','Penl2','Penl5','Draw','Block']

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
    '2012020018':'HTML events contain mislabeled events.',
    '2013020971':'On March 10th, 2014, Stars forward Rich Peverley suffered from a cardiac episode midgame and as a result, the remainder of the game was postponed.  \nThe game resumed on April 9th, and the only goal scorer in the game, Blue Jackets forward Nathan Horton, did not appear in the resumed game due to injury.  Interestingly, Horton would never play in the NHL again.',
    '2018021133':'Game between Lightning and Capitals has incorrectly labeled event teams (i.e. WSH TAKEAWAY - #71 CIRELLI (Cirelli is a Tampa Bay skater in this game)).',
    '2019020876':'Due to the frightening collapse of Blues defensemen Jay Bouwmeester, a game on February 2nd, 2020 between the Ducks and Blues was postponed.  \nWhen the game resumed, Ducks defensemen Hampus Lindholm, who assisted on a goal in the inital game, did not play in the resumed match.'
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

events = ['faceoff','hit','giveaway','takeaway','blocked-shot','missed-shot','shot-on-goal','goal','penalty']

## SCRAPE FUNCTIONS ##
def nhl_scrape_game(game_ids,split_shifts = False, remove = ['period-start','period-end','challenge','stoppage','shootout-complete','game-end'],verbose = False, sources = False, errors = False):
    #Given a set of game_ids (NHL API), return complete play-by-play information as requested
    # param 'game_ids' - NHL game ids (or list formatted as ['random', num_of_games, start_year, end_year])
    # param 'split_shifts' - boolean which splits pbp and shift events if true
    # param 'remove' - list of events to remove from final dataframe
    # param 'xg' - xG model to apply to pbp for aggregation
    # param 'verbose' - boolean which adds additional event info if true
    # param 'sources - boolean scraping the html and json sources to a master directory if true 
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
    prog = 0
    for game_id in game_ids:
        print("Scraping data from game " + str(game_id) + "...",end="")
        start = time.perf_counter()

        try:
            #Retrieve data
            info = get_game_info(game_id)
            data = combine_data(info, sources)
                
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
        #For each day, call NHL api and retreive info on all games of selected game
        inc = start+timedelta(days=i)
        print("Scraping games on " + str(inc)[:10]+"...")
        
        get = rs.get(api+str(inc)[:10]).json()
        gameWeek = pd.json_normalize(list(pd.json_normalize(get['gameWeek'])['games'])[0])
        
        #Return nothing if there's nothing
        if gameWeek.empty:
            game.append(gameWeek)
        else:
            gameWeek['date'] = get['gameWeek'][0]['date']

            gameWeek['season_type'] = gameWeek['gameType']
            gameWeek['away_team_abbr'] = gameWeek['awayTeam.abbrev']
            gameWeek['home_team_abbr'] = gameWeek['homeTeam.abbrev']
            gameWeek['game_title'] = gameWeek['away_team_abbr'] + " @ " + gameWeek['home_team_abbr'] + " - " + gameWeek['date']
            
            front_col = ['id','season','date','season_type','game_title','away_team_abbr','home_team_abbr']
            gameWeek = gameWeek[front_col+[col for col in gameWeek.columns.to_list() if col not in front_col]]

        game.append(gameWeek)
        
    #Concatenate all games
    df = pd.concat(game)
    
    #Return: specificed schedule data
    return df

def nhl_scrape_season(season,split_shifts = False, season_types = [2,3], remove = ['period-start','period-end','game-end','challenge','stoppage'], start = "09-01", end = "08-01", local=False, local_path = "schedule/schedule.csv", verbose = False, sources = False, errors = False):
    #Given season, scrape all play-by-play occuring within the season
    # param 'season' - NHL season to scrape
    # param 'split_shifts' - boolean which splits pbp and shift events if true
    # param 'remove' - list of events to remove from final dataframe
    # param 'start' - Start date in season
    # param 'end' - End date in season
    # param 'local' - boolean indicating whether to use local file to scrape game_ids
    # param 'local_path' - path of local file
    # param 'verbose' - boolean which adds additional event info if true
    # param 'sources - boolean scraping the html and json sources to a master directory if true 
    # param 'errors' - boolean returning game ids which did not scrape if true

    #Determine whether to use schedule data in repository or to scrape
    if local:
        load = pd.read_csv(local_path)
        load['date'] = pd.to_datetime(load['date'])
        
        start = f'{(season[0:4] if int(start[0:2])>=9 else season[4:8])}-{int(start[0:2])}-{int(start[3:5])}'
        end =  f'{(season[0:4] if int(end[0:2])>=9 else season[4:8])}-{int(end[0:2])}-{int(end[3:5])}'
        
        load = load.loc[(load['season'].astype(str)==season)&
                        (load['season_type'].isin(season_types))&
                        (load['date']>=start)&(load['date']<=end)]
        
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
    if split_shifts:
        data = nhl_scrape_game(game_ids,split_shifts=True,remove=remove,verbose=verbose,sources=sources,errors=errors)
    else:
        data = nhl_scrape_game(game_ids,remove=remove,verbose=verbose,sources=sources,errors=errors)
    
    end = time.perf_counter()
    secs = end - start
    
    print(f'Finished season scrape in {(secs/60)/60:.2f} hours.')
    #Return: Complete pbp and shifts data for specified season as well as dataframe of game_ids which failed to return data
    return data

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

def nhl_scrape_player_data(player_ids):
    #Given player id, return player information
    infos = []
    for player_id in player_ids:
        player_id = int(player_id)
        api = f'https://api-web.nhle.com/v1/player/{player_id}/landing'

        data = pd.json_normalize(rs.get(api).json())

        #Add name column
        data['fullName'] = (data['firstName.default'] + " " + data['lastName.default']).str.upper()

        #Append
        infos.append(data)

    df = pd.concat(infos)
    #Return: player data
    return df

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

def nhl_shooting_impacts(agg,type):
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
        for stat in per_sixty[10:len(per_sixty)]:
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
            for stat in per_sixty:
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

def nhl_calculate_stats(pbp,type,season_types,game_strength,split_game=False,roster_path="rosters/nhl_rosters.csv",shot_impact=False):
    #Given play-by-play, seasonal information, game_strength, rosters, and xG model, return aggregated stats
    # param 'pbp' - play-by-play dataframe
    # param 'type' - type of stats to calculate ('skater', 'goalie', or 'team')
    # param 'season' - season or timeframe of events in play-by-play
    # param 'season_type' - list of season types (preseason, regular season, or playoffs) to include in aggregation
    # param 'game_strength' - list of game_strengths to include in aggregation
    # param 'split_game' - boolean which if true groups aggregation by game
    # param 'roster_path' - path to roster file
    # param 'shot_impact' - boolean determining if the shot impact model will be applied to the dataset

    print(f"Calculating statistics for all games in the provided play-by-play data at {game_strength} for {type}s...\nSeasons included: {pbp['season'].drop_duplicates().to_list()}...")
    start = time.perf_counter()

    #Check if xG column exists and apply model if it does not
    try:
        pbp['xG']
    except KeyError: 
        pbp = wsba_xG(pbp)

    #Filter by season types, remove shootouts, remove shots with no coordinates, and remove shots on empty nets
    pbp_noshot = pbp.loc[(pbp['season_type'].isin(season_types)) & ~(pbp['event_type'].isin(fenwick_events))]
    pbp_shot = pbp.loc[(pbp['season_type'].isin(season_types)) & (pbp['event_type'].isin(fenwick_events)) & (pbp['empty_net']<1) & (pbp['x'].notna()) & (pbp['y'].notna())]

    pbp = pd.concat([pbp_shot,pbp_noshot])

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
    if type == 'goalie':
        complete = calc_goalie(pbp,game_strength,second_group)

        #Set TOI to minute
        complete['TOI'] = complete['TOI']/60

        #Add per 60 stats
        for stat in ['FF','FA','xGF','xGA','GF','GA','CF','CA','GSAx']:
            complete[f'{stat}/60'] = (complete[stat]/complete['TOI'])*60
            
        complete['GF%'] = complete['GF']/(complete['GF']+complete['GA'])
        complete['xGF%'] = complete['xGF']/(complete['xGF']+complete['xGA'])
        complete['FF%'] = complete['FF']/(complete['FF']+complete['FA'])
        complete['CF%'] = complete['CF']/(complete['CF']+complete['CA'])

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
        complete = complete.rename(columns={'fullName':'Goalie',
                                            'headshot':'Headshot',
                                            'positionCode':'Position',
                                            'shootsCatches':'Handedness',
                                            'heightInInches':'Height (in)',
                                            'weightInPounds':'Weight (lbs)',
                                            'birthDate':'Birthday',
                                            'birthCountry':'Nationality'})
        
        #WSBA
        complete['WSBA'] = complete['Goalie']+complete['Team']+complete['Season'].astype(str)

        #Add player age
        complete['Birthday'] = pd.to_datetime(complete['Birthday'])
        complete['season_year'] = complete['Season'].astype(str).str[4:8].astype(int)
        complete['Age'] = complete['season_year'] - complete['Birthday'].dt.year

        #Find player headshot
        complete['Headshot'] = 'https://assets.nhle.com/mugs/nhl/'+complete['Season'].astype(str)+'/'+complete['Team']+'/'+complete['ID'].astype(int).astype(str)+'.png'

        end = time.perf_counter()
        length = end-start
        print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')

        head = ['Goalie','ID','Game'] if 'Game' in complete.columns else ['Goalie','ID']
        complete = complete[head+[
            "Season","Team",'WSBA',
            'Headshot','Position','Handedness',
            'Height (in)','Weight (lbs)',
            'Birthday','Age','Nationality',
            'GP','TOI',
            "GF","FF","xGF","xGF/FF","GF/xGF","FshF%",
            "GA","FA","xGA","xGA/FA","GA/xGA","FshA%",
            'CF','CA',
            'GSAx',
            'RushF','RushA','RushFxG','RushAxG','RushFG','RushAG'
        ]+[f'{stat}/60' for stat in ['FF','FA','xGF','xGA','GF','GA','CF','CA','GSAx']]]

        #Apply shot impacts if necessary
        if shot_impact:
            complete = nhl_shooting_impacts(complete,'goalie')
        
        end = time.perf_counter()
        length = end-start
        print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')

        return complete
        
    elif type == 'team':
        complete = calc_team(pbp,game_strength,second_group)

        #WSBA
        complete['WSBA'] = complete['Team']+complete['Season'].astype(str)

        #Set TOI to minute
        complete['TOI'] = complete['TOI']/60

        #Add per 60 stats
        for stat in per_sixty[10:len(per_sixty)]:
            complete[f'{stat}/60'] = (complete[stat]/complete['TOI'])*60
            
        complete['GF%'] = complete['GF']/(complete['GF']+complete['GA'])
        complete['xGF%'] = complete['xGF']/(complete['xGF']+complete['xGA'])
        complete['FF%'] = complete['FF']/(complete['FF']+complete['FA'])
        complete['CF%'] = complete['CF']/(complete['CF']+complete['CA'])

        head = ['Team','Game'] if 'Game' in complete.columns else ['Team']
        complete = complete[head+[
            'Season','WSBA',
            'GP','TOI',
            "GF","FF","xGF","xGF/FF","GF/xGF","FshF%",
            "GA","FA","xGA","xGA/FA","GA/xGA","FshA%",
            'CF','CA',
            'GF%','FF%','xGF%','CF%',
            'HF','HA','HF%',
            'Penl','Penl2','Penl5','PIM','Draw','PENL%',
            'Give','Take','PM%',
            'Block',
            'RushF','RushA','RushFxG','RushAxG','RushFG','RushAG'
        ]+[f'{stat}/60' for stat in per_sixty[10:len(per_sixty)]]]
        #Apply shot impacts if necessary
        if shot_impact:
            complete = nhl_shooting_impacts(complete,'team')
        
        end = time.perf_counter()
        length = end-start
        print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')

        return complete
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
        complete['xGF%'] = complete['xGF']/(complete['xGF']+complete['xGA'])
        complete['FF%'] = complete['FF']/(complete['FF']+complete['FA'])
        complete['CF%'] = complete['CF']/(complete['CF']+complete['CA'])

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

        #Add player age
        complete['Birthday'] = pd.to_datetime(complete['Birthday'])
        complete['season_year'] = complete['Season'].astype(str).str[4:8].astype(int)
        complete['Age'] = complete['season_year'] - complete['Birthday'].dt.year

        #Find player headshot
        complete['Headshot'] = 'https://assets.nhle.com/mugs/nhl/'+complete['Season'].astype(str)+'/'+complete['Team']+'/'+complete['ID'].astype(int).astype(str)+'.png'

        #Remove goalies that occasionally appear in a set
        complete = complete.loc[complete['Position']!='G']
        #Add WSBA ID
        complete['WSBA'] = complete['Player']+complete['Season'].astype(str)+complete['Team']

        #Add per 60 stats
        for stat in per_sixty:
            complete[f'{stat}/60'] = (complete[stat]/complete['TOI'])*60

        #Shot Type Metrics
        type_metrics = []
        for type in shot_types:
            for stat in per_sixty[:3]:
                type_metrics.append(f'{type.capitalize()}{stat}')

        head = ['Player','ID','Game'] if 'Game' in complete.columns else ['Player','ID']
        complete = complete[head+[
            "Season","Team",'WSBA',
            'Headshot','Position','Handedness',
            'Height (in)','Weight (lbs)',
            'Birthday','Age','Nationality',
            'GP','TOI',
            "Gi","A1","A2",'P1','P',
            'Give','Take','PM%','HF','HA','HF%',
            "Fi","xGi",'xGi/Fi',"Gi/xGi","Fshi%",
            "GF","FF","xGF","xGF/FF","GF/xGF","FshF%",
            "GA","FA","xGA","xGA/FA","GA/xGA","FshA%",
            'Ci','CF','CA','CF%',
            'FF%','xGF%','GF%',
            'Rush',"Rush xG",'Rush G',"GC%","AC%","GI%","FC%","xGC%",
            'F','FW','FL','F%',
            'Penl','Penl2','Penl5',
            'Draw','PIM','PENL%',
            'Block',
            'OZF','NZF','DZF',
            'OZF%','NZF%','DZF%',
        ]+[f'{stat}/60' for stat in per_sixty]+type_metrics].fillna(0).sort_values(['Player','Season','Team','ID'])
        
        #Apply shot impacts if necessary (Note: this will remove skaters with fewer than 150 minutes of TOI due to the shot impact TOI rule)
        if shot_impact:
            complete = nhl_shooting_impacts(complete,'skater')
        
        end = time.perf_counter()
        length = end-start
        print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')

        return complete

def nhl_plot_skaters_shots(pbp,skater_dict,strengths,marker_dict=event_markers,onice = 'indv',title = True,legend=False):
    #Returns dict of plots for specified skaters
    # param 'pbp' - pbp to plot data
    # param 'skater_dict' - skaters to plot shots for (format: {'Patrice Bergeron':['20242025','BOS']})
    # param 'strengths' - strengths to include in plotting
    # param 'marker_dict' - dict with markers to use for events
    # param 'onice' - can set which shots to include in plotting for the specified skater ('indv', 'for', 'against')
    # param 'title' - bool including title when true
    # param 'legend' - bool which includes legend if true
    # param 'xg' - xG model to apply to pbp for plotting

    print(f'Plotting the following skater shots: {skater_dict}...')

    #Iterate through skaters, adding plots to dict
    skater_plots = {}
    for skater in skater_dict.keys():
        skater_info = skater_dict[skater]
        title = f'{skater} Fenwick Shots for {skater_info[1]} in {skater_info[0][2:4]}-{skater_info[0][6:8]}' if title else ''
        #Key is formatted as PLAYERSEASONTEAM (i.e. PATRICE BERGERON20212022BOS)
        skater_plots.update({f'{skater}{skater_info[0]}{skater_info[1]}':[plot_skater_shots(pbp,skater,skater_info[0],skater_info[1],strengths,title,marker_dict,onice,legend)]})

    #Return: list of plotted skater shot charts
    return skater_plots

def nhl_plot_games(pbp,events,strengths,game_ids='all',marker_dict=event_markers,team_colors={'away':'primary','home':'primary'},legend=False):
    #Returns dict of plots for specified games
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

    game_plots = {}
    #Iterate through games, adding plot to dict
    for game in game_ids:
        game_plots.update({game:[plot_game_events(pbp,game,events,strengths,marker_dict,team_colors,legend)]})

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
    dfs = [pd.read_parquet(f"https://f005.backblazeb2.com/file/weakside-breakout/pbp/{season}.parquet") for season in seasons]

    return pd.concat(dfs)

def repo_load_seasons():
    #List of available seasons to scrape

    return seasons
