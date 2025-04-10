import requests as rs
import pandas as pd
import numpy as np
import re
import json

### SCRAPING FUNCTIONS ###
# Provided in this file are functions vital to the scraping functions in the WSBA Hockey Python package. #
# Functions in this file scrape from the HockeyTech APIs, which host leagues such as the AHL, PWHL, OHL, etc. #

## ORDER OF OPERATIONS ##
# Convert JSON text to usable format
# Retreive and parse JSON

## GLOBAL VARIABLES ##
# Different leagues within the HockeyTech APIs have unique keys
keys = {'AHL':'50c2cd9b5e18e390',
        'PWHL':'694cfeed58c932ee'}

def parse_json(game_id,league):
    #Given game_id and league, return game information
    api = f'https://lscluster.hockeytech.com/feed/index.php?feed=statviewfeed&view=gameCenterPlayByPlay&game_id={game_id}&key={keys[league]}&client_code={league.lower()}&lang=en&league_id=&callback=angular.callbacks._8'
    
    #Retrieve data
    data = rs.get(api).text

    #Remove the text header displaying callbacks
    pattern = r'angular\.callbacks\._\d+\('
    json_str = re.sub(pattern, '', data).rstrip(');')
    pbp_json = json.loads(json_str)

    #Loop through events, collecting and parsing as much necessary data as possible
    event_log = []
    events = pbp_json
    for event in events:
        events_dict = {}
        event_log.append(pd.DataFrame(events_dict))
    
    data = pd.concat(event_log)
    return data                          

    

parse_json(1027549,'AHL').to_csv('ahl_data.csv',index=False)