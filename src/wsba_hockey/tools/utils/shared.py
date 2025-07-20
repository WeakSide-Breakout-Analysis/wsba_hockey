import os
import time
import json
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup, SoupStrainer

## SHARED FUCNCTIONS ##
# Most code in this file originates (entirely or partially) from the hockey_scraper package by Harry Shomer #

dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir, "team_tri_codes.json"), "r" ,encoding="utf-8") as f:
    TEAMS = json.load(f)['teams']

def get_team(team):
    #Parse team header in HTML
    return TEAMS.get(team.upper(), team.upper()).upper()
    
def convert_to_seconds(minutes):
    #Convert time formatted as MM:SS in a period to raw seconds
    if minutes == '-16:0-':
        return '1200'      #Sometimes in the html at the end of the game the time is -16:0-

    #Validate time (invalid times are generally ignored)
    try:
        x = time.strptime(minutes.strip(' '), '%M:%S')
    except ValueError:
        return None

    return timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()

def get_contents(game_html):
    #Parse NHL HTML PBP document
    parsers = ["html5lib", "lxml", "html.parser"]
    strainer = SoupStrainer('td', attrs={'class': re.compile(r'bborder')})

    for parser in parsers:
        # parse_only only works with lxml for some reason
        if parser == "lxml":
            soup = BeautifulSoup(game_html, parser, parse_only=strainer)
        else:
            soup = BeautifulSoup(game_html, parser)

        tds = soup.find_all("td", {"class": re.compile('.*bborder.*')})

        if len(tds) > 0:
            break

    return tds

def get_soup(shifts_html):
    #Convert html document to soup
    parsers = ["lxml", "html.parser", "html5lib"]

    for parser in parsers:
        soup = BeautifulSoup(shifts_html, parser)
        td = soup.findAll(True, {'class': ['playerHeading + border', 'lborder + bborder']})

        if len(td) > 0:
            break

    return td, get_teams(soup)

def get_teams(soup):
    #Find and return list of teams a given document's match (for HTML shifts parsing)
    team = soup.find('td', class_='teamHeading + border')  # Team for shifts
    team = team.get_text()

    #Find home team
    teams = soup.find_all('td', {'align': 'center', 'style': 'font-size: 10px;font-weight:bold'})
    regex = re.compile(r'>(.*)<br/?>')
    home_team = regex.findall(str(teams[7]))

    return [team, home_team[0]]