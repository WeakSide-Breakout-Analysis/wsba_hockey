import os
import pandas as pd
import matplotlib.pyplot as plt
import wsba_hockey as wsba

### WSBA HOCKEY ###
## Provided below are some tests of package capabilities

dir = os.path.dirname(os.path.realpath(__file__))

#Test scrape of random games
wsba.nhl_scrape_game(['random',1,2007,2024]).to_csv(f'{dir}/samples/sample_random_game.csv',index=False)

#Standings Scraping
wsba.nhl_scrape_standings(20222023).to_csv(f'{dir}/samples/sample_standings.csv',index=False)

#WSBA_Database Testing
# Play-by-Play Scraping
# Sample skater stats for game
# Plot shots in games from test data

db = wsba.NHL_Database('sample_db')
db.add_games([2021020045])
db.add_stats('sample_skater_stats','skater',[2,3],['5v5'])
db.add_game_plots(['missed-shot','shot-on-goal','goal'],['5v5'])
db.export_data(f'{dir}/samples/')