import pandas as pd
import matplotlib.pyplot as plt
import wsba_hockey as wsba

### WSBA HOCKEY ###
## Provided below are some tests of package capabilities

#Standings Scraping
wsba.nhl_scrape_standings(20222023).to_csv('tests/samples/sample_standings.csv',index=False)

#WSBA_Database Testing
# Play-by-Play Scraping
# Sample skater stats for game
# Plot shots in games from test data

db = wsba.NHL_Database('sample_db')
db.add_games([2021020045])
db.add_stats('sample_skater_stats','skater',[2,3],['5v5'])
db.add_game_plots(['missed-shot','shot-on-goal','goal'],['5v5'])
db.export_data('tests/samples/')