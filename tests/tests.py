import pandas as pd
import matplotlib.pyplot as plt
import wsba_hockey as wsba

### WSBA HOCKEY ###
## Provided below are some tests of package capabilities

#Standings Scraping
wsba.nhl_scrape_standings(20222023).to_csv('samples/sample_standings.csv',index=False)

#Play-by-Play Scraping
pbp = wsba.nhl_scrape_game(['random',3,2010,2024],remove=[])
pbp.to_csv('tests/samples/sample_pbp.csv',index=False)

#Sample skater stats for game
wsba.nhl_calculate_stats(pbp,type='skater',season_types=[2],game_strength=['5v5']).to_csv('tests/samples/sample_stats.csv',index=False)

#Plot shots in games from test data
plots = wsba.nhl_plot_games(pbp,['shot-on-goal','missed-shot','goal'],['5v5'])

for game_id,plot in plots.items():
    plot[0].savefig(f'tests/samples/plots/{game_id}.png')
    