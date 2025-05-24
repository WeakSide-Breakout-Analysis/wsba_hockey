import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plot as wsba_plt
import matplotlib.pyplot as plt
import var
from datetime import datetime
from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_widget 

#Sidebar config
with ui.sidebar():
    ui.input_date('game_date','Date')
    ui.input_selectize('game_select',
                      'Select a game:',
                      [])

@reactive.calc
def df():
    #Determine which season to load based on the input date
    #Adjust the front-year and append the back-year to create the season to load
    front_year = (input.game_date().year-1) if input.game_date().month < 8 else input.game_date().year
    season = f'{front_year}{front_year+1}'

    #Load appropriate dataframe
    return pd.read_parquet(f'pbp/parquet/nhl_pbp_{season}.parquet')

@reactive.effect
@reactive.event(input.game_date)
def _():
    #Find games based on date

    #Configure selectable games
    load = df()
    games = load.loc[load['game_date'].astype(str)==str(input.game_date()),['game_title','game_id']].drop_duplicates().set_index('game_id').to_dict()['game_title']
    
    #Update the selectize with games from selected date
    ui.update_selectize('game_select',choices=games)

@reactive.event(input.game_select)
def plays():
    #Return plays from selected game
    pbp = df()
    return pbp.loc[pbp['game_id'].astype(str)==str(input.game_select())]

@render_widget
def plot_game():
    df = plays()

    if df.empty :
        return wsba_plt.wsba_rink()
    
    else:
        df = wsba_plt.prep(df,events=var.fenwick_events)
        game_title = df['game_title'].to_list()[0]
        colors = wsba_plt.colors(df)

        rink = wsba_plt.wsba_rink()

        plot = px.scatter(df,x='x',y='y',
                        size='size',color='Team',
                        color_discrete_map=colors,
                        hover_name='Description',
                        hover_data=['x','y',
                                    'Event Distance from Attacking Net',
                                    'Event Angle to Attacking Net',
                                    'xG'])
        
        for trace in plot.data:
            rink.add_trace(trace)
            
        return rink.update_layout(
                title=dict(text=game_title,
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top',
                ),
                
                legend=dict(
                    orientation='h',
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top',
                )
                )