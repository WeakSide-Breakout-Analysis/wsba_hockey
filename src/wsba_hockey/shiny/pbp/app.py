import pandas as pd
import plotly.express as px
import var
import plot as wsba_plt
from urllib.parse import *
from shiny import *
from shinywidgets import output_widget, render_widget 

app_ui = ui.page_fluid(
    output_widget("plot_game"),
)

def server(input, output, session):
    @output()
    @render_widget
    def plot_game():
        #Retreive query parameters
        search = session.input[".clientdata_url_search"]()
        query = parse_qs(urlparse(search).query)
        
        #If no input data is provided automatically provide a select game and plot all 5v5 fenwick shots
        if 'event_type' not in query.keys() or 'strength_state' not in query.keys():
            if 'game_id' not in query.keys():
                query = {'game_id':['2024021000'],'event_type':var.fenwick_events,'strength_state':['5v5']}
            else:
                query.update({'event_type':var.fenwick_events,'strength_state':['5v5']})

        #Determine which season to load based on the input game_id
        front_year = int(query['game_id'][0][0:4])
        season = f'{front_year}{front_year+1}'
        #Load appropriate dataframe
        df = pd.read_parquet(f'https://f005.backblazeb2.com/file/weakside-breakout/pbp/{season}.parquet')
    
        #Prepare dataframe for plotting based on URL parameters
        df = df.loc[df['game_id'].astype(str).isin(query['game_id'])]
        df = wsba_plt.prep(df,events=query['event_type'],strengths=query['strength_state'])

        #Return empty rink if no data exists else continue
        if df.empty:
            return wsba_plt.wsba_rink()
        else:
            game_title = df['game_title'].to_list()[0]
            colors = wsba_plt.colors(df)
            rink = wsba_plt.wsba_rink()

            plot = px.scatter(df,
                              x='x', y='y',
                              size='size',
                              color='Team',
                              color_discrete_map=colors,
                              hover_name='Description',
                              hover_data=['Event Num.', 'Period', 'Time (in seconds)',
                                          'Away Score', 'Home Score', 'x', 'y',
                                          'Event Distance from Attacking Net',
                                          'Event Angle to Attacking Net',
                                          'xG'])

            for trace in plot.data:
                rink.add_trace(trace)

            return rink.update_layout(
                title=dict(text=game_title,
                           x=0.5, y=0.94,
                           xanchor='center',
                           yanchor='top'),
                legend=dict(
                    orientation='h',
                    x=0.49,
                    y=0.065,
                    xanchor='center',
                    yanchor='bottom',
                )
            )

app = App(app_ui, server)