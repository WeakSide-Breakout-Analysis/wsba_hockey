import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import urllib.request
import PIL
from wsba_hockey.tools.xg_model import *
from hockey_rink import NHLRink
from hockey_rink import CircularImage

event_markers = {
    'faceoff':'X',
    'hit':'P',
    'blocked-shot':'v',
    'missed-shot':'o',
    'shot-on-goal':'D',
    'goal':'*',
    'giveaway':'1',
    'takeaway':'2',
}

def wsba_rink(display_range='offense',rotation = 0):
    img = 'tools/utils/wsba.png'
    rink = NHLRink(center_logo={
        "feature_class": CircularImage,
        "image_path": img,
        "length": 25, "width": 25,
        "x": 0, "y": 0,
        "radius": 14,    
        "zorder": 11,
        }
        )
    rink.draw(
            display_range=display_range,
            rotation=rotation,
            despine=True
        )

def prep_plot_data(pbp,events,strengths,marker_dict=event_markers):
    try: pbp['xG']
    except:
        pbp = wsba_xG(pbp)
        pbp['xG'] = np.where(pbp['xG'].isna(),0,pbp['xG'])

    pbp['WSBA'] = pbp['event_player_1_name']+pbp['season'].astype(str)+pbp['event_team_abbr']
    
    pbp['x_plot'] = np.where(pbp['x']<0,-pbp['y_adj'],pbp['y_adj'])
    pbp['y_plot'] = abs(pbp['x_adj'])

    pbp['home_on_ice'] = pbp['home_on_1'].astype(str) + ";" + pbp['home_on_2'].astype(str) + ";" + pbp['home_on_3'].astype(str) + ";" + pbp['home_on_4'].astype(str) + ";" + pbp['home_on_5'].astype(str) + ";" + pbp['home_on_6'].astype(str)
    pbp['away_on_ice'] = pbp['away_on_1'].astype(str) + ";" + pbp['away_on_2'].astype(str) + ";" + pbp['away_on_3'].astype(str) + ";" + pbp['away_on_4'].astype(str) + ";" + pbp['away_on_5'].astype(str) + ";" + pbp['away_on_6'].astype(str)

    pbp['onice_for'] = np.where(pbp['home_team_abbr']==pbp['event_team_abbr'],pbp['home_on_ice'],pbp['away_on_ice'])
    pbp['onice_against'] = np.where(pbp['away_team_abbr']==pbp['event_team_abbr'],pbp['home_on_ice'],pbp['away_on_ice'])

    pbp['size'] = np.where(pbp['xG']<=0,40,pbp['xG']*400)
    pbp['marker'] = pbp['event_type'].replace(marker_dict)

    pbp = pbp.loc[(pbp['event_type'].isin(events))]
    
    if strengths != 'all':
        pbp = pbp.loc[(pbp['strength_state'].isin(strengths))]

    return pbp

def league_shots(pbp,events,strengths):
    pbp = prep_plot_data(pbp,events,strengths)

    print(pbp[['event_player_1_name','xG','x_plot','y_plot']].head(10))

    [x,y] = np.round(np.meshgrid(np.linspace(-42.5,42.5,85),np.linspace(0,100,100)))
    xgoals = griddata((pbp[f'x_plot'],pbp[f'y_plot']),pbp['xG'],(x,y),method='cubic',fill_value=0)
    xgoals_smooth = gaussian_filter(xgoals,sigma = 3)

    return xgoals_smooth

def plot_skater_shots(pbp, player, season, team, strengths, title = None, marker_dict=event_markers, onice='for', legend=False):
    shots = ['missed-shot','shot-on-goal','goal']
    pbp = prep_plot_data(pbp,shots,strengths,marker_dict)
    pbp = pbp.loc[(pbp['season'].astype(str)==season)&((pbp['away_team_abbr']==team)|(pbp['home_team_abbr']==team))]

    team_data = pd.read_csv('teaminfo/nhl_teaminfo.csv')
    team_color = list(team_data.loc[team_data['WSBA']==f'{team}{season}','Primary Color'])[0]
    team_color_2nd = list(team_data.loc[team_data['WSBA']==f'{team}{season}','Secondary Color'])[0]

    if onice in ['for','against']:
        skater = pbp.loc[(pbp[f'onice_{onice}'].str.contains(player.upper()))]
        skater['color'] = np.where(skater['event_player_1_name']==player.upper(),team_color,team_color_2nd)

    else:
        skater = pbp.loc[pbp['event_player_1_name']==player.upper()]
        skater['color'] = team_color

    fig, ax = plt.subplots()
    wsba_rink(rotation=90)

    for event in shots:
        plays = skater.loc[skater['event_type']==event]
        ax.scatter(plays['x_plot'],plays['y_plot'],plays['size'],plays['color'],marker=event_markers[event],label=event,zorder=5)
    
    ax.set_title(title) if title else ''
    ax.legend().set_visible(legend)
    ax.legend().set_zorder(1000)
    
    return fig
    
def plot_game_events(pbp,game_id,events,strengths,marker_dict=event_markers,team_colors={'away':'secondary','home':'primary'},legend=False):
    pbp = prep_plot_data(pbp,events,strengths,marker_dict)
    pbp = pbp.loc[pbp['game_id'].astype(str)==str(game_id)]
    
    away_abbr = list(pbp['away_team_abbr'])[0]
    home_abbr = list(pbp['home_team_abbr'])[0]
    date = list(pbp['game_date'])[0]
    season = list(pbp['season'])[0]

    team_data = pd.read_csv('teaminfo/nhl_teaminfo.csv')
    team_info ={
        'away_color':'#000000' if list(team_data.loc[team_data['WSBA']==f'{away_abbr}{season}','Secondary Color'])[0]=='#FFFFFF' else list(team_data.loc[team_data['WSBA']==f'{away_abbr}{season}',f'{team_colors['away'].capitalize()} Color'])[0],
        'home_color': list(team_data.loc[team_data['WSBA']==f'{home_abbr}{season}',f'{team_colors['home'].capitalize()} Color'])[0],
        'away_logo': f'tools/logos/png/{away_abbr}{season}.png',
        'home_logo': f'tools/logos/png/{home_abbr}{season}.png',
    }

    pbp['color'] = np.where(pbp['event_team_abbr']==away_abbr,team_info['away_color'],team_info['home_color'])

    fig, ax = plt.subplots()
    wsba_rink(display_range='full')

    for event in events:
        plays = pbp.loc[pbp['event_type']==event]
        ax.scatter(plays['x_adj'],plays['y_adj'],plays['size'],plays['color'],marker=event_markers[event],edgecolors='white',label=event,zorder=5)

    ax.set_title(f'{away_abbr} @ {home_abbr} - {date}')
    ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center',ncol=1).set_visible(legend)

    return fig
