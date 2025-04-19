import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from tools.xg_model import *
from hockey_rink import NHLRink
from hockey_rink import CircularImage

event_colors = {
    'faceoff':'black',
    'hit':'yellow',
    'blocked-shot':'pink',
    'missed-shot':'red',
    'shot-on-goal':'purple',
    'goal':'blue',
    'giveaway':'orange',
    'takeaway':'green',
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

def prep_plot_data(pbp,events,strengths,color_dict=event_colors,xg='moneypuck'):
    try: pbp['xG']
    except:
        if xg == 'wsba':
            pbp = wsba_xG(pbp)
        else:
            pbp = moneypuck_xG(pbp)
        pbp['xG'] = np.where(pbp['xG'].isna(),0,pbp['xG'])

    pbp['WSBA'] = pbp['event_player_1_name']+pbp['season'].astype(str)+pbp['event_team_abbr']
    
    pbp['x_plot'] = pbp['y_fixed']*-1
    pbp['y_plot'] = pbp['x_fixed']

    pbp['size'] = np.where(pbp['xG']<=0,100,pbp['xG']*1000)
    pbp['color'] = pbp['event_type'].replace(color_dict)

    pbp = pbp.loc[(pbp['event_type'].isin(events))&
                  (pbp['event_distance']<=89)&
                  (pbp['x_fixed']<=89)&
                  (pbp['strength_state'].isin(strengths))]
    
    return pbp

def league_shots(pbp,cord = 'fixed'):
    [x,y] = np.round(np.meshgrid(np.linspace(0,100,100),np.linspace(-42.5,42.5,85)))
    xgoals = griddata((pbp[f'x_{cord}'],pbp[f'y_{cord}']),pbp['xG'],(x,y),method='cubic',fill_value=0)
    xgoals_smooth = gaussian_filter(xgoals,sigma = 3)

    fig = plt.figure(figsize=(10,12), facecolor='w', edgecolor='k')
    plt.imshow(xgoals_smooth,origin = 'lower')
    plt.colorbar(orientation = 'horizontal', pad = 0.05)
    plt.title('xGoal Array',fontdict={'fontsize': 15})
    plt.show()

    return xgoals_smooth


def plot_skater_shots(pbp, player, season, team, strengths, title = None, color_dict=event_colors, legend=False,xg='moneypuck'):
    shots = ['missed-shot','shot-on-goal','goal']
    pbp = prep_plot_data(pbp,shots,strengths,color_dict,xg)
    skater = pbp.loc[pbp['WSBA'] == f'{player.upper()}{season}{team}']

    fig, ax = plt.subplots()
    wsba_rink(rotation=90)

    for event in shots:
        plays = skater.loc[skater['event_type']==event]
        ax.scatter(plays['x_plot'],plays['y_plot'],plays['size'],plays['color'],label=event)
    
    ax.set_title(title) if title else ''
    ax.legend().set_visible(legend)
    
    return fig
    
def plot_game_events(pbp,game_id,events,strengths,color_dict=event_colors,legend=False,xg='moneypuck'):
    pbp = prep_plot_data(pbp,events,strengths,color_dict,xg)
    pbp = pbp.loc[pbp['game_id'].astype(str)==game_id]

    away_abbr = list(pbp['away_team_abbr'])[0]
    home_abbr = list(pbp['home_team_abbr'])[0]
    date = list(pbp['game_date'])[0]

    fig, ax = plt.subplots()
    wsba_rink(display_range='full')

    for event in events:
        plays = pbp.loc[pbp['event_type']==event]
        ax.scatter(plays['x_adj'],plays['y_adj'],plays['size'],plays['color'],label=event)

    ax.set_title(f'{away_abbr} @ {home_abbr} - {date}')
    ax.legend(bbox_to_anchor =(0.5,-0.4), loc='lower center',ncol=1).set_visible(legend)

    return fig
