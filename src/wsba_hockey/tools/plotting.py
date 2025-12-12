import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from hockey_rink import NHLRink
from hockey_rink import CircularImage
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from wsba_hockey.tools.xg_model import *

### PLOTTING FUNCTIONS ###
# Provided in this file are basic plotting functions for the WSBA Hockey Python package. #

## GLOBAL VARIABLES ##

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

legend_elements = [
    Line2D([0], [0], marker='o', color='blue', label='missed-shot', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='D', color='blue', label='shot-on-goal', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='*', color='blue', label='goal', markersize=8, linestyle='None'),
]

dir = os.path.dirname(os.path.realpath(__file__))
info_path = os.path.join(dir,'teaminfo\\nhl_teaminfo.csv')
img_path = os.path.join(dir,'utils\\wsba.png')

def wsba_rink(display_range='offense',rotation = 0):
    rink = NHLRink(center_logo={
        "feature_class": CircularImage,
        "image_path": img_path,
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

    pbp['WSBA'] = pbp['event_player_1_id'].astype(str)+pbp['season'].astype(str)+pbp['event_team_abbr']
    
    pbp['event_team_abbr_2'] = np.where(pbp['event_team_venue']=='home',pbp['away_team_abbr'],pbp['home_team_abbr'])
    
    pbp['x_plot'] = np.where(pbp['x']<0,-pbp['y_adj'],pbp['y_adj'])
    pbp['y_plot'] = abs(pbp['x_adj'])

    pbp['home_on_ice'] = pbp['home_on_1'].astype(str) + ";" + pbp['home_on_2'].astype(str) + ";" + pbp['home_on_3'].astype(str) + ";" + pbp['home_on_4'].astype(str) + ";" + pbp['home_on_5'].astype(str) + ";" + pbp['home_on_6'].astype(str)
    pbp['away_on_ice'] = pbp['away_on_1'].astype(str) + ";" + pbp['away_on_2'].astype(str) + ";" + pbp['away_on_3'].astype(str) + ";" + pbp['away_on_4'].astype(str) + ";" + pbp['away_on_5'].astype(str) + ";" + pbp['away_on_6'].astype(str)

    pbp['home_on_ice_id'] = pbp['home_on_1_id'].astype(str) + ";" + pbp['home_on_2_id'].astype(str) + ";" + pbp['home_on_3_id'].astype(str) + ";" + pbp['home_on_4_id'].astype(str) + ";" + pbp['home_on_5_id'].astype(str) + ";" + pbp['home_on_6_id'].astype(str)
    pbp['away_on_ice_id'] = pbp['away_on_1_id'].astype(str) + ";" + pbp['away_on_2_id'].astype(str) + ";" + pbp['away_on_3_id'].astype(str) + ";" + pbp['away_on_4_id'].astype(str) + ";" + pbp['away_on_5_id'].astype(str) + ";" + pbp['away_on_6_id'].astype(str)

    pbp['onice_for_name'] = np.where(pbp['home_team_abbr']==pbp['event_team_abbr'],pbp['home_on_ice'],pbp['away_on_ice'])
    pbp['onice_against_name'] = np.where(pbp['away_team_abbr']==pbp['event_team_abbr'],pbp['home_on_ice'],pbp['away_on_ice'])

    pbp['onice_for_id'] = np.where(pbp['home_team_abbr']==pbp['event_team_abbr'],pbp['home_on_ice_id'],pbp['away_on_ice_id'])
    pbp['onice_against_id'] = np.where(pbp['away_team_abbr']==pbp['event_team_abbr'],pbp['home_on_ice_id'],pbp['away_on_ice_id'])

    pbp['strength_state_2'] = pbp['strength_state'].str[::-1]

    pbp['strength_state'] = np.where(pbp['strength_state'].isin(['5v5','5v4','4v5']),pbp['strength_state'],'Other')
    pbp['strength_state_2'] = np.where(pbp['strength_state_2'].isin(['5v5','5v4','4v5']),pbp['strength_state_2'],'Other')

    pbp['size'] = np.where(pbp['xG']<0.05,20,pbp['xG']*400)
    pbp['marker'] = pbp['event_type'].replace(marker_dict)

    pbp = pbp.loc[(pbp['event_type'].isin(events))]
    
    if strengths != 'all':
        pbp = pbp.loc[(pbp['strength_state'].isin(strengths))]

    return pbp

def gen_heatmap(pbp, player, season, team, strengths, strengths_title = None, title = None):
    pbp = pbp.loc[(pbp['season']==season)]
    
    df = prep_plot_data(pbp,['missed-shot','shot-on-goal','goal'],strengths)

    df = df.fillna(0)
    df = df.loc[(df['x_adj'].notna())&(df['y_adj'].notna())]

    fig, ax = plt.subplots(1, 1, figsize=(10,12), facecolor='w', edgecolor='k')
    wsba_rink(display_range='full')

    if isinstance(player, int):
        id_mod = '_id'
    else:
        id_mod = '_name'
        player = player.upper()

    for sit in ['for','against']:
        if sit == 'for':
            df['x'] = abs(df['x_adj'])
            df['y'] = np.where(df['x_adj']<0,-df['y_adj'],df['y_adj'])
            df['event_distance'] = abs(df['event_distance'].fillna(0))
            df = df.loc[(df['event_distance']<=89)&(df['x']<=89)&(df['empty_net']==0)]

            x_min = 0
            x_max = 100

            if strengths != 'all':
                df = df.loc[((df['strength_state'].isin(strengths)))]

        else:
            df['x'] = -abs(df['x_adj'])
            df['y'] = np.where(df['x_adj']>0,-df['y_adj'],df['y_adj'])
            df['event_distance'] = -abs(df['event_distance'])
            df = df.loc[(df['event_distance']>-89)&(df['x']>-89)&(df['empty_net']==0)]

            x_min = -100
            x_max = 0

            if strengths != 'all':
                df = df.loc[((df['strength_state_2'].isin(strengths)))]

        [x,y] = np.round(np.meshgrid(np.linspace(x_min,x_max,(x_max-x_min)),np.linspace(-42.5,42.5,85)))
        xgoals = griddata((df['x'],df['y']),df['xG'],(x,y),method='cubic',fill_value=0)
        xgoals = np.where(xgoals < 0,0,xgoals)
        xgoals_smooth = gaussian_filter(xgoals,sigma=3)

        if sit == 'for':
            player_shots = df.loc[(df[f'onice_for{id_mod}'].str.contains(str(player)))&(df['event_team_abbr']==team)]
        else:
            player_shots = df.loc[(df[f'onice_against{id_mod}'].str.contains(str(player)))&(df['event_team_abbr_2']==team)]
        [x,y] = np.round(np.meshgrid(np.linspace(x_min,x_max,(x_max-x_min)),np.linspace(-42.5,42.5,85)))
        xgoals_player = griddata((player_shots['x'],player_shots['y']),player_shots['xG'],(x,y),method='cubic',fill_value=0)
        xgoals_player = np.where(xgoals_player < 0,0,xgoals_player)

        difference = (gaussian_filter(xgoals_player,sigma = 3)) - xgoals_smooth
        data_min= difference.min()
        data_max= difference.max()
    
        if abs(data_min) > data_max:
            data_max = data_min * -1
        elif data_max > abs(data_min):
            data_min = data_max * -1
        
        cont = ax.contourf(
            x, y, difference,
            alpha=0.6,
            cmap='bwr',
            levels=np.linspace(data_min, data_max, 12),
            vmin=data_min,
            vmax=data_max
        )
    
    ax.text(-50, -50, 'Defense', ha='center', va='bottom', fontsize=12)
    ax.text(50, -50, 'Offense', ha='center', va='bottom', fontsize=12)
    
    cbar = fig.colorbar(
        cont,
        ax=ax,
        orientation='horizontal',
        shrink=0.75,
        fraction=0.05,
        pad=0.05,
        ticks=[data_min,data_min/2,0,data_max/2,data_max]
    )

    cbar.ax.set_xticklabels(['Lower xG', '', 'Average xG', '', 'Higher xG'])
    cbar.set_label('Compared to League Average', fontsize=12)
    
    if not strengths_title:
        strengths_title = ', '.join(strengths) if np.logical_and(isinstance(strengths, list),strengths_title==None) else strengths.title()
    
    fig.text(0.5, 0.16, f'Strength(s): {strengths_title}', ha='center', fontsize=10)

    plt.title(title)

    return fig

def league_shots(pbp,events,strengths):
    pbp = prep_plot_data(pbp,events,strengths)

    print(pbp[['event_player_1_name','xG','x_plot','y_plot']].head(10))

    [x,y] = np.round(np.meshgrid(np.linspace(-42.5,42.5,85),np.linspace(0,100,100)))
    xgoals = griddata((pbp[f'x_plot'],pbp[f'y_plot']),pbp['xG'],(x,y),method='cubic',fill_value=0)
    xgoals_smooth = gaussian_filter(xgoals,sigma = 3)

    return xgoals_smooth

def plot_skater_shots(pbp, player, season, team, strengths, strengths_title = None, title = None, marker_dict=event_markers, onice='for', legend=False):
    shots = ['goal','missed-shot','shot-on-goal']
    pbp = prep_plot_data(pbp,shots,strengths,marker_dict)
    pbp = pbp.loc[(pbp['season']==season)&((pbp['away_team_abbr']==team)|(pbp['home_team_abbr']==team))]

    team_data = pd.read_csv(info_path)
    team_color = list(team_data.loc[team_data['WSBA']==f'{team}{season}','primary_color'])[0]
    team_color_2nd = list(team_data.loc[team_data['WSBA']==f'{team}{season}','secondary_color'])[0]

    if isinstance(player, int):
        id_mod = '_id'
    else:
        id_mod = '_name'
        player = player.upper()

    if onice in ['for','against']:
        skater = pbp.loc[(pbp[f'onice_{onice}{id_mod}'].str.contains(str(player)))]
        skater['color'] = np.where(skater[f'event_player_1{id_mod}']==player,team_color,team_color_2nd)
    else:
        skater = pbp.loc[pbp[f'event_player_1{id_mod}']==player]
        skater['color'] = team_color

    fig, ax = plt.subplots()
    wsba_rink(rotation=90)

    for event in shots:
        plays = skater.loc[skater['event_type']==event]
        ax.scatter(plays['x_plot'],plays['y_plot'],plays['size'],plays['color'],marker=event_markers[event],label=event,zorder=5)
    
    ax.set_title(title) if title else ''
    ax.legend().set_visible(legend)
    ax.legend().set_zorder(1000)
    
    if not strengths_title:
        strengths_title = ', '.join(strengths) if np.logical_and(isinstance(strengths, list),strengths_title==None) else strengths.title()

    fig.text(0.5, 0.16, f'Strength(s): {strengths_title}', ha='center', fontsize=10)

    return fig
    
def plot_game_events(pbp,game_id,events,strengths,marker_dict=event_markers,team_colors={'away':'secondary','home':'primary'},legend=False):
    pbp = prep_plot_data(pbp,events,strengths,marker_dict)
    pbp = pbp.loc[pbp['game_id'].astype(str)==str(game_id)]
    
    away_abbr = list(pbp['away_team_abbr'])[0]
    home_abbr = list(pbp['home_team_abbr'])[0]
    date = list(pbp['game_date'])[0]
    season = list(pbp['season'])[0]

    team_data = pd.read_csv(info_path)
    team_info ={
        'away_color':'#000000' if list(team_data.loc[team_data['WSBA']==f'{away_abbr}{season}','secondary_color'])[0]=='#FFFFFF' else list(team_data.loc[team_data['WSBA']==f'{away_abbr}{season}',f'{team_colors['away']}_color'])[0],
        'home_color': list(team_data.loc[team_data['WSBA']==f'{home_abbr}{season}',f'{team_colors['home']}_color'])[0],
        'away_logo': f'tools/logos/png/{away_abbr}{season}.png',
        'home_logo': f'tools/logos/png/{home_abbr}{season}.png',
    }

    pbp['color'] = np.where(pbp['event_team_abbr']==away_abbr,team_info['away_color'],team_info['home_color'])

    fig, ax = plt.subplots()
    wsba_rink(display_range='full')

    for event in events:
        plays = pbp.loc[pbp['event_type']==event]
        ax.scatter(plays['x_adj'],plays['y_adj'],plays['size'],plays['color'],marker=event_markers[event],edgecolors='black' if event=='goal' else 'white',label=event,zorder=5)

    ax.set_title(f'{away_abbr} @ {home_abbr} - {date}')
    ax.legend(handles=legend_elements, bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=1).set_visible(legend)

    return fig
