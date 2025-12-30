import os
import json
import pandas as pd
import numpy as np
import wsba_hockey.tools.agg as agg
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from wsba_hockey.tools.columns import col_map

### GAME SCORE MODEL FUNCTIONS ###
# Provided in this file are functions vital to the game score model in the WSBA Hockey Python package. #

## GLOBAL VARIABLES ##
skater = [
    "points",
    "penalties_drawn_percentage",
    "puck_management_percentage",
    "faceoff_percentage",
    "even_strength_expected_goals_contribution_percentage",
    "even_strength_expected_goals_for",
    "even_strength_expected_goals_against",
    "power_play_expected_goals_contribution_percentage",
    "power_play_expected_goals_for",
    "power_play_expected_goals_against",
    "short_handed_expected_goals_contribution_percentage",
    "short_handed_expected_goals_for",
    "short_handed_expected_goals_against"
]

goalie = [
    "expected_goals_for_percentage",
    "goals_against_per_expected_goals_against"
]

dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir,'game_score\\')
roster_path = os.path.join(dir,'rosters\\nhl_rosters.csv')
schedule_path = os.path.join(dir,'schedule\\schedule.csv')

def game_score_model(pbp,types=['skater','goalie']):
    for t in types:
        stats = agg.calc_game_score_features(pbp,t).rename(columns=col_map()['stats'])

        target = 'team_goal_differential'
        features = stats[skater if t=='skater' else goalie]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        model = Ridge(alpha=1.0)
        model.fit(scaled,stats[target])

        model_json = {
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_.item(),
            "features": list(features.columns),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist()
        }

        with open(os.path.join(model_path, f'wsba_gs_{t}.json'), "w") as f:
            json.dump(model_json, f, indent=4)
