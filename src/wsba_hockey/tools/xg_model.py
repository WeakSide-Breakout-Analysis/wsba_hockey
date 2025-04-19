import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.sparse as sp
import joblib
from zipfile import ZipFile
import requests as rs

### XG_MODEL FUNCTIONS ###
# Provided in this file are functions vital to the goal prediction model in the WSBA Hockey Python package. #

## GLOBAL VARIABLES ##
#Newest season
new_full = '20242025'
new = '2024'

def prep_xG_data(pbp):
    #Prep data for xG training and calculation

    events = ['faceoff','hit','giveaway','takeaway','blocked-shot','missed-shot','shot-on-goal','goal']
    shot_types = ['wrist','deflected','tip-in','slap','backhand','snap','wrap-around','poke','bat','cradle','between-legs']
    fenwick_events = ['missed-shot','shot-on-goal','goal']
    
    #Informal groupby
    data = pbp.sort_values(by=['season','game_id','period','seconds_elapsed','event_num'])

    #Add event time details - prevent leaking between games by setting value to zero when no time has occured in game
    data["seconds_since_last"] = np.where(data['seconds_elapsed']==0,0,data['seconds_elapsed']-data['seconds_elapsed'].shift(1))
    data["event_length"] = np.where(data['seconds_elapsed']==0,0,data['seconds_since_last'].shift(-1))
    
    #Create last event columns
    data["event_team_last"] = data['event_team_abbr'].shift(1)
    data["event_type_last"] = data['event_type'].shift(1)
    data["x_fixed_last"] = data['x_fixed'].shift(1)
    data["y_fixed_last"] = data['y_fixed'].shift(1)
    data["zone_code_last"] = data['zone_code'].shift(1)    

    data.sort_values(['season','game_id','period','seconds_elapsed','event_num'],inplace=True)
    data['score_state'] = np.where(data['away_team_abbr']==data['event_team_abbr'],data['away_score']-data['home_score'],data['home_score']-data['away_score'])
    data['strength_diff'] = np.where(data['away_team_abbr']==data['event_team_abbr'],data['away_skaters']-data['home_skaters'],data['home_skaters']-data['away_skaters'])
    data['fenwick_state'] = np.where(data['away_team_abbr']==data['event_team_abbr'],data['away_fenwick']-data['home_fenwick'],data['home_fenwick']-data['away_fenwick'])
    data['distance_from_last'] = np.sqrt((data['x_fixed'] - data['x_fixed_last'])**2 + (data['y_fixed'] - data['y_fixed_last'])**2)

    #Rush and rebounds are included and graded off of the speed of the event (an event cannot be a rush event unless it also occurs in the offensive zone)
    data['rush_mod'] = np.where((data['event_type'].isin(fenwick_events))&(data['zone_code_last'].isin(['N','D']))&(data['x_fixed']>25)&(data['seconds_since_last']<5),5-data['seconds_since_last'],0)
    data['rebound_mod'] = np.where((data['event_type'].isin(fenwick_events))&(data['event_type_last'].isin(fenwick_events))&(data['seconds_since_last']<3),3-data['seconds_since_last'],0)

    #Create boolean variables
    data["is_goal"]=(data['event_type']=='goal').astype(int)
    data["is_home"]=(data['home_team_abbr']==data['event_team_abbr']).astype(int)

    #Boolean variables for shot types and prior events
    for shot in shot_types:
        data[shot] = (data['shot_type']==shot).astype(int)
    for event in events[0:len(events)-1]:
        data[f'prior_{event}_same'] = ((data['event_type_last']==event)&(data['event_team_last']==data['event_team_abbr'])).astype(int)
        data[f'prior_{event}_opp'] = ((data['event_type_last']==event)&(data['event_team_last']!=data['event_team_abbr'])).astype(int)
    
    data['prior_faceoff'] = (data['event_type_last']=='faceoff').astype(int)
    
    #Return: pbp data prepared to train and calculate the xG model
    return data

def wsba_xG(pbp, train = False, overwrite = False, model_path = "tools/xg_model/wsba_xg.joblib", train_runs = 20, cv_runs = 20):
    #Train and calculate the WSBA Expected Goals model
    
    target = "is_goal"
    continous = ['event_distance',
                'event_angle',
                'seconds_elapsed',
                'period',
                'x_fixed',
                'y_fixed',
                'x_fixed_last',
                'y_fixed_last',
                'distance_from_last',
                'seconds_since_last',
                'score_state',
                'strength_diff',
                'fenwick_state',
                'rush_mod',
                'rebound_mod']
    boolean = ['is_home',
            'wrist',
            'deflected',
            'tip-in',
            'slap',
            'backhand',
            'snap',
            'wrap-around',
            'poke',
            'bat',
            'cradle',
            'between-legs',
            'prior_shot-on-goal_same',
            'prior_missed-shot_same',
            'prior_blocked-shot_same',
            'prior_giveaway_same',
            'prior_takeaway_same',
            'prior_hit_same',
            'prior_shot-on-goal_opp',
            'prior_missed-shot_opp',
            'prior_blocked-shot_opp',
            'prior_giveaway_opp',
            'prior_takeaway_opp',
            'prior_hit_opp',
            'prior_faceoff']
    
    #Prep Data
    pbp = prep_xG_data(pbp)
    #Filter unwanted date:
    #Shots must occur in specified events and strength states, occur before the shootout, and have valid coordinates
    events = ['faceoff','hit','giveaway','takeaway','blocked-shot','missed-shot','shot-on-goal','goal']
    fenwick_events = ['missed-shot','shot-on-goal','goal']
    strengths = ['3v3',
                '3v4',
                '3v5',
                '4v3',
                '4v4',
                '4v5',
                '4v6',
                '5v3',
                '5v4',
                '5v5',
                '5v6',
                '6v4',
                '6v5']
    
    data = pbp.loc[(pbp['event_type'].isin(events))&
                   (pbp['strength_state'].isin(strengths))&
                   (pbp['period'] < 5)&
                   (pbp['x_fixed'].notna())&
                   (pbp['y_fixed'].notna())&
                   ~((pbp['x_fixed']==0)&(pbp['y_fixed']==0)&(pbp['x_fixed'].isin(fenwick_events))&(pbp['event_distance']!=90))]

    #Convert to sparse
    data_sparse = sp.csr_matrix(data[[target]+continous+boolean])

    #Target and Predictors
    is_goal_vect = data_sparse[:, 0].A
    predictors = data_sparse[:, 1:]

    #XGB DataModel
    xgb_matrix = xgb.DMatrix(data=predictors,label=is_goal_vect)

    if train == True:
        # Number of runs
        run_num = train_runs

        # DataFrames to store results
        best_df = pd.DataFrame(columns=["max_depth", "eta", "gamma", "subsample", "colsample_bytree", "min_child_weight", "max_delta_step"])
        best_ll = pd.DataFrame(columns=["ll", "ll_rounds", "auc", "auc_rounds", "seed"])

        # Loop
        for i in range(run_num):
            print(f"### LOOP: {i+1} ###")
            
            param = {
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "auc"],
                "max_depth": 6,
                "eta": np.random.uniform(0.06, 0.11),
                "gamma": np.random.uniform(0.06, 0.12),
                "subsample": np.random.uniform(0.76, 0.84),
                "colsample_bytree": np.random.uniform(0.76, 0.8),
                "min_child_weight": np.random.randint(5, 23),
                "max_delta_step": np.random.randint(4, 9)
            }
            
            # Cross-validation
            seed = np.random.randint(0, 10000)
            np.random.seed(seed)
            
            cv_results = xgb.cv(
                params=param,
                dtrain=xgb_matrix,
                num_boost_round=1000,
                nfold=5,
                early_stopping_rounds=25,
                metrics=["logloss", "auc"],
                seed=seed
            )
            
            # Record results
            best_df.loc[i] = param
            best_ll.loc[i] = [
                cv_results["test-logloss-mean"].min(),
                cv_results["test-logloss-mean"].idxmin(),
                cv_results["test-auc-mean"].max(),
                cv_results["test-auc-mean"].idxmax(),
                seed
            ]

        # Combine results
        best_all = pd.concat([best_df, best_ll], axis=1).dropna()

        # Arrange to get best run
        best_all = best_all.sort_values(by="auc", ascending=False)

        if overwrite == True:
            best_all.to_csv("xg_model/testing/xg_model_training_runs.csv",index=False)
        else: 
            best_old = pd.read_csv("xg_model/testing/xg_model_training_runs.csv")
            best_comb = pd.concat([best_old,best_all])
            best_comb.to_csv("xg_model/testing/xg_model_training_runs.csv",index=False)

        # Final parameters
        param_7_EV = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "eta": 0.068,
            "gamma": 0.12,
            "subsample": 0.78,
            "max_depth": 6,
            "colsample_bytree": 0.76,
            "min_child_weight": 5,
            "max_delta_step": 5,
        }

        # CV rounds Loop
        run_num = cv_runs
        cv_test = pd.DataFrame(columns=["AUC_rounds", "AUC", "LL_rounds", "LL", "seed"])

        for i in range(run_num):
            print(f"### LOOP: {i+1} ###")
            
            seed = np.random.randint(0, 10000)
            np.random.seed(seed)
            
            cv_rounds = xgb.cv(
                params=param_7_EV,
                dtrain=xgb_matrix,
                num_boost_round=1000,
                nfold=5,
                early_stopping_rounds=25,
                metrics=["logloss", "auc"],
                seed=seed
            )
            
            # Record results
            cv_test.loc[i] = [
                cv_rounds["test-auc-mean"].idxmax(),
                cv_rounds["test-auc-mean"].max(),
                cv_rounds["test-logloss-mean"].idxmin(),
                cv_rounds["test-logloss-mean"].min(),
                seed
            ]

        # Clean results and sort to find the number of rounds to use and seed
        cv_final = cv_test.sort_values(by="AUC", ascending=False)
        if overwrite == True:
            cv_final.to_csv("xg_model/testing/xg_model_cv_runs.csv",index=False)
        else:
            cv_old = pd.read_csv("xg_model/testing/xg_model_cv_runs.csv")
            cv_comb = pd.concat([cv_old,cv_final])
            cv_comb.to_csv("xg_model/testing/xg_model_cv_runs.csv")
        cv_final.loc[len(cv_final)] = cv_test.mean()

        # Train the final model
        np.random.seed(556)
        
        if overwrite == False:
            model = joblib.load(model_path)
        else:
            ""

        model = xgb.train(
            params=param_7_EV,
            dtrain=xgb_matrix,
            num_boost_round=189,
            verbose_eval=2
        )

        joblib.dump(model,model_path)
        
    else:
        model = joblib.load(model_path)
        pbp['xG'] = np.where(pbp['event_type'].isin(fenwick_events),model.predict(xgb_matrix),"")
        return pbp

def moneypuck_xG(pbp,repo_path = "tools/xg_model/moneypuck/shots_2007-2023.zip"):
    #Given play-by-play, return itself with xG column sourced from MoneyPuck.com

    #If file is already in the repository downloading is not necessary
    try:
        db = pd.read_parquet("tools/xg_model/moneypuck/shots/shots_2007-2023.parquet")
    except:
        url = 'https://peter-tanner.com/moneypuck/downloads/shots_2007-2023.zip'

        response = rs.get(url)

        if response.status_code == 200:
            with open(repo_path, 'wb') as file:
                file.write(response.content)
            print('File downloaded successfully')
        else:
            print('Failed to download file')

        with ZipFile(repo_path, 'r') as zObject: 
            zObject.extractall( 
                path="tools/xg_model/moneypuck/shots/")
        
        db = pd.read_csv("tools/xg_model/moneypuck/shots/shots_2007-2023.csv")  
    
    #Repeat process with active/most recent season
    #For the new/recent season, only scrape if the supplied pbp data contains the season
    if new in list(pbp['season'].astype(str).str[0:4]):
        url = f'https://peter-tanner.com/moneypuck/downloads/shots_{new}.zip'
        repo_path = f"tools/xg_model/moneypuck/shots_{new}.zip"

        response = rs.get(url)

        if response.status_code == 200:
            with open(repo_path, 'wb') as file:
                file.write(response.content)
            print('File downloaded successfully')
        else:
            print('Failed to download file')

        with ZipFile(repo_path, 'r') as zObject: 
            zObject.extractall( 
                path="tools/xg_model/moneypuck/shots/")
            
        new_season = pd.read_csv(f"tools/xg_model/moneypuck/shots/shots_{new}.csv")
        #Convert to parquet
        new_season.to_parquet(f"tools/xg_model/moneypuck/shots/shots_{new}.csv",index=False)
    else:
        new_season = pd.DataFrame()
    #Combine shots
    moneypuck = pd.concat([db,new_season])

    #Find game ids that occur in supplied pbp and filter moneypuck shots accordingly
    moneypuck['game_id'] = moneypuck['season'].astype(str)+"0"+moneypuck['game_id'].astype(str)
    moneypuck['event'] = moneypuck['event'].replace({
        "SHOT":"shot-on-goal",
        "MISS":"missed-shot",
        "BLOCK":"blocked-shot",
        "GOAL":"goal"
    })
    
    #Manual Team Rename
    moneypuck['teamCode'] = moneypuck['teamCode'].replace({
        "L.A":"LAK",
        "N.J":"NJD",
        "S.J":"SJS",
        "T.B":"TBL",
    })
    pbp['event_team_abbr'] = pbp['event_team_abbr'].replace({
        "L.A":"LAK",
        "N.J":"NJD",
        "S.J":"SJS",
        "T.B":"TBL",
        "PHX":'ARI'
    })

    #Managing oddities in datatypes
    moneypuck[['game_id','period','time']] = moneypuck[['game_id','period','time']].astype(int)
    pbp[['game_id','period','seconds_elapsed']] = pbp[['game_id','period','seconds_elapsed']].astype(int)

    #Modify and merge
    moneypuck = moneypuck[['game_id','period','time','event','teamCode','shooterPlayerId','xGoal']]
    comb = pd.merge(pbp,moneypuck
                    ,left_on=['game_id','period','seconds_elapsed','event_type','event_team_abbr','event_player_1_id']
                    ,right_on=['game_id','period','time','event','teamCode','shooterPlayerId']
                    ,how='left')
    
    #Drop and rename
    pbp_xg = comb.drop(columns=['time', 'event', 'teamCode', 'shooterPlayerId']).rename(columns={'xGoal':'xG'})
    
    if pbp_xg['xG'].isnull().all():
        print("No MoneyPuck xG values were found for this game...")

    #Return: play-by-play with moneypuck xG column
    return pbp_xg
