import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.sparse as sp
import joblib
import wsba_main as wsba
import tools.scraping as scraping
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

### XG_MODEL FUNCTIONS ###
# Provided in this file are functions vital to the goal prediction model in the WSBA Hockey Python package. #

## GLOBAL VARIABLES ##

target = "is_goal"
continuous = ['event_distance',
            'event_angle',
            'seconds_elapsed',
            'period',
            'x_adj',
            'y_adj',
            'distance_from_last',
            'angle_from_last',
            'seconds_since_last',
            'speed_from_last',
            'speed_of_angle_from_last',
            'score_state',
            'strength_diff'
            ]
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
        'prior_faceoff',
        'regular',
        'empty_net',
        'offwing',
        'rush',
        'rebound'
        ]

events = ['faceoff','hit','giveaway','takeaway','blocked-shot','missed-shot','shot-on-goal','goal']
shot_types = ['wrist','deflected','tip-in','slap','backhand','snap','wrap-around','poke','bat','cradle','between-legs']
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

def fix_players(pbp):
    #Add/fix player info for shooters and goaltenders
    print('Adding player info to pbp...')

    #Load roster and all players
    roster = pd.read_csv('rosters/nhl_rosters.csv').drop_duplicates(['id'])[['fullName','id','shootsCatches']]

    #Some players are missing from the roster file (generally in newer seasons); add these manually
    miss = list(pbp.loc[~(pbp['event_player_1_id'].isin(list(roster['id'])))&(pbp['event_player_1_id'].notna()),'event_player_1_id'].drop_duplicates())
    if miss:
        add = wsba.nhl_scrape_player_data(miss).rename(columns={'playerId':'id'})[['fullName','id','shootsCatches']]
        roster = pd.concat([roster,add]).reset_index(drop=True)

    #Conversion dict
    roster['id'] = roster['id'].astype(str)
    roster_dict = roster.set_index('id').to_dict()['shootsCatches']
    names_dict = roster.set_index('id').to_dict()['fullName']

    #Add player names
    for i in range(3):
        pbp[f'add_player_{i+1}_name'] = np.where(pbp[f'event_player_{i+1}_name'].isna(),pbp[f'event_player_{i+1}_id'].astype(str).replace(names_dict),np.nan)
        pbp[f'event_player_{i+1}_name'] = pbp[f'event_player_{i+1}_name'].combine_first(pbp[f'add_player_{i+1}_name'])

    pbp['event_goalie_name'] = pbp['event_goalie_id'].astype(str).replace(names_dict)

    #Add hands
    pbp['event_player_1_hand'] = pbp['event_player_1_id'].astype(str).str.replace('.0','').replace(roster_dict)
    pbp['event_player_1_hand'] = pbp['event_player_1_hand'].replace('nan',np.nan)

    return pbp

def prep_xG_data(data):
    #Prep data for xG training and calculation

    #Informal groupby
    data = data.sort_values(by=['season','game_id','period','seconds_elapsed','event_num'])

    #Recalibrate times series data with current data
    data['seconds_since_last'] = data['seconds_elapsed'] - data['seconds_elapsed'].shift(1) 
    #Prevent leaking between games by setting value to zero when no time has occured in game
    data["seconds_since_last"] = np.where(data['seconds_elapsed']==0,0,data['seconds_since_last'])
    
    #Create last event columns
    data["event_team_last"] = data['event_team_abbr'].shift(1)
    data["event_type_last"] = data['event_type'].shift(1)
    data["x_adj_last"] = data['x_adj'].shift(1)
    data["y_adj_last"] = data['y_adj'].shift(1)
    data["zone_code_last"] = data['zone_code'].shift(1)

    data.sort_values(['season','game_id','period','seconds_elapsed','event_num'],inplace=True)
    
    #Contextual Data (for score state minimize the capture to four goals)
    data['score_state'] = np.where(data['away_team_abbr']==data['event_team_abbr'],data['away_score']-data['home_score'],data['home_score']-data['away_score'])
    data['score_state'] = np.where(data['score_state']>4,4,data['score_state'])
    data['score_state'] = np.where(data['score_state']<-4,-4,data['score_state'])

    data['strength_diff'] = np.where(data['away_team_abbr']==data['event_team_abbr'],data['away_skaters']-data['home_skaters'],data['home_skaters']-data['away_skaters'])
    data['strength_state_venue'] = data['away_skaters'].astype(str)+'v'+data['home_skaters'].astype(str)
    data['distance_from_last'] = np.sqrt((data['x_adj'] - data['x_adj_last'])**2 + (data['y_adj'] - data['y_adj_last'])**2)
    data['angle_from_last'] = np.degrees(np.arctan2(abs(data['y_adj'] - data['y_adj_last']), abs(89 - (data['x_adj']-data['x_adj_last']))))

    #Event speeds
    data['speed_from_last'] = np.where(data['seconds_since_last']==0,0,data['distance_from_last']/data['seconds_since_last'])
    data['speed_of_angle_from_last'] = np.where(data['seconds_since_last']==0,0,data['angle_from_last']/data['seconds_since_last'])

    #Rush and rebounds are labelled
    data['rush'] = np.where((data['event_type'].isin(fenwick_events))&(data['zone_code_last'].isin(['N','D']))&(data['x_adj']>25)&(data['seconds_since_last']<=5),1,0)
    data['rebound'] = np.where((data['event_type'].isin(fenwick_events))&(data['event_type_last'].isin(fenwick_events))&(data['seconds_since_last']<=2),1,0)

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
    
    #Misc variables
    data['empty_net'] = np.where((data['event_type'].isin(fenwick_events))&(data['event_goalie_id'].isna()),1,0)
    data['regular'] = (data['season_type']==2).astype(int)
    data['offwing'] = np.where(((data['y_adj']<0)&(data['event_player_1_hand']=='L'))|((data['y_adj']>=0)&(data['event_player_1_hand']=='R')),1,0)
    
    #Return: pbp data prepared to train and calculate the xG model
    return data

def wsba_xG(pbp, hypertune = False, train = False, model_path = "tools/xg_model/wsba_xg.joblib", train_runs = 20, cv_runs = 20):
    #Train and calculate the WSBA Expected Goals model

    #Add index for future merging
    pbp['event_index'] = pbp.index

    #Recalibrate coordinates
    pbp = scraping.adjust_coords(pbp)

    #Fix strengths
    pbp['strength_state'] = np.where((pbp['season_type']==3)&(pbp['period']>4),(np.where(pbp['event_team_abbr']==pbp['away_team_abbr'],pbp['away_skaters'].astype(str)+"v"+pbp['home_skaters'].astype(str),pbp['home_skaters'].astype(str)+"v"+pbp['away_skaters'].astype(str))),pbp['strength_state'])

    #Filter unwanted data:
    #Shots must occur in specified events and strength states, occur in open play, and have valid coordinates    
    pbp_prep = pbp.loc[(pbp['event_type'].isin(events))&
                   (pbp['strength_state'].isin(strengths))&
                   (pbp['x'].notna())&
                   (pbp['y'].notna())]

    #Prep Data
    data = prep_xG_data(pbp_prep)

    #Reduce to fenwick shots
    data = data.loc[data['event_type'].isin(fenwick_events)]
    
    #Convert to sparse
    data_sparse = sp.csr_matrix(data[[target]+continuous+boolean])

    #Target and Predictors
    is_goal_vect = data_sparse[:, 0].A
    predictors = data_sparse[:, 1:]

    #XGB DataModel
    xgb_matrix = xgb.DMatrix(data=predictors,label=is_goal_vect,feature_names=(continuous+boolean))

    if train:
        if hypertune:
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

            best_all.to_csv("tools/xg_model/testing/xg_model_training_runs.csv",index=False)

            # Final parameters
            param_7_EV = {
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "auc"],
                "gamma": best_all['gamma'].iloc[0],
                "subsample": best_all['subsample'].iloc[0],
                "max_depth": best_all['max_depth'].iloc[0],
                "colsample_bytree": best_all['colsample_bytree'].iloc[0],
                "min_child_weight": best_all['min_child_weight'].iloc[0],
                "max_delta_step": best_all['max_delta_step'].iloc[0],
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
            cv_final.to_csv("tools/xg_model/testing/xg_model_cv_runs.csv",index=False)
        else:
            # Load previous parameters
            best_all = pd.read_csv('tools/xg_model/testing/xg_model_training_runs.csv')
            cv_final = pd.read_csv("tools/xg_model/testing/xg_model_cv_runs.csv")

            print('Loaded hyperparameters...')
            # Final parameters
            param_7_EV = {
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "auc"],
                "gamma": best_all['gamma'].iloc[0],
                "subsample": best_all['subsample'].iloc[0],
                "max_depth": best_all['max_depth'].iloc[0],
                "colsample_bytree": best_all['colsample_bytree'].iloc[0],
                "min_child_weight": best_all['min_child_weight'].iloc[0],
                "max_delta_step": best_all['max_delta_step'].iloc[0],
            }

        print('Training model...')
        seed = int(cv_final['seed'].iloc[0])
        np.random.seed(seed)
        model = xgb.train(
            params=param_7_EV,
            dtrain=xgb_matrix,
            num_boost_round=int(cv_final['AUC_rounds'].iloc[0]),
            verbose_eval=2,
        )
        
        #Save model
        joblib.dump(model,model_path)
        
    else:
        model = joblib.load(model_path)

        #Predict goal
        data['xG'] = model.predict(xgb_matrix)

        #Drop previous xG if it exists
        pbp = pbp.drop(columns=['xG'],errors='ignore')

        #Merge
        comm = list(data.columns.intersection(pbp.columns))
        comm.remove('event_index')
        data = data.drop(columns=comm)
        pbp_xg = pd.merge(pbp,data,how='left')

        return pbp_xg

def feature_importance(model):
    print('Feature importance for WSBA xG Model...')
    model = joblib.load(model)

    xgb.plot_importance(model)
    plt.savefig('tools/xg_model/metrics/feature_importance.png',bbox_inches='tight')

def roc_auc_curve(pbp,model):
    print('ROC-AUC Curve for WSBA xG Model...')

    #Recalibrate coordinates
    pbp = scraping.adjust_coords(pbp)

    #Filter unwanted data:
    #Shots must occur in specified events and strength states, occur in open play, and have valid coordinates    
    pbp_prep = pbp.loc[(pbp['event_type'].isin(events))&
                   (pbp['strength_state'].isin(strengths))&
                   (pbp['period'] < 5)&
                   (pbp['x'].notna())&
                   (pbp['y'].notna())]

    pbp = prep_xG_data(pbp_prep) 
    model = joblib.load(model)

    data = pbp.loc[pbp['event_type'].isin(fenwick_events)]
    
    data_sparse = sp.csr_matrix(data[[target]+continuous+boolean])

    is_goal_vect = data_sparse[:, 0].A
    predictors = data_sparse[:, 1:]
    
    xgb_matrix = xgb.DMatrix(data=predictors,label=is_goal_vect,feature_names=(continuous+boolean))

    pred = model.predict(xgb_matrix)
    fpr, tpr, _ = roc_curve(is_goal_vect, pred)
    roc_auc = auc(fpr,tpr)
    
    plt.figure()
    plt.plot(fpr,tpr,label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("WSBA xG ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig('tools/xg_model/metrics/roc_auc_curve.png')

def reliability(pbp,model):
    print('Reliability for WSBA xG Model...')

    #Recalibrate coordinates
    pbp = scraping.adjust_coords(pbp)

    #Filter unwanted data:
    #Shots must occur in specified events and strength states, occur in open play, and have valid coordinates    
    pbp_prep = pbp.loc[(pbp['event_type'].isin(events))&
                   (pbp['strength_state'].isin(strengths))&
                   (pbp['period'] < 5)&
                   (pbp['x'].notna())&
                   (pbp['y'].notna())]

    pbp = prep_xG_data(pbp_prep) 
    model = joblib.load(model)

    data = pbp.loc[pbp['event_type'].isin(fenwick_events)]
    
    data_sparse = sp.csr_matrix(data[[target]+continuous+boolean])

    is_goal_vect = data_sparse[:, 0].A
    predictors = data_sparse[:, 1:]
    
    xgb_matrix = xgb.DMatrix(data=predictors,label=is_goal_vect,feature_names=(continuous+boolean))

    pred = model.predict(xgb_matrix)
    fop, mpv = calibration_curve(is_goal_vect, pred, strategy='uniform')

    plt.figure()
    plt.plot(mpv, fop, "s-", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.title("WSBA xG Reliability Diagram")
    plt.xlabel("Predicted Probability (mean)")
    plt.ylabel("Fraction of positives")
    plt.legend(loc="best")
    plt.savefig('tools/xg_model/metrics/reliability.png')