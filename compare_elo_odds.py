from datetime import timedelta, date
import requests
import json
import numpy as np
import pandas as pd
import time
import os
### OBJECTS NEEDED ###
NTST_API = os.getenv('NS_API_KEY')
sport_array = ["asiabb", "amerbb", "cbb", "eurobb", "gl", 
               "kbo", "khl", "mhk", "mlb", "milb", "mbb", 
               "mb2", "mb3", "nba", "mbia", "mbjc", "nhl",
               "npb", "wbb", "wb2", "wb3", "wbia", "wnba", 
               "pfb", "cfb"]

date_today = date.today()
### LEAGUES TO USE TOMORROW'S DATE IN ###
# AsiaBB
# KBO 
# NPB 

### FUNCTIONS NEEDED ###

## Function 1: Function to get league forecasts from National Statistical v3 API ##
## Uses packages: 
# requests
# from datetime import date, datetime, timedelta
## Inputs: 
# string value of league to check forecast of, used as URL query parameter
# date to use in URL  # This is parsed from system date
## Outputs: either a parsed JSON object or a signifier that the league had no data for the date requested

def getNatStatData(sport_name, api_key, game_date):
    try:
        game_date = str(game_date)
        res = requests.get(url=f"https://api3.natst.at/{api_key}/forecasts/{sport_name}/{game_date}")

        res.raise_for_status()

        ntst_json = res.json()

        return(ntst_json)
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error:{errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Connection Error:{errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error:{errt}")
    except requests.exceptions.RequestException as err:
        print(f"Oops Error: Something Else:{err}")

# Function from this answer https://stackoverflow.com/questions/25833613/safe-method-to-get-value-of-nested-dictionary?page=2&tab=scoredesc#tab-top
def deep_get(d: dict, *keys, default=np.nan):
    """ Safely get a nested value from a dict

    Example:
        config = {'device': None}
        deep_get(config, 'device', 'settings', 'light')
        # -> None
        
    Example:
        config = {'device': True}
        deep_get(config, 'device', 'settings', 'light')
        # -> TypeError

    Example:
        config = {'device': {'settings': {'light': 'bright'}}}
        deep_get(config, 'device', 'settings', 'light')
        # -> 'light'

    Note that it returns `default` is a key is missing or when it's None.
    It will raise a TypeError if a value is anything else but a dict or None.
    
    Args:
        d: The dict to descend into
        keys: A sequence of keys to follow
        default: Custom default value
    """
    # Descend while we can
    try:
        for k in keys:
            d = d[k]
    # If at any step a key is missing, return default
    except KeyError:
        return default
    # If at any step the value is not a dict...
    except TypeError:
        # ... if it's a None, return default. Assume it would be a dict.
        if d is None:
            return default
        # ... if it's something else, raise
        else:
            raise
    # If the value was found, return it
    else:
        return d

def parseELOMLValue(val):
    try:
        num_val = float(val) / 100
        return num_val
    except ValueError as ve:
        print(ve)
    except TypeError:
        return np.nan

def calcImpliedProb(val):
    try:
        if val < 0:
            val = abs(val)
            implied_prob = val / (1 + val)
        else:
            implied_prob = 1 / (1 + val)
        return implied_prob
    except ValueError as ve:
        print(ve)
    except TypeError:
        return np.nan

## Function 2: Function to grab all ELO prediction and moneyline odds available ##
## This includes:
# Sport: imputed from supplied acronym from function 1
# League: forecasts -> forecast_game -> League. If unavailable, will default to sport acronym.
# Home team name: forecasts -> forecast_gameid -> home 
# Away team name: forecasts -> forecast_gameid -> visitor
# Home team ELO: forecasts -> forecast_gameid -> forecast -> elo -> helowinexp parse as float, will need to be divided by 100, return NA if not found
# Away team ELO: forecasts -> forecast_gameid -> forecast -> elo -> velowinexp parse as float, will need to be divided by 100, return NA if not found
# Home team moneyline odds: forecasts -> forecast_gameid -> forecast -> moneyline -> vismoneyline parse as float (this is inverted for some reason), will need to be divided by 100, return NA if not found
# Away team moneyline odds: forecasts -> forecast_gameid -> forecast -> moneyline -> homemoneyline parse as float (this is inverted for some reason), will need to be divided by 100, return NA if not found
# Home team moneyline implied odds: calculated with function 3 with "Home team moneyline odds"
# Away team moneyline implied odds: calculated with function 3 with "Away team moneyline odds"
# Moneyline value index: calculated by taking quotient of either Home/Away team ELO and Home/Away team moneyline implied odds
## Uses packages: pandas

def makeELOMLGameDF(forecasts_json):
    league_name = forecasts_json.get("League", pd.NA)
    home_team = forecasts_json.get("home", pd.NA)
    away_team = forecasts_json.get("visitor", pd.NA)
    home_elo = parseELOMLValue(deep_get(forecasts_json, "forecast", "elo", "helowinexp"))
    away_elo = parseELOMLValue(deep_get(forecasts_json, "forecast", "elo", "velowinexp"))
    home_ml = parseELOMLValue(deep_get(forecasts_json, "forecast", "moneyline", "vismoneyline"))
    away_ml = parseELOMLValue(deep_get(forecasts_json, "forecast", "moneyline", "homemoneyline"))
    home_implied_prob = calcImpliedProb(home_ml)
    away_implied_prob = calcImpliedProb(away_ml)
    home_ml_val_idx = home_elo / home_implied_prob
    away_ml_val_idx = away_elo / away_implied_prob

    game_data = {
        'league_name': np.repeat(league_name, 2),
        'setting': ["home", "away"],
        'team_name': [home_team, away_team],
        'elo': [home_elo, away_elo],
        'ml': [home_ml, away_ml],
        'implied_odds': [home_implied_prob, away_implied_prob],
        'ml_value_idx': [home_ml_val_idx, away_ml_val_idx]
    }

    game_df = pd.DataFrame.from_dict(game_data)

    game_df.reset_index(drop=True, inplace=True)

    return game_df


def makeLeagueDF(sport_name, api_key, game_date):
    tomorrow_sports = ["asiabb", "kbo", "npb"]

    if sport_name in tomorrow_sports:
        game_date = game_date + timedelta(days=1)

    sport_json = getNatStatData(sport_name=sport_name, api_key=api_key, game_date=game_date)

    if sport_json["success"] == "0":
        if sport_json["error"]["message"] == "NO_DATA":
            print(sport_json["error"]["detail"])
            pass
        else:
            raise ValueError("NatStat API query error.")
    else:
        ntst_forecasts = sport_json.get("forecasts")

        ntst_data_list = []

        for game in ntst_forecasts:
            ntst_game_json = ntst_forecasts.get(game)
            ntst_game_df = makeELOMLGameDF(ntst_game_json)
            ntst_data_list.append(ntst_game_df)
    
        league_df = pd.concat(ntst_data_list)

        league_df.insert(loc=0, column="today_date", value=date.today())
        league_df.insert(loc=0, column="game_date", value=game_date)
        league_df.insert(loc=0, column="sport_name", value=sport_name)

        league_df.dropna(subset=['ml_value_idx'], inplace=True)

        return league_df

### SCRIPT EXECUTION STARTS ###

ntst_list = []

for i in range(len(sport_array)):
    sport = makeLeagueDF(sport_array[i], api_key=NTST_API, game_date=date_today)
    ntst_list.append(sport)
    print(f"Done with sport {sport_array[i]}")
    time.sleep(3)
 
ntst_df_today = pd.concat(ntst_list)

ntst_df_today.sort_values(by=['today_date', 'ml_value_idx'], ascending=False, inplace=True)

ntst_df_full = pd.read_csv("./ntst_value_data.csv", index_col=None)

ntst_df_combined = pd.concat([ntst_df_today, ntst_df_full])

print(ntst_df_combined)

ntst_df_combined.to_csv("./ntst_value_data.csv", index=False)

## Inputs: 
# JSON output from Function 1
## Outputs: A row of a pandas dataframe to be appended to the full data CSV file. Each row is a TEAM.
# Output columns
# Sport
# League
# Team name
# Setting (Home or Away)
# ELO
# Moneyline
# Implied Odds
# Moneyline Value Index
# Betting date
# Game date 

## Function 3: Function to calculate implied probabililty percentage from moneyline odds
## Inputs: Home/Away moneyline odds value
# Store if moneyline was positive or negative in a variable
# Convert only negative values to positive by multiplying by -1
# If value was positive:
# Result is the quotient of 1 / 1 + moneyline value
# If value was negative:
# Result is moneyline value / 1 + moneyline value
## Outputs: 
# Home/Away team moneyline implied odds 


### Run at ~11am EST! 


