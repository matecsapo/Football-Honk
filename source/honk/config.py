from goose.data.goose_data_structures.identifiers import League, Season

# current Season
current_season = Season(2026)

# set of models
# dict of model_name --> league, active/inactive status, refresh function
from honk.refresh import refresh_live_sprm, refresh_live_wprm 

models = {
    "ENG-Premier League_sprm" : (League("ENG-Premier League"), True, refresh_live_sprm),
    "ESP-La Liga_sprm" : (League("ESP-La Liga"), True, refresh_live_sprm),
    "GER-Bundesliga_sprm" : (League("GER-Bundesliga"), False, refresh_live_sprm),
    "ITA-Serie A_sprm" : (League("ITA-Serie A"), True, refresh_live_sprm),
    "FRA-Ligue 1_sprm" : (League("FRA-Ligue 1"), True, refresh_live_sprm),
    "ENG-Premier League_wprm" : (League("ENG-Premier League"), True, refresh_live_wprm),
    "ESP-La Liga_wprm" : (League("ESP-La Liga"), True, refresh_live_wprm),
    "GER-Bundesliga_wprm" : (League("GER-Bundesliga"), False, refresh_live_wprm),
    "ITA-Serie A_wprm" : (League("ITA-Serie A"), True, refresh_live_wprm),
    "FRA-Ligue 1_wprm" : (League("FRA-Ligue 1"), True, refresh_live_wprm)
}

# flagship models for each supported league
    # dict of league --> model_name
    # these models' projections are shown by honk_app
flagship_models = {
    League("ENG-Premier League") : "ENG-Premier League_wprm",
    League("ESP-La Liga") : "ESP-La Liga_wprm",
    League("GER-Bundesliga") : "GER_Bundesliga_wprm",
    League("ITA-Serie A") : "ITA-Serie A_wprm",
    League("FRA-Ligue 1") : "FRA-Ligue 1_wprm"
}