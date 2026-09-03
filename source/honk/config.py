from goose.data.goose_data_structures.identifiers import League, Season

# current Season
current_season = Season(2026)

# set of models
# dict of model_name --> active/inactive status, refresh_model_op, refresh_projections_op
from honk.models.static_poi_reg.train_league_sprm import refresh_live_sprm, refresh_live_sprm_projections
from honk.models.weighted_poi_reg.weighted_poi_reg_train_scripts import refresh_live_wprm, refresh_live_wprm_projections

models = {
    "ENG-Premier League_sprm" : (True, refresh_live_sprm, refresh_live_sprm_projections),
    "ESP-La Liga_sprm" : (True, refresh_live_sprm, refresh_live_sprm_projections),
    "GER-Bundesliga_sprm" : (False, refresh_live_sprm, refresh_live_sprm_projections),
    "ITA-Serie A_sprm" : (True, refresh_live_sprm, refresh_live_sprm_projections),
    "FRA-Ligue 1_sprm" : (True, refresh_live_sprm, refresh_live_sprm_projections),
    "ENG-Premier League_wprm" : (True, refresh_live_wprm, refresh_live_wprm_projections),
    "ESP-La Liga_wprm" : (True, refresh_live_wprm, refresh_live_wprm_projections),
    "GER-Bundesliga_wprm" : (False, refresh_live_wprm, refresh_live_wprm_projections),
    "ITA-Serie A_wprm" : (True, refresh_live_wprm, refresh_live_wprm_projections),
    "FRA-Ligue 1_wprm" : (True, refresh_live_wprm, refresh_live_wprm_projections)
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