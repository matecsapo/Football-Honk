from goose.data.goose_data_structures.identifiers import League, Season

# current Season
current_season = Season(2026)

# list of leagues currently modelled + project by Football-Honk
# list of (league, active/inactive status)
modelled_leagues = [
        (League("ENG-Premier League"), True),
        (League("ESP-La Liga"), True),
        (League("GER-Bundesliga"), True),
        (League("ITA-Serie A"), True),
        (League("FRA-Ligue 1"), True)
    ]

# defines league --> (flagship model, model train operation) used for producing league's publicizied Football-Honk projections
from honk.models.static_poi_reg.train_league_sprm import train_league_sprm
flagship_models = {
    League("ENG-Premier League") : ("ENG-Premier League_sprm", train_league_sprm),
    League("ESP-La Liga") : ("ESP-La Liga_sprm", train_league_sprm),
    League("GER-Bundesliga") : ("GER-Bundesliga_sprm", train_league_sprm),
    League("ITA-Serie A") : ("ITA-Serie A_sprm", train_league_sprm),
    League("FRA-Ligue 1") : ("FRA-Ligue 1_sprm", train_league_sprm)
}

# API for Football-Data.org access
football_data_org_key = "c6bef419ed014946ba52c4249e2e7fa7"