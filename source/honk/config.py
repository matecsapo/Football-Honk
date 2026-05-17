from goose.data.goose_data_structures import League

# list of leagues currently modelled + project by Football-Honk
# list of (league, active/inactive status)
modelled_leagues = [
        (League("ENG-Premier League"), True),
        (League("ESP-La Liga"), False),
        (League("GER-Bundesliga"), False),
        (League("ITA-Serie A"), True),
        (League("FRA-Ligue 1"), False)
    ]

# defines league --> (flagship model, model train operation) used for producing league's publicizied Football-Honk projections
from honk.models.model_train_scripts.train_sprm import train_sprm
flagship_models = {
    League("ENG-Premier League") : ("ENG-Premier League_sprm", train_sprm),
    League("ESP-La Liga") : ("ESP-La Liga_sprm", train_sprm),
    League("GER-Bundesliga") : ("GER-Bundesliga_sprm", train_sprm),
    League("ITA-Serie A") : ("ITA-Serie A_sprm", train_sprm),
    League("FRA-Ligue 1") : ("FRA-Ligue 1_sprm", train_sprm)
}