# list of leagues currently modelled + project by Football-Honk
modelled_leagues = [
        "ENG-Premier League",
        #"ESP-La Liga",
        "GER-Bundesliga",
        "ITA-Serie A",
        "FRA-Ligue 1"
    ]

# defines league --> (flagship model, model train operation) used for producing league's publicizied Football-Honk projections
from honk.models.model_train_scripts.train_sprm import train_sprm
flagship_models = {
    "ENG-Premier League" : ("ENG-Premier League_sprm", train_sprm),
    #"ESP-La Liga" : ("ESP-La Liga_sprm", train_sprm),
    "GER-Bundesliga" : ("GER-Bundesliga_sprm", train_sprm),
    "ITA-Serie A" : ("ITA-Serie A_sprm", train_sprm),
    "FRA-Ligue 1" : ("FRA-Ligue 1_sprm", train_sprm)
}