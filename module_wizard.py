# !pip install pandas
# !pip install numpy

# !pip install matplotlib
# !pip install seaborn
# !pip install qbstyles

# !pip install sklearn

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

# !pip install textblob
# !pip install nltk

# Embeddeds
import os
import time
import math
import re
import random

# Basics
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from qbstyles import mpl_style

# Models
## Regression
### Tree Models
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
### Linear Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso

## Classification
### Tree Models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
### Linear Models
from sklearn.linear_model import LogisticRegression

# Pre-Processing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Model Error Metrics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from textblob import TextBlob
from textblob import Word

from nltk.corpus import stopwords

# PANDAS SET_OPTIONS
class PandasOptions():
    Options = {
        "1": "display.max_columns",
        "2": "display.max_rows",
        "3": "display.width",
        "4": "display.expand_frame_repr",
        "5": "display.max_colwidth"
    }
    def PrintOptions(self):
        for key, value in self.Options.items():
            print(f"{key}: {value}")
    def SetOptions(self, *args):
        Choices = list(args)
        for i in Choices:
            # print(self.Options[str(i)])
            pd.set_option(self.Options[str(i)], None)
    def ResetOptions(self, *args):
        Choices = list(args)
        for i in Choices:
            # print(self.Options[str(i)])
            pd.reset_option(self.Options[str(i)])

class ImportModules():
    Profiles = {
        "sklearn.preprocessing": [
            "LabelEncoder",
            "OneHotEncoder",
            "StandardScaler",
            "RobustScaler",
            "MinMaxScaler"
        ],
        "sklearn.linear_model": [
            "LinearRegression",
            "Ridge",
            "ElasticNet",
            "Lasso"
        ],
        "sklearn.metrics": [
            "mean_squared_error",
            "mean_absolute_error",
            "mean_absolute_percentage_error"
        ],
        "sklearn.model_selection": [
            "train_test_split",
            "cross_val_score",
            "GridSearchCV",
            "RandomizedSearchCV",
            "TimeSeriesSplit"
        ]
    }
    def Basics(self):
        BasicModules = {
            "pandas": "pd",
            "numpy": "np",
            "matplotlib.pyplot": "plt",
            "seaborn": "sns"
        }
        for module, abbr in BasicModules.items():
            print(module, abbr)
            globals()[abbr] = __import__(module)
    def TreeModels(self, kind):
        if kind == "regressor":
            CBR = __import__("catboost", fromlist=["CatBoostRegressor"])
            LGBMR = __import__("lightgbm", fromlist=["LGBMRegressor"])
            XGBR = __import__("xgboost", fromlist=["XGBRegressor"])
            return CBR.CatBoostRegressor, LGBMR.LGBMRegressor, XGBR.XGBRegressor
        elif kind == "classification":
            globals()["CatBoostClassifier"] = __import__("catboost", fromlist=["CatBoostClassifier"])
            globals()["LGBMClassifier"] = __import__("lightgbm", fromlist=["LGBMClassifier"])
            globals()["XGBClassifier"] = __import__("xgboost", fromlist=["XGBClassifier"])
    def ImportModuleSet(self, profile):
        modules = self.Profiles[profile]
        for module in modules:
            print(module)
            __import__(profile,  fromlist=[module])


