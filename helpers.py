#region Imports
# Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
# sklearn.model_selection
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
# sklearn.linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import make_scorer
# Tree Regression Models
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# Tree Classification Models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# Others
from qbstyles import mpl_style
from typing import Optional
from module_wizard import PandasOptions
#endregion

class HelperFunctions():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def QuickView(self):
        print(f"""
Rows: {self.dataframe.shape[0]}
Columns: {self.dataframe.shape[1]}
* * *

HEAD:
{self.dataframe.head()}
* * *

TAIL:
{self.dataframe.tail()}
* * *

SAMPLES:
{self.dataframe.sample(5)}
        """)

    def Variables(self):
        dtypes = self.dataframe.dtypes
        nulls = self.dataframe.isnull().sum()
        counts = self.dataframe.count()
        fulls = self.dataframe.notnull().sum() / self.dataframe.shape[0]
        nunique = self.dataframe.nunique()
        means = self.dataframe.describe(include="all").T["mean"]
        medians = self.dataframe.describe(include="all").T["50%"]
        stds = self.dataframe.describe(include="all").T["std"]
        mins = self.dataframe.describe(include="all").T["min"]
        maxs = self.dataframe.describe(include="all").T["max"]
        print(
            pd.DataFrame({"DTypes": dtypes,
                          "Nulls": nulls,
                          "Counts": counts,
                          "Non-Nulls": fulls,
                          "Number-Of-Uniques": nunique,
                          "Means": means,
                          "Medians": medians,
                          "Std-Deviation": stds,
                          "Minimums": mins,
                          "Maximums": maxs})
        )

    def GrabColNames(self,
                     cat_th=10,
                     car_th=20,
                     verbose=False):
        cat_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in self.dataframe.columns if
                       self.dataframe[col].nunique() < cat_th and self.dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in self.dataframe.columns if
                       self.dataframe[col].nunique() > car_th and self.dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # Numerical Columns
        num_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        # Results
        if verbose:
            print(f"Observations: {self.dataframe.shape[0]}")
            print(f"Variables: {self.dataframe.shape[1]}")
            print(f'cat_cols: {len(cat_cols)}')
            print(f'num_cols: {len(num_cols)}')
            print(f'cat_but_car: {len(cat_but_car)}')
            print(f'num_but_cat: {len(num_but_cat)}')

        return cat_cols, num_cols, cat_but_car

    def CategoricalsByTarget(self,
                             col,
                             target,
                             rare: Optional[float] = None):
        temp = self.dataframe.groupby(col, dropna=False).agg(Count=(col, lambda x: x.isnull().count()), \
                                                             Ratio=(
                                                                 col,
                                                                 lambda x: x.isnull().count() / len(self.dataframe)), \
                                                             Target_Ratio=(
                                                                 target,
                                                                 lambda x: x.sum() / self.dataframe[target].sum())) \
            .sort_values("Count", ascending=False).reset_index()
        if rare is not None:
            rares = temp.loc[temp["Ratio"] <= float(rare), col].tolist()
            self.dataframe.loc[self.dataframe[col].isin(rares), col] = "Rare Category"
            print("---- Done! --- ")
            print(self.dataframe.groupby(col).agg(Count=(col, lambda x: x.count()), \
                                                  Ratio=(col, lambda x: x.count() / len(self.dataframe)), \
                                                  Target_Ratio=(
                                                      target, lambda x: x.sum() / self.dataframe[target].sum())) \
                  .sort_values("Count", ascending=False).reset_index(), "\n")
        else:
            print(temp, "\n")

    def Outliers(self,
                 col,
                 low_Quantile=0.25,
                 high_Quantile=0.75,
                 adjust=False):
        Q1 = self.dataframe[col].quantile(low_Quantile)
        Q3 = self.dataframe[col].quantile(high_Quantile)
        IQR = Q3 - Q1
        low_Limit = Q1 - (1.5 * IQR)
        up_Limit = Q3 + (1.5 * IQR)

        if len(self.dataframe[self.dataframe[col] > up_Limit]) > 0:
            print(col, ": Higher Outlier!")
        if len(self.dataframe[self.dataframe[col] < low_Limit]) > 0:
            print(col, ": Lower Outlier!")

        if adjust:
            self.dataframe.loc[(self.dataframe[col] < low_Limit), col] = low_Limit
            self.dataframe.loc[(self.dataframe[col] > up_Limit), col] = up_Limit
            print(f"{col}: Done!")

    def ExtractFromDatetime(self,
                            col,
                            year=True,
                            month=True,
                            day=True,
                            hour=False,
                            week=False,
                            dayofweek=False,
                            names=False):
        self.dataframe[col] = pd.to_datetime(self.dataframe[col])
        if year:
            self.dataframe["YEAR"] = self.dataframe[col].dt.year
        if month:
            if names:
                self.dataframe["MONTH_NAME"] = self.dataframe[col].dt.month_name()
            else:
                self.dataframe["MONTH"] = self.dataframe[col].dt.month
        if day:
            self.dataframe["DAY"] = self.dataframe[col].dt.day
        if hour:
            self.dataframe["HOUR"] = self.dataframe[col].dt.hour
        if week:
            self.dataframe["WEEK"] = self.dataframe[col].dt.isocalendar().week
        if dayofweek:
            if names:
                self.dataframe["DAYOFWEEK_NAME"] = self.dataframe[col].dt.day_name()
            else:
                self.dataframe["DAYOFWEEK"] = self.dataframe[col].dt.dayofweek + 1
    def Errors(self,
               y_true,
               y_pred):
        print("MAPE:", mean_absolute_percentage_error(y_true, y_pred))
        print("MAE:", mean_absolute_error(y_true, y_pred))
        print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))
    def FitModel(self,
                 estimator,
                 X,
                 y,
                 tag="Train"):
        print(f"# * ~  -- * -- --{estimator.__class__.__name__}-- -- * -- ~ * #")
        model = estimator.fit(X, y)
        print("Train Score:", model.score(X, y))
        y_pred = model.predict(X)
        print(f"# * ~  -- --{tag} Dataset-- -- ~ * #")
        self.Errors(y, y_pred)

    def FeatureImportance(self,
                          model,
                          features,
                          num=10):
        mpl_style(dark=True)
        FI = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
        plt.figure(num=f"{model.__class__.__name__}", figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="Value", y="Feature", data=FI.sort_values(by="Value", ascending=False)[0:num])
        plt.title("Features")
        plt.tight_layout()
        plt.get_current_fig_manager()
        plt.show()

    def Scaler(self,
               col,
               scaler):
        Scalers = {
            "standard": StandardScaler,
            "robust": RobustScaler,
            "minmax": MinMaxScaler
        }
        S = Scalers[scaler]
        new_col = S().fit_transform(self.dataframe[[col]])
        new_col = [np.round(x[0], decimals=4) for x in new_col]
        self.dataframe[col] = new_col