import pandas as pd

class HelperFunctions():
    from typing import Optional
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
        print(f"""
{self.dataframe.info()} 
* * * 

NUMBER OF NULLS:
{self.dataframe.isnull().sum()}        
* * * 

NUMBER OF UNIQUES:
{self.dataframe.nunique()}   
* * *   
        """)

    def GrabColNames(self, cat_th=10, car_th=20, verbose=False):
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

        return cat_cols, cat_but_car, num_cols

    def CategoricalsByTarget(self, col, target, rare: Optional[float] = None):
        temp = self.dataframe.groupby(col, dropna=False).agg(Count=(col, lambda x: x.isnull().count()), \
                                                             Ratio=(col, lambda x: x.isnull().count() / len(self.dataframe)), \
                                                             Target_Ratio=(target, lambda x: x.sum() / self.dataframe[target].sum())) \
                                                        .sort_values("Count", ascending=False).reset_index()
        if rare is not None:
            rares = temp.loc[temp["Ratio"] <= float(rare), col].tolist()
            self.dataframe.loc[self.dataframe[col].isin(rares), col] = "Rare Category"
            print("---- Done! --- ")
            print(self.dataframe.groupby(col).agg(Count=(col, lambda x: x.count()), \
                                                  Ratio=(col, lambda x: x.count() / len(self.dataframe)), \
                                                  Target_Ratio=(target, lambda x: x.sum() / self.dataframe[target].sum())) \
                                             .sort_values("Count", ascending=False).reset_index(), "\n")
        else:
            print(temp, "\n")

    def Outliers(self, col, low_Quantile=0.25, high_Quantile=0.75, adjust=False):
        Q1 = self.dataframe[col].quantile(low_Quantile)
        Q3 = self.dataframe[col].quantile(high_Quantile)
        IQR = Q3 - Q1
        low_Limit = Q1 - (1.5 * IQR)
        up_Limit = Q3 + (1.5 * IQR)

        if len(self.dataframe[self.dataframe[col] > up_Limit]) > 0:print(col, ": Higher Outlier!")
        if len(self.dataframe[self.dataframe[col] < low_Limit]) > 0:print(col, ": Lower Outlier!")

        if adjust:
            self.dataframe.loc[(self.dataframe[col] < low_Limit), col] = low_Limit
            self.dataframe.loc[(self.dataframe[col] > up_Limit), col] = up_Limit
            print(f"{col}: Done!")

    def ExtractFromDatetime(self, col):
        self.dataframe[col] = pd.to_datetime(self.dataframe[col])
        self.dataframe["YEAR"] = self.dataframe[col].dt.year
        self.dataframe["MONTH"] = self.dataframe[col].dt.month
        self.dataframe["DAY"] = self.dataframe[col].dt.day
        self.dataframe["DAYOFWEEK"] = self.dataframe[col].dt.dayofweek + 1
        self.dataframe["HOUR"] = self.dataframe[col].dt.hour
        self.dataframe["WEEK"] = self.dataframe[col].dt.isocalendar().week
