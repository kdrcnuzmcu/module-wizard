
class HelperFunctions():
    def EDA(self, dataframe):
        print(f"Rows: {dataframe.shape[0]}")
        print(f"Columns: {dataframe.shape[1]}")
        print(" - - - - - ")

        print("HEAD")
        print(dataframe.head())
        print(" - - - - - ")
        print("TAIL")
        print(dataframe.tail())
        print(" - - - - - ")
        print("SAMPLES")
        print(dataframe.sample(5))



import pandas as pd
train = pd.read_csv(r"C:\Users\kdrcn\OneDrive\Masaüstü\Py\House-Price\train.csv")
HelperFunctions().EDA(train)