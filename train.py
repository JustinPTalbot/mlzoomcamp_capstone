import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import pickle

def ingest_csv():
    df = pd.read_csv('EV_cars.csv')
    return df

def data_prep(df):
    df.columns = df.columns.str.lower()
    df.rename({"price.de.":"price","acceleration..0.100.":"acceleration0to100"}, inplace = True, axis = 1)

    # We want price to be the target variable, so we'll have to remove records where price is null
    df = df[~df["price"].isna()]

    # Let's remove any rows w/ nulls since there are very few of them
    df = df[~df["fast_charge"].isna()]

    # Let's just grab the numeric columns
    df = df[["battery","efficiency","fast_charge","range","top_speed","acceleration0to100","price"]]

    # dropping target variable
    y = np.log(df.price)
    del df['price']

    # vectorizing data
    dv = DictVectorizer(sparse=False)
    data_dict = df.to_dict(orient='records')
    data = dv.fit_transform(data_dict)

    # converting to DMatrix
    features = dv.feature_names_
    DMatrix = xgb.DMatrix(data, label = y, feature_names = features)
    return DMatrix, dv

def train_xgboost(DMatrix):
    xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
    }

    model_final = xgb.train(xgb_params, DMatrix, num_boost_round=35)
    return model_final

if __name__ == "__main__":
    df = ingest_csv()
    DMatrix, dv = data_prep(df)
    model = train_xgboost(DMatrix)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("dv.pkl", "wb") as f:
        pickle.dump(dv, f)
