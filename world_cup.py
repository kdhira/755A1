import pandas as pd
import numpy as np
import df_selector

def preprocess_clf(df):
    return _preprocess(df,False)

def preprocess_reg(df):
    return _preprocess(df,True)

def _preprocess(df,is_reg):
    # Remove irrelevant columns
    df.drop(['Date','Location','Phase','Team1_Continent','Team2_Continent'],axis=1,inplace=True)

    # Features
    features = df.iloc[:,0:-2].copy()
    # Target
    target_column = -1
    if is_reg:
        target_column = -2
    target=df.iloc[:,target_column].copy()

    features_num = features.drop(['Team1','Team2','Normal_Time'], axis=1,inplace=False)
    features_cat = features[['Team1','Team2','Normal_Time']].copy()

    full_pipeline = df_selector.create_pipeline(features_num, features_cat)
    features_prepared = pd.DataFrame(data=full_pipeline.fit_transform(features))

    return features_prepared, target


