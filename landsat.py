import pandas as pd
import numpy as np
import df_selector

def preprocess_clf(df):
    return _preprocess(df,False)

def preprocess_reg(df):
    return _preprocess(df,True)

def _preprocess(df,is_reg):
    # Features
    features = df.iloc[:,0:-1].copy()
    # Target
    target_column = -1
    target=df.iloc[:,target_column].copy()

    full_pipeline = df_selector.create_num_pipeline(features)
    features_prepared = pd.DataFrame(data=full_pipeline.fit_transform(features))

    return features_prepared, target


