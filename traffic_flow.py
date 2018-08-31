import pandas as pd
import numpy as np
import df_selector

def preprocess_clf(df):
    return _preprocess(df,False)

def preprocess_reg(df):
    return _preprocess(df,True)

def _preprocess(df,is_reg):
    # Features
    feature_columns = ['Segment_23(t-9)','Segment_23(t-8)','Segment_23(t-7)','Segment_23(t-6)','Segment_23(t-5)','Segment_23(t-4)','Segment_23(t-3)','Segment_23(t-2)','Segment_23(t-1)','Segment_23(t)']
    features = df.loc[:,feature_columns].copy()
    # Target
    target_column = -1
    target=df.iloc[:,target_column].copy()

    full_pipeline = df_selector.create_num_pipeline(features)
    features_prepared = pd.DataFrame(data=full_pipeline.fit_transform(features))

    return features_prepared, target


