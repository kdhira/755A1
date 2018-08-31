import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames in this wise manner yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

def _create_num_pipeline(num_features, strat='median'):
    return Pipeline([
        ('selector', DataFrameSelector(list(num_features))),
        ('imputer', Imputer(strategy=strat)),
        ('std_scaler', StandardScaler()),
    ])

def _create_cat_pipeline(cat_features, drop_invar=True):
    return Pipeline([
        ('selector', DataFrameSelector(list(cat_features))),
        ('cat_encoder', cs.OneHotEncoder(drop_invariant=drop_invar)),
    ])

def create_pipeline(num_features, cat_features, strat='median', drop_invar=True):
    return FeatureUnion(transformer_list=[
        ("num_pipeline", _create_num_pipeline(num_features, strat)),
        ("cat_pipeline", _create_cat_pipeline(cat_features, drop_invar)),
    ])

def create_num_pipeline(num_features, strat='median'):
    return FeatureUnion(transformer_list=[
        ("num_pipeline", _create_num_pipeline(num_features, strat))
    ])

def create_cat_pipeline(cat_features, drop_invar=True):
    return FeatureUnion(transformer_list=[
        ("cat_pipeline", _create_cat_pipeline(cat_features, drop_invar)),
    ])
