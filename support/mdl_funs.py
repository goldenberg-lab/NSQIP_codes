import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss


# Sanity checker
def stopifnot(stmt):
    if not stmt:
        sys.exit('error! Statement is not True')

# Return some number of chunks with no index exceeding mbatch
def idx_iter(n ,mbatch):
    mbatch = min(n, mbatch)
    nfull = n // mbatch
    npart = (n % mbatch > 0) + 0
    idx_splits = np.split(np.arange(mbatch * nfull), nfull)
    if npart == 1:
        idx_splits = idx_splits + [np.arange(mbatch * nfull,n)]
    return (idx_splits)

"""
ENCODER FUNCTION TO NORMALIZE CONTINOUS VARIABLES AND ONE-HOT-ENCODE CATEGORICAL
"""
class normalize():
    def __init__(self,copy=False):
        self.copy = copy

    def fit(self,data):
        stopifnot(isinstance(data,pd.DataFrame)) # Needs to be dataframe
        self.p = data.shape[1]
        self.cidx = np.where(data.dtypes == 'object')[0]
        self.nidx = np.where(~(data.dtypes == 'object'))[0]
        self.cenc = ohe(sparse=False, dtype=int, handle_unknown='ignore', drop='first')
        #self.cenc.categories_ = [np.sort(data.iloc[:, x].unique()) for x in self.cidx]
        self.cenc.categories_ = [list(data.iloc[:, x].value_counts().index) for x in self.cidx]
        self.cenc.drop_idx_ = np.repeat(0, len(self.cenc.categories_))
        # Total feature size: categories + num
        self.p2 = sum([len(x) - 1 for x in self.cenc.categories_]) + len(self.nidx)
        self.nenc = ss(copy=self.copy)
        self.nenc.mean_ = data.iloc[:, self.nidx].mean().values
        self.nenc.scale_ = data.iloc[:, self.nidx].std().values
        self.cn = list(self.cenc.get_feature_names(data.columns[self.cidx].astype(str))) + \
                  data.columns[self.nidx].to_list()
# self=normalize();data=X_test_ii





