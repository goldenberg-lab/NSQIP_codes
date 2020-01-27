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
def idx_iter(n ,mbatch, ss=None):
    mbatch = min(n, mbatch)
    nfull = n // mbatch
    npart = (n % mbatch > 0) + 0
    nmax = mbatch * nfull
    batch_idx = np.append(np.repeat(np.arange(nfull)+1,mbatch),np.repeat(nfull+npart,n-nmax))
    if ss is not None:
        np.random.seed(ss)
        batch_idx = np.random.choice(batch_idx,n,replace=False)
    return([np.where(batch_idx == ii)[0] for ii in np.arange(nfull+npart)+1])

"""
ENCODER FUNCTION TO NORMALIZE CONTINOUS VARIABLES AND ONE-HOT-ENCODE CATEGORICAL
"""

def x_batch(data, cidx, enc, iter):
    stopifnot(len(cidx) == len(enc) == len(iter))
    xmat = []
    for jj, check in enumerate(iter):
        if check:
            xmat.append(enc[jj].transform(data.iloc[:, cidx[jj]]))
    return(np.hstack(xmat))

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
        self.lst_enc = [self.cenc, self.nenc]
        self.lst_cidx = [self.cidx, self.nidx]
        self.lst_iter = [len(z) > 0 for z in self.lst_cidx]


    def transform(self,data,check=False):
        stopifnot(isinstance(data, pd.DataFrame))
        stopifnot(data.shape[1] == self.p)
        if not check:
            return(x_batch(data, self.lst_cidx, self.lst_enc, self.lst_iter))
        else: # check for new data types to encode
            new_vals = [list(np.setdiff1d(data.iloc[:, jj].unique(), uvals)) for
                        jj, uvals in zip(self.cidx, self.cenc.categories_)]
            diff_vals = np.where(np.array([len(z) for z in new_vals]) > 0)[0]
            if len(diff_vals) > 0:
                data = data.copy()  # protect columns from being overwritten
                print("new categorical values! Setting to default")
                for jj in diff_vals:
                    cjj = self.cidx[jj]  # column in reference
                    data.iloc[:, cjj] = np.where(data.iloc[:, cjj].isin(new_vals[jj]),
                                                 self.cenc.categories_[jj][0], data.iloc[:, cjj])
            return( x_batch(data, self.lst_cidx, self.lst_enc, self.lst_iter) )


# self=normalize();data=X_test_ii





