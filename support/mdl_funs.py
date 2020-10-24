import numpy as np
import pandas as pd
import sys
import re
from functools import reduce
import itertools
import scipy as sp

from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn.feature_extraction.text import CountVectorizer as cv

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


# Fast way to append lists
def ljoin(x):
    return(list(itertools.chain.from_iterable(x)))

# Split strings based on |
def tok_fun(ss):
    if ss == 'nan':
        ss = ''
    return (re.sub('[^a-z\\s\\|]', '', ss.lower()).split('|'))

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

class col_encoder():

    def __init__(self,dropfirst=True,quantize=False, sparse=False,
                 nbins=10, dtype=float):
        self.dropfirst = dropfirst
        self.quantize = quantize
        self.nbins = nbins
        self.sparse = sparse
        self.dtype = dtype

    def fit(self, x): # Fit the encoder/scaler
        self.n = x.shape[0]
        self.p = x.shape[1]
        dt1 = pd.Series([type(x.iloc[0][kk]).__name__ for kk in range(self.p)])
        dt2 = x.dtypes.astype(str).reset_index(drop=True)
        self.dt = pd.Series(np.where(dt1.isin(['int64', 'float64']) & dt2.isin(['int64', 'float64']),'float','str'))
        if not all(self.dt.values == 'float'):
                self.dt[~(self.dt.values == 'float')] = \
                    np.where(x.loc[:, ~(self.dt.values == 'float')].apply(lambda x: x.str.contains('\\|', na=False).any()),
                     'lst',self.dt[~(self.dt.values == 'float')])
        self.cn = np.array(x.columns)
        stopifnot( all( self.dt.isin(['float','lst','str']) ) )
        self.cidx = np.where(self.dt == 'str')[0]
        self.nidx = np.where(self.dt == 'float')[0]
        self.tidx = np.where(self.dt == 'lst')[0]
        stopifnot( all(np.sort(reduce(np.union1d, [self.cidx, self.nidx, self.tidx])) == np.arange(self.p)) )
        self.iter = {'cenc':True, 'nenc': True, 'tenc': True}
        self.all_enc = {}

        #############################################################
        # --- Encoder (i): Categorical/ordinal integer features --- #

        if len(self.cidx)>0:
            self.cenc = ohe(sparse=self.sparse,dtype=self.dtype,
                            handle_unknown='ignore',drop=None)
            self.cenc.categories_ = [np.unique(x.iloc[:, kk]) for kk in self.cidx]
            self.cmode = [x.iloc[:, kk].mode()[0] for kk in self.cidx]
            cmode_idx = np.array([np.where(vec == mm)[0][0] for vec, mm in zip(self.cenc.categories_, self.cmode)])
            cum_idx = np.append([0],np.cumsum([len(z) for z in self.cenc.categories_]))
            if self.dropfirst:
                self.cenc.drop_idx = [np.arange(s1,s2)[idx] for s1, s2, idx in zip(cum_idx[:-1],cum_idx[1:], cmode_idx)]
            else:
                self.cenc.drop_idx = []
            self.cenc.p = cum_idx.max() - len(self.cenc.drop_idx) # How many features after dropping most common
            self.cenc.cn = list(np.delete(self.cenc.get_feature_names(self.cn[self.cidx]),self.cenc.drop_idx))
            self.all_enc['cenc'] = self.cenc
        else:
            self.iter['cenc'] = False

        ###############################################
        # --- Encoder (ii): Continuous numerical ---- #

        if len(self.nidx) > 0:
            if self.quantize:
                u_nidx = np.array([len(x.iloc[:,kk].unique()) for kk in self.nidx])
                self.nidx1 = self.nidx[u_nidx>31] # quantize
                self.nidx2 = self.nidx[u_nidx <= 31] # one-hot-encode
                self.nenc = {'enc':{},'cn':{}}
                if len(self.nidx1) > 0:
                    self.nenc1 = KD(n_bins=self.nbins, strategy='quantile')
                    if not self.sparse:
                        self.nenc1.encode = 'onehot-dense'
                    self.nenc1.fit(x.iloc[:, self.nidx1])
                    self.nenc1.cn = ljoin([cn+'_q'+pd.Series(qq).astype(str) for cn, qq in
                      zip(self.cn[self.nidx1],[np.arange(len(z)-1)+1 for z in self.nenc1.bin_edges_])])
                    self.nenc['enc']['nenc1'] = self.nenc1
                    self.nenc['cn']['nenc1'] = self.nenc1.cn
                if len(self.nidx2) > 0:
                    self.nenc2 =  ohe(sparse=self.sparse,handle_unknown='ignore',drop=None)
                    self.nenc2.fit(x.iloc[:, self.nidx2])
                    self.nenc2.cn = self.nenc2.get_feature_names(self.cn[self.nidx2])
                    self.nenc['enc']['nenc2'] = self.nenc2
                    self.nenc['cn']['nenc2'] = self.nenc2.cn
                self.nenc['cn'] = ljoin(list(self.nenc['cn'].values()))
                self.all_enc['nenc'] = self.nenc
            else:
                self.nenc = ss(copy=False)
                self.nenc.mean_ = x.iloc[:, self.nidx].mean(axis=0).values
                self.nenc.scale_ = x.iloc[:, self.nidx].std(axis=0).values
                self.nenc.n_features_in_ = self.nidx.shape[0]
                self.nenc.p = self.nidx.shape[0]
                self.nenc.cn = list(self.cn[self.nidx])
                self.all_enc['nenc'] = self.nenc
        else:
            self.iter['nenc'] = False

        ################################################
        # --- Encoder (iii): Tokenize text blocks ---- #

        if len(self.tidx) > 0:
            self.tenc = dict(zip(self.cn[self.tidx],
                [cv(tokenizer=lambda x: tok_fun(x),lowercase=False,
            token_pattern=None, binary=True) for z in range(self.tidx.shape[0])]))
            self.tenc = {'cv': self.tenc}
            for kk, jj in enumerate(self.cn[self.tidx]):
                self.tenc['cv'][jj].fit(x.loc[:,jj].astype('U'))
            self.tenc['p'] = sum([len(z.vocabulary_) for z in self.tenc['cv'].values()])
            self.tenc['cn'] = ljoin([l+'_'+pd.Series(list(z.vocabulary_.keys())) for
                                     z,l in zip(self.tenc['cv'].values(),self.tenc['cv'].keys())])
            self.all_enc['tenc'] = self.tenc
        else:
            self.iter['tenc'] = False

        # Store all in dictionary to iteration over self.iter
        self.enc_transform = {'cenc':self.cenc_transform, 'nenc':self.nenc_transform, 'tenc':self.tenc_transform}
        # Get the valid categories
        self.tt = np.array(list(self.iter.keys()))[np.where(list(self.iter.values()))[0]]
        # Get full feature names
        cn = []
        for ee in self.tt:
            if hasattr(self.all_enc[ee], 'cn'):
                cn.append(self.all_enc[ee].cn)
            else:
                cn.append(self.all_enc[ee]['cn'])
        cn = ljoin(cn)
        self.cn_transform = cn

    def cenc_transform(self,x):
        tmp = self.cenc.transform(x.iloc[:,self.cidx])
        if self.sparse:
            tmp = tmp.tocsc()[:,np.setdiff1d(np.arange(tmp.shape[1]),self.cenc.drop_idx)]
        else:
            tmp = np.delete(tmp,self.cenc.drop_idx,1)
        return(tmp)

    def nenc_transform(self,x):
        if self.quantize:
            mat = []
            if len(self.nidx1) > 0:
                tmp = self.nenc['enc']['nenc1'].transform(x.iloc[:, self.nidx1])
                if not self.sparse:
                    tmp = tmp.astype(int)
                mat.append(tmp)
            if len(self.nidx2) > 0:
                mat.append(self.nenc['enc']['nenc2'].transform(x.iloc[:, self.nidx2]))
            if self.sparse:
                return(sp.sparse.hstack(mat))
            else:
                return(np.hstack(mat))
        else:
            tmp = self.nenc.transform(x.iloc[:, self.nidx])
            if self.sparse:
                return(sp.sparse.csc_matrix(tmp))
            return(tmp)

    def tenc_transform(self,x):
        if self.sparse:
            return(sp.sparse.hstack([self.tenc['cv'][jj].transform(x.loc[:, jj].astype('U')) for jj in self.cn[self.tidx]]))
        else:
            return(np.hstack([self.tenc['cv'][jj].transform(x.loc[:, jj].astype('U')).toarray() for jj in self.cn[self.tidx]]))

    # Transform new example
    def transform(self,x):
        stopifnot(isinstance(x, pd.DataFrame))
        stopifnot(x.shape[1] == self.p)
        holder = []
        for ee in self.tt:
            holder.append(self.enc_transform[ee](x))
        if self.sparse:
            mat = sp.sparse.hstack(holder)
        else:
            mat = np.hstack(holder)
        return (mat)

class normalize():
    def __init__(self,drop='first'):
        self.drop = drop

    def fit(self,data):
        if not isinstance(data,pd.DataFrame): # Needs to be dataframe
            data = pd.DataFrame(data)
        self.p = data.shape[1]
        self.cidx = np.where(data.dtypes == 'object')[0]
        self.nidx = np.where(~(data.dtypes == 'object'))[0]
        self.cenc = ohe(sparse=False, dtype=int, handle_unknown='ignore',
                        drop=self.drop)
        self.cenc.categories_ = [list(data.iloc[:, x].value_counts().index) for
                                 x in self.cidx]
        self.cenc.drop_idx_ = np.repeat(0, len(self.cenc.categories_))
        # Total feature size: categories + num
        self.p2 = sum([len(x) - 1 for x in self.cenc.categories_]) + len(self.nidx)
        self.nenc = ss()
        self.nenc.mean_ = data.iloc[:, self.nidx].mean().values
        self.nenc.scale_ = data.iloc[:, self.nidx].std().values
        self.nenc.n_features_in_ = self.nidx.shape[0]
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





