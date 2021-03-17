import os
import sys
import socket
import numpy as np
import pandas as pd
from scipy.stats import f

# Find where we ware
def find_dir_nsqip():
    dir_base = os.getcwd()
    cpu = socket.gethostname()
    # Set directory based on CPU name
    if cpu == 'RT5362WL-GGB':
        if os.name == 'posix':
            print('Predator (WSL)')
            dir_nsqip = '/mnt/d/projects/NSQIP'
        elif os.name == 'nt':
            print('Predator (Windows)')
            dir_nsqip = 'D:\\projects\\NSQIP'
        else:
            assert False
    elif cpu == 'snowqueen':
        print('On snowqueen machine')
        dir_nsqip = os.path.join(dir_base, '..')
    else:
        sys.exit('Where are we?!')
    return dir_nsqip

def gg_save(fn,fold,gg,width,height):
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)
    gg.save(path, width=width, height=height)

def rvec(x):
    return np.atleast_2d(x)

def cvec(x):
    return rvec(x).T

# df=qq.copy();cn_gg='gg';cn_vv='vv';cn_val='val'
"""
cn_gg : name of group column
cn_vv : variable column name
cn_val : value name to calculate variance
"""
def decomp_var(df, cn_gg, cn_vv, cn_val):
    # (i) Calculate the within group sum of squares
    res_w = df.copy().groupby([cn_vv,cn_gg]).apply(lambda x: 
       pd.Series({'SSw':np.sum((x[cn_val]-x[cn_val].mean())**2)})).reset_index()
    res_w = res_w.groupby(cn_vv).SSw.sum().reset_index()
    # (ii) Calculate the between group sum of squares
    res_b = df.copy().groupby([cn_vv,cn_gg]).apply(lambda x: 
                     pd.Series({'xbar':x[cn_val].mean(),'n':x[cn_val].shape[0]})).reset_index()
    res_b = res_b.merge(df.groupby(cn_vv)[cn_val].mean().reset_index().rename(columns={cn_val:'mu'}))
    res_b = res_b.assign(SSb=lambda x: x.n*(x.xbar - x.mu)**2).groupby(cn_vv).SSb.sum().reset_index()
    # (iii) Ensure it lines up (SStot == 0)
    res_tot = res_w.merge(res_b).assign(SStot=lambda x: x.SSw+x.SSb)
    # (iv) Under null of no difference between groups, should have an F-distribution after DoF adjustment
    tmp = df.groupby(cn_vv).apply(lambda x: pd.Series({'n':x.shape[0], 'k':x[cn_gg].unique().shape[0]})).reset_index()
    res_tot = res_tot.merge(tmp,'left',cn_vv)
    res_tot = res_tot.assign(dof_b = lambda x: x.k - 1, dof_w=lambda x: x.n-x.k)
    res_tot = res_tot.assign(Fstat=lambda x: (x.SSb/x.dof_b)/(x.SSw/x.dof_w))
    res_tot['pval'] = 1 - f.cdf(res_tot.Fstat, res_tot.dof_b, res_tot.dof_w)
    res_tot['gg'] = cn_gg
    return res_tot


def makeifnot(path):
    if not os.path.exists(path):
        print('Making path')
        os.makedir(path)

def gg_color_hue(n):
    from colorspace.colorlib import HCL
    hues = np.linspace(15, 375, num=n + 1)[:n]
    hcl = []
    for h in hues:
        hcl.append(HCL(H=h, L=65, C=100).colors()[0])
    return hcl

def stopifnot(check,stmt=None):
    if stmt is None:
        stmt = 'error! Statement is not True'
    if not check:
        sys.exit(stmt)


#fun = metrics.r2_score; data = [tmp_y_test ,yhat_test]; nbs=999
def bs_wrapper(fun,data,nbs=999,ss=1234):
    stat_bl = fun(*data) # baseline statistic
    stopifnot(len(np.unique([x.shape[0] for x in data]))==1)
    n = data[0].shape[0]
    np.random.seed(ss)
    stat_bs = np.repeat(0,nbs).astype(float)
    for ii in range(nbs):
        idx_ii = np.random.choice(n,n,replace=True)
        stat_bs[ii] = fun(*[x[idx_ii] for x in data])
    bias = stat_bs.mean() - stat_bl
    ci = np.append(stat_bl, np.quantile(stat_bs - bias, q=[0.025, 0.975]))
    return(ci)


def strat_split(lbls, nsplit):
    ulbls = np.unique(lbls)
    idx_lbls = [np.where(lbls == u)[0] for u in ulbls]
    idx_splits = [np.arange(len(z)) % nsplit for z in idx_lbls]
    [np.random.shuffle(z) for z in idx_splits]
    idx_lbls = [[idx[np.where(splits == kk)[0]] for kk in range(nsplit)] for
                splits, idx in zip(idx_splits, idx_lbls)]
    test_idx = [np.sort(np.concatenate([idx_lbls[jj][kk] for jj in range(len(ulbls))])) for kk in range(nsplit)]
    return (test_idx)


def rand_split(lbls, nsplit):
    idx_lbls = np.arange(len(lbls))
    idx_splits = idx_lbls % nsplit
    np.random.shuffle(idx_splits)
    idx_tests = [idx_lbls[np.where(idx_splits == kk)[0]] for kk in range(nsplit)]
    return (idx_tests)

# # Grid-Search class
# params = {'l1_ratio':np.arange(0.1,1,0.1), 'alpha':np.exp(np.linspace(np.log(1e-3),np.log(1),9))}
# estimator = ElasticNet(fit_intercept=True, normalize=False)
# self = gridsearch(param_dict=params, estimator=estimator, metric=auc,splitter=rand_split); nsamp=None; ss=1234; ncv=5
class gridsearch():
    def __init__(self, param_dict, estimator, metric, splitter):
        self.param_dict = param_dict
        self.estimator = estimator
        self.metric = metric
        self.splitter = splitter
        self.hypernames = list(self.param_dict.keys())
        self.hypervals = list(self.param_dict.values())
        gg = np.array(np.meshgrid(*self.hypervals)).reshape([len(self.hypernames),
                                                             np.prod([len(z) for z in self.hypervals])]).T
        self.hyperdf = pd.DataFrame(gg, columns=self.hypernames)
        stopifnot(all([hasattr(self.estimator, z) for z in self.hypernames]))

    def fit(self, X, y, ncv=5, nsamp=None, ss=1234):
        n = X.shape[0]
        stopifnot(n == y.shape[0])
        if nsamp is None:
            nsamp = self.hyperdf.shape[0]
        np.random.seed(ss)
        # Random hyperparameter index
        hidx = np.sort(np.random.choice(self.hyperdf.shape[0], nsamp, replace=False))
        self.hyperdf = self.hyperdf.loc[hidx].reset_index(drop=True)
        self.hyperdf.insert(self.hyperdf.shape[1], 'metric', np.NaN)
        idx_test = self.splitter(lbls=y, nsplit=ncv)
        idx_all = np.arange(n)
        self.metrics = np.zeros([nsamp, ncv]) * np.NaN
        for jj in range(ncv):
            #print('Fold %i of %i' % (jj + 1, ncv))
            idx_train_jj = ~np.in1d(idx_all, idx_test[jj])
            idx_test_jj = ~idx_train_jj
            for ii, rr in self.hyperdf.iterrows():
                [setattr(self.estimator, nn, rr[nn]) for nn in self.hypernames]
                self.estimator.fit(X=X[idx_train_jj], y=y[idx_train_jj])
                self.metrics[ii,jj] = self.metric(y[idx_test_jj], self.estimator.predict(X=X[idx_test_jj]))
        self.hyperdf['metric'] = self.metrics.mean(axis=1)
        slice = self.hyperdf.loc[self.hyperdf.metric.idxmax()]
        # Self winning model
        [setattr(self.estimator, nn, vv) for nn, vv in
         zip(list(slice[self.hypernames].index),list(slice[self.hypernames].values))]
        self.estimator.fit(X,y)

