import sys
import numpy as np
import pandas as pd

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

