import numpy as np
from scipy.stats import norm
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from statsmodels.stats import multitest
from support.support_funs import stopifnot
from support.mdl_funs import normalize, idx_iter

def pred_batch(data, n, splits, mdls, cidx, enc, iter):
    stopifnot(len(cidx) == len(mdls) == len(enc) == len(iter))
    pmat = []
    for ii in range(n):
        ridx = splits[ii]
        holder = []
        for jj, check in enumerate(iter):
            if check:
                x_ii = enc[jj].transform(data.iloc[ridx, cidx[jj]])
                holder.append(mdls[jj].predict_proba(x_ii)[:, 1:])
        pmat.append(np.hstack(holder))
    return(np.vstack(pmat))

def x_batch(data, n, splits, cidx, enc, iter):
    stopifnot(len(cidx) == len(enc) == len(iter))
    xmat = []
    for ii in range(n):
        ridx = splits[ii]
        holder = []
        for jj, check in enumerate(iter):
            if check:
                holder.append(enc[jj].transform(data.iloc[ridx, cidx[jj]]))
        xmat.append(np.hstack(holder))
    return(np.vstack(xmat))


#self = mbatch_NB(method='gaussian');mbatch=100000;data=Xtrain;lbls=ytrain.values
class mbatch_NB():
    def __init__(self, method='bernoulli'):
        self.method = method.lower()
        stopifnot(self.method in ['bernoulli', 'gaussian'])
        self.fun_methods = {'bernoulli':{'mdl':BernoulliNB(fit_prior=False, binarize=0),
                                         'fit':self.fit_bernoulli,
                                         'predict':self.predict_bernoulli},
                            'gaussian':{'mdl':GaussianNB(),
                                        'fit':self.fit_gaussian,
                                        'predict':self.predict_gaussian}} #,'fit':self.fit_gaussian

    def fit(self, data, lbls, mbatch=100):
        self.n, self.p = data.shape
        stopifnot(len(lbls) == self.n)
        self.enc = normalize(copy=False)
        self.enc.fit(data)  # Get the one-hot encoders
        # Make lists of the indices
        self.lst_enc = [self.enc.cenc, self.enc.nenc]
        self.lst_cidx = [self.enc.cidx, self.enc.nidx]
        self.lst_iter = [len(z) > 0 for z in self.lst_cidx]
        # For each feature we need: sum(x), sum(x**2), sum(x,y)
        self.fun_methods[self.method]['fit'](data,lbls,mbatch)

    def fit_bernoulli(self,data,lbls,mbatch):
        self.ulbls = np.unique(lbls)
        lidx = [np.where(lbls == kk)[0] for kk in self.ulbls]
        midx = [z.min() for z in lidx]
        # Add model list for stacking
        self.lst_mdls = [BernoulliNB(), GaussianNB()]
        # If there is not cidx/nidx, it will skipp
        for jj, check in enumerate(self.lst_iter):
            if check:
                self.lst_mdls[jj].fit(X=self.lst_enc[jj].transform(data.iloc[midx, self.lst_cidx[jj]]),y=lbls[midx])

        idx_splits = idx_iter(self.n , mbatch)
        niter = len(idx_splits)
        # fit Naive Bayes
        for ii in range(niter):
            if (ii + 1) % np.ceil(niter / 10).astype(int) == 0:
                print('----- Training NB: batch %i of %i -----' % (ii + 1, niter))
            ridx = np.setdiff1d(idx_splits[ii], midx) # no double counting for midx
            y_ii = lbls[ridx]
            for jj, check in enumerate(self.lst_iter):
                if check:
                    x_ii = self.lst_enc[jj].transform(data.iloc[ridx, self.lst_cidx[jj]])
                    self.lst_mdls[jj].partial_fit(x_ii, y_ii, self.ulbls)

        print('----- Learning stacker -----')
        phat = pred_batch(data,niter,idx_splits,self.lst_mdls,self.lst_cidx,self.lst_enc,self.lst_iter)
        self.stacker = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs')
        self.stacker.fit(phat, lbls)

    def fit_gaussian(self,data,lbls,mbatch):
        mbatch = min(self.n, mbatch)
        niter = self.n // mbatch + (self.n % mbatch > 0)
        lbls = lbls.reshape([self.n,1])
        # Use tensors for false np.linalg.tensorsolve
        self.di_moments = {'gram':np.zeros([2,2,self.enc.p2]),
                           'igram': np.zeros([2, 2, self.enc.p2]),
                           'inner':np.zeros([2,1,self.enc.p2])}
        # Top right of gram, and top of inner and fixed independent of X
        self.di_moments['gram'][0,0,:] = self.n
        self.di_moments['inner'][0,0,:] = sum(lbls)
        for ii in range(niter):
            if (ii + 1) % np.ceil(niter / 10).astype(int) == 0:
                print('----- Training NB: batch %i of %i -----' % (ii + 1, niter))
            if (ii+1) == niter:
                ridx =  np.arange(mbatch*ii, self.n)
            else:
                ridx = np.arange(mbatch * ii, mbatch * (ii + 1))
            X_ii1 = self.enc.cenc.transform(data.iloc[ridx, self.enc.cidx])
            X_ii2 = self.enc.nenc.transform(data.iloc[ridx, self.enc.nidx])
            y_ii = lbls[ridx]
            s1_ii, s2_ii = X_ii1.sum(axis=0), X_ii2.sum(axis=0)
            self.di_moments['gram'][[0,1],[1,0],:] += np.concatenate((s1_ii,s2_ii))
            self.di_moments['gram'][1, 1, :X_ii1.shape[1]] += s1_ii # sum of 1^2 is still sum of 1
            self.di_moments['gram'][1, 1, X_ii1.shape[1]:] += np.sum(X_ii2**2,axis=0)
            self.di_moments['inner'][1,0,:] += np.concatenate((y_ii.T.dot(X_ii1).flatten(),
                                                               y_ii.T.dot(X_ii2).flatten()))
        del X_ii1, X_ii2
        # Calculate inverse-gram & coefficients
        self.Bhat = np.zeros([4, self.enc.p2])
        for kk in range(self.enc.p2):
            self.di_moments['igram'][:,:,kk] = np.linalg.inv(self.di_moments['gram'][:,:,kk])
            self.Bhat[0:2,kk] = self.di_moments['igram'][:, :, kk].dot(self.di_moments['inner'][:,:,kk]).flatten()
        # Calculate the standard error
        for ii in range(niter):
            if (ii+1) == niter:
                ridx =  np.arange(mbatch*ii, self.n)
            else:
                ridx = np.arange(mbatch * ii, mbatch * (ii + 1))
            X_ii = np.c_[self.enc.cenc.transform(data.iloc[ridx, self.enc.cidx]),
                         self.enc.nenc.transform(data.iloc[ridx, self.enc.nidx])]
            y_ii = lbls[ridx]
            self.Bhat[2,:] += np.sum((y_ii - ((X_ii * self.Bhat[[1],:]) + self.Bhat[[0],:]))**2,0)
        self.Bhat[2, :] /= (self.n - 2) # two degrees of freedom
        self.Bhat[3, :] = 1 - self.Bhat[2, :]/lbls.var() # r-squared
        self.Bhat[3, :] = np.where(self.Bhat[3, :] < 0, 0, self.Bhat[3, :])
        zscore = self.Bhat[1, :] / np.sqrt(self.di_moments['igram'][1, 1, :] * self.Bhat[2, :])
        pvals = 2*(1-norm.cdf(np.abs(zscore)))
        pvals[self.Bhat[3, :] == 0] = 1
        pvals[self.Bhat[3, :] > 0] = multitest.fdrcorrection(pvals[self.Bhat[3, :] > 0],alpha=0.1)[1]
        self.Bhat[3, :] = np.where(pvals < 0.1, self.Bhat[3, :], 0)
        print('%i of %i features remain' % (sum(self.Bhat[3, :] > 0), self.enc.p2))

    def predict(self, data, mbatch=None):
        stopifnot( data.shape[1] == self.p )
        # Check categoreis line up for predict
        new_vals = [list(np.setdiff1d(data.iloc[:, jj].unique(),uvals)) for
            jj, uvals in zip(self.enc.cidx,self.enc.cenc.categories_)]
        diff_vals = np.where(np.array([len(z) for z in new_vals])>0)[0]
        if len(diff_vals) > 0:
            data = data.copy() # protect columns from being overwritten
            print("new categorical values! Setting to default")
            for jj in diff_vals:
                cjj = self.enc.cidx[jj] # column in reference
                data.iloc[:, cjj] = np.where(data.iloc[:,cjj].isin(new_vals[jj]),
                            self.enc.cenc.categories_[jj][0],data.iloc[:, cjj])
        # Processing information
        n_pred = data.shape[0]
        if mbatch is None:
            mbatch = n_pred
        idx_splits = idx_iter(n_pred, mbatch)
        niter = len(idx_splits)
        return(self.fun_methods[self.method]['predict'](data,niter,idx_splits))

    def predict_bernoulli(self,data,niter,splits):
        pmat = pred_batch(data, niter, splits, self.lst_mdls, self.lst_cidx, self.lst_enc, self.lst_iter)
        return(self.stacker.predict_proba(pmat))

    def predict_gaussian(self,data,niter,splits):
        x_pred = x_batch(data,niter,splits,self.lst_cidx, self.lst_enc, self.lst_iter)
        b0 = np.average(a=self.Bhat[0, :], weights=self.Bhat[3, :])
        # Weight by r-squared, statistical significance, and whether the value is zero
        eta = np.average(a=x_pred * self.Bhat[[1], :] + b0, axis=1, weights=self.Bhat[3, :]*~(x_pred==0))
        return(eta)