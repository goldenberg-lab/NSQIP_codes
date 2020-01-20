import numpy as np
import pandas as pd
import sys

from scipy.special import softmax
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn import metrics
from sklearn.covariance import graphical_lasso
from statsmodels.stats import multitest
#from acc_funs import auc, pairwise_auc

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression

def stopifnot(stmt):
    if not stmt:
        sys.exit('error! Statement is not True')

"""
FUNCTION TO IMPLEMENT LDA AND QDA
https://arxiv.org/pdf/1906.02590.pdf
"""

def idx_iter(n ,mbatch):
    mbatch = min(n, mbatch)
    nfull = n // mbatch
    npart = (n % mbatch > 0) + 0
    idx_splits = np.split(np.arange(mbatch * nfull), nfull)
    if npart == 1:
        idx_splits = idx_splits + [np.arange(mbatch * nfull,n)]
    return (idx_splits)

#self=batch_qda(mbatch=25000,method='qda');nlam=100
#data=X_df.loc[train_idx,cn_complete];lbls=tmp_y[train_idx].values
#data=X_df.loc[test_idx,cn_complete];lbls=tmp_y[test_idx].values

# self=normalize();data=X_test_ii
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

#self = mbatch_NB();mbatch=10000;data=X_test_ii;lbls=y_train_ii
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
        # For each feature we need: sum(x), sum(x**2), sum(x,y)
        self.fun_methods[self.method]['fit'](data,lbls,mbatch)

    def fit_bernoulli(self,data,lbls,mbatch):
        self.ulbls = np.unique(lbls)
        lidx = [np.where(lbls == kk)[0] for kk in self.ulbls]
        midx = [z.min() for z in lidx]
        self.mdl_cat, self.mdl_num = BernoulliNB(), GaussianNB()
        self.mdl_cat.fit(X=self.enc.cenc.transform(data.iloc[midx, self.enc.cidx]),y=lbls[midx])
        self.mdl_num.fit(X=self.enc.nenc.transform(data.iloc[midx, self.enc.nidx]), y=lbls[midx])
        idx_splits = idx_iter(self.n , mbatch)
        niter = len(idx_splits)
        # fit Naive Bayes
        for ii in range(niter):
            if (ii + 1) % np.ceil(niter / 10).astype(int) == 0:
                print('----- Training NB: batch %i of %i -----' % (ii + 1, niter))
            ridx = np.setdiff1d(idx_splits[ii], midx) # no double counting for midx
            x_ii_cat = self.enc.cenc.transform(data.iloc[ridx, self.enc.cidx])
            x_ii_num = self.enc.nenc.transform(data.iloc[ridx, self.enc.nidx])
            y_ii = lbls[ridx]
            self.mdl_cat.partial_fit(x_ii_cat, y_ii, self.ulbls)
            self.mdl_num.partial_fit(x_ii_num, y_ii, self.ulbls)
        # stack naive bayes
        phat = np.zeros([self.n, 2*(len(self.ulbls)-1)])
        for ii in range(niter):
            ridx = idx_splits[ii]
            x_ii_cat = self.enc.cenc.transform(data.iloc[ridx, self.enc.cidx])
            x_ii_num = self.enc.nenc.transform(data.iloc[ridx, self.enc.nidx])
            phat[ridx,:] = np.c_[self.mdl_cat.predict_proba(x_ii_cat)[:,1:],
                                self.mdl_num.predict_proba(x_ii_num)[:,1:]]
        # Stack with logistic
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

    def predict(self, data):
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
        x_pred = np.c_[self.enc.cenc.transform(data.iloc[:, self.enc.cidx]),
                       self.enc.nenc.transform(data.iloc[:, self.enc.nidx])]
        return(self.fun_methods[self.method]['predict'](x_pred))

    def predict_bernoulli(self,x_pred):
        x_stack = np.c_[self.mdl_cat.predict_proba(x_pred[:, :-len(self.enc.nidx)])[:, 1:],
                        self.mdl_num.predict_proba(x_pred[:, -len(self.enc.nidx):])[:, 1:]]
        return(self.stacker.predict_proba(x_stack))

    def predict_gaussian(self,x_pred):
        b0 = np.average(a=self.Bhat[0, :], weights=self.Bhat[3, :])
        # Weight by r-squared, statistical significance, and whether the value is zero
        eta = np.average(a=x_pred * self.Bhat[[1], :] + b0, axis=1, weights=self.Bhat[3, :]*~(x_pred==0))
        return(eta)

#self = batch_ols(); data = data2.copy(); lbls = lbls2.copy()
class batch_ols():
    def __init__(self,mbatch=25000):
        self.mbatch = mbatch

    def fit(self,data,lbls):
        self.enc = normalize(copy=False)
        self.enc.fit(data) # train the feature encoder
        self.p = self.enc.p2
        self.n = data.shape[0]
        if len(lbls.shape)==1:
            lbls = lbls.reshape([self.n, 1])
        self.k = lbls.shape[1]
        self.mbatch = min(self.n, self.mbatch)
        niter = self.n // self.mbatch + (self.n % self.mbatch > 0)
        self.XX = np.zeros([self.p+1, self.p+1]) # +1 for intercept
        self.Xy, self.mu_ = np.zeros([self.p+1,self.k]), np.zeros(self.p+1)
        for ii in range(niter):
            if (ii+1) % max((niter // 10),1) == 0:
                print('iteration %i of %i' % (ii+1, niter))
            if (ii+1) == niter:
                ridx =  np.arange(self.mbatch*ii, self.n)
            else:
                ridx = np.arange(self.mbatch * ii, self.mbatch * (ii + 1))
            X_ii = np.c_[np.ones([ridx.shape[0],1]),
                            self.enc.cenc.transform(data.iloc[ridx,self.enc.cidx]),
                            self.enc.nenc.transform(data.iloc[ridx,self.enc.nidx])]
            self.mu_ += X_ii.sum(axis=0)
            y_ii = lbls[ridx]
            self.XX += X_ii.T.dot(X_ii)
            self.Xy += X_ii.T.dot(y_ii)
        self.XX /= self.n
        self.Xy /= self.n
        self.mu_ /= self.n
        self.scale_ = self.XX.diagonal() - self.mu_**2
        self.bhat = np.linalg.solve(self.XX, self.Xy)

    # comp = -1, implies higher is better
    def tune(self, data, lbls, nlam=100, metric_fun=metrics.r2_score, comp=-1):
        lam_seq = np.exp(np.linspace(np.log(0.01), np.log(np.abs(self.Xy).max() / 0.1), nlam))
        #lam_seq = np.append(0,lam_seq)
        x_tune = np.c_[np.ones([data.shape[0], 1]),
                    self.enc.cenc.transform(data.iloc[:, self.enc.cidx]),
                    self.enc.nenc.transform(data.iloc[:, self.enc.nidx])]
        stopifnot(x_tune.shape[1] == (self.p+1))
        self.metric = metric_fun(lbls,self.predict(data))
        self.lam_star = 0
        for jj in range(nlam):
            lam_jj = lam_seq[jj]
            bhat_jj = np.linalg.solve(self.XX + lam_jj * np.diag(self.scale_), self.Xy)
            eta_jj = x_tune.dot(bhat_jj)
            metric_jj = metric_fun(lbls, eta_jj) #print(jj);print(metric_jj)
            if int(np.sign(metric_jj - self.metric))==comp:
                break
            self.bhat = bhat_jj
            self.metric = metric_jj
            self.lam_star = lam_jj

    def predict(self, data):
        x_pred = np.c_[np.ones([data.shape[0], 1]),
                       self.enc.cenc.transform(data.iloc[:, self.enc.cidx]),
                       self.enc.nenc.transform(data.iloc[:, self.enc.nidx])]
        stopifnot(x_pred.shape[1] == (self.p + 1))
        return(x_pred.dot(self.bhat))

class batch_discriminant():
    def __init__(self,mbatch=25000, method='lda'):
        self.mbatch = mbatch
        self.method = method
        stopifnot( self.method in ['lda','qda'] )
        self.fun_methods = {'lda':{'fit':self.fit_lda, 'predict':self.predict_lda},
                            'qda':{'fit':self.fit_qda, 'predict':self.predict_qda}}

    def fit(self,data,lbls):
        self.enc = normalize(copy=False)
        self.enc.fit(data) # train the feature encoder
        n = data.shape[0]
        self.p = self.enc.p2
        self.uy = np.unique(lbls)
        self.k = len(self.uy)
        idx = [np.where(lbls == z)[0] for z in self.uy] # training indices by class label
        self.ny = dict(zip(self.uy,[len(z) for z in idx]))
        self.mbatch = min(max(self.ny.values()), self.mbatch)
        idx = [np.split(ii[0:(nn - nn%self.mbatch)],nn//self.mbatch) + [ii[(nn - nn % self.mbatch):]]
                            for nn, ii in zip(self.ny.values(), idx)]
        idx = dict(zip(self.uy, [[y[z] for z in np.where([len(x)>0 for x in y])[0]] for y in idx] ))
        self.fun_methods[self.method]['fit'](data,lbls,idx) # Fit with method

    def fit_qda(self,data,lbls,idx):
        self.di_moments = dict(zip(self.uy,[{'mu':np.repeat(0,self.p).astype(float),
              'Sigma':np.zeros([self.p,self.p]), 'iSigma':np.zeros([self.p,self.p])} for z in self.uy]))
        for yy in self.uy:
            self.di_moments[yy]['n'] = self.ny[yy]
            for ii in idx[yy]:
                x_ii = np.c_[self.enc.cenc.transform(data.iloc[ii,self.enc.cidx]),
                             self.enc.nenc.transform(data.iloc[ii,self.enc.nidx])]
                self.di_moments[yy]['mu'] += x_ii.sum(axis=0)
                self.di_moments[yy]['Sigma'] += x_ii.T.dot(x_ii)
            # Adjust raw numbers
            self.di_moments[yy]['mu'] = self.di_moments[yy]['mu'].reshape([self.p, 1]) / self.ny[yy]
            self.di_moments[yy]['Sigma'] = (self.di_moments[yy]['Sigma'] - \
                self.ny[yy]*self.di_moments[yy]['mu'].dot(self.di_moments[yy]['mu'].T)) / (self.ny[yy]-1)
            #self.di_moments[yy]['ldet'] = np.log(np.linalg.det(self.di_moments[yy]['Sigma']))
            #self.di_moments[yy]['iSigma'] = np.linalg.pinv(self.di_moments[yy]['Sigma'])
            self.di_moments[yy]['iSigma'] = graphical_lasso(emp_cov=self.di_moments[yy]['Sigma'], alpha=0.001)

    def fit_lda(self, data, lbls, idx):
        n = data.shape[0]
        self.di_moments = {'Mu':np.zeros([self.p,self.k]), 'mu':np.zeros([self.p, 1]),
                'Sigma':np.zeros([self.p,self.p]),'iSigma':np.zeros([self.p,self.p])}
        for jj, yy in enumerate(self.uy):
            for ii in idx[yy]:
                x_ii = np.c_[self.enc.cenc.transform(data.iloc[ii, self.enc.cidx]),
                             self.enc.nenc.transform(data.iloc[ii, self.enc.nidx])]
                s_ii = x_ii.sum(axis=0)
                self.di_moments['Mu'][:,jj] += s_ii
                self.di_moments['Sigma'] += x_ii.T.dot(x_ii)
            self.di_moments['Mu'][:, jj] /= self.ny[yy]
        self.di_moments['mu'] = ((self.di_moments['Mu'] *
                    np.array(list(self.ny.values())).reshape([1,self.k])).sum(axis=1) / n).reshape([self.p,1])
        self.di_moments['Sigma'] = (self.di_moments['Sigma'] -
                            n*self.di_moments['mu'].dot(self.di_moments['mu'].T)) / (n-1)
        self.di_moments['iSigma'] = np.linalg.pinv(self.di_moments['Sigma'])
        self.di_moments['const'] = np.sum(self.di_moments['iSigma'].dot(self.di_moments['Mu']) *
                            self.di_moments['Mu'],axis=0).reshape([1,self.k])

    def tune(self,data,lbls):
        x_tune = np.c_[self.enc.cenc.transform(data.iloc[:, self.enc.cidx]),
                     self.enc.nenc.transform(data.iloc[:, self.enc.nidx])]
        #graphical_lasso(emp_cov=mdl_qda.di_moments[0]['Sigma'], alpha=0.001)

    def predict(self,data):
        x_pred = np.c_[self.enc.cenc.transform(data.iloc[:, self.enc.cidx]),
                     self.enc.nenc.transform(data.iloc[:, self.enc.nidx])]
        return(self.fun_methods[self.method]['predict'](x_pred))

    def predict_lda(self,x):
        eta = x.dot(self.di_moments['iSigma'].dot(self.di_moments['Mu'])) - 0.5*self.di_moments['const']
        return(softmax(eta,axis=1))

    def predict_qda(self,x):
        eta = np.zeros([x.shape[0],self.k])
        for jj, yy in enumerate(self.uy):
            x_yy = x - self.di_moments[yy]['mu'].T
            eta[:,jj] = -0.5*np.sum(x_yy.dot(self.di_moments[yy]['iSigma']) * x_yy, axis=1) #-0.5*self.di_moments[yy]['ldet']
        return(softmax(eta,axis=1))

