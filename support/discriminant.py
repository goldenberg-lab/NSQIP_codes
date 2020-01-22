"""
FUNCTION TO IMPLEMENT LDA AND QDA
https://arxiv.org/pdf/1906.02590.pdf
"""

import numpy as np
from support.support_funs import stopifnot
from support.mdl_funs import normalize
from sklearn.covariance import graphical_lasso
from scipy.special import softmax


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
            self.di_moments[yy]['Sigma'] = (self.di_moments[yy]['Sigma'] -
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
