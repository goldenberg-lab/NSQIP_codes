import numpy as np
from sklearn import metrics
from support.mdl_funs import normalize
from support.support_funs import stopifnot

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