import numpy as np
from sklearn.preprocessing import StandardScaler as SS
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return(1/(1+np.exp(-z)))
def w_logit(z):
    pz = sigmoid(z)
    return(pz * (1-pz))

# False Positive Control (Logistic) Lasso
#self = FPC_logit(False)
class FPC():
    def __init__(self,standardize):
        self.standardize = standardize

    def fit(self,X,y,lam_fpc):
        n, self.p = X.shape
        if self.standardize:
            self.enc = SS().fit(X)
        else:
            self.enc = SS()
            self.enc.mean_ = np.repeat(0, self.p)
            self.enc.scale_ = np.repeat(1, self.p)
        Xtil = self.enc.transform(X)
        ybar = y.mean()
        lmax = max(np.abs(Xtil.T.dot(y - ybar) / n))
        lmin = lmax * 0.001
        lseq = np.exp(np.linspace(np.log(lmax),np.log(lmin),100))
        self.l1 = Lasso(fit_intercept=True, normalize=False,copy_X=False, warm_start=True)
        e2 = np.repeat(0.0, len(lseq))
        ll2 = e2.copy()
        for ii, ll in enumerate(lseq):
            self.l1.alpha = ll
            self.l1.fit(Xtil,y)
            r2 = np.sqrt(sum((y - self.l1.predict(Xtil))**2))
            e2[ii], ll2[ii] = r2, n * ll / r2
            if ll2[ii] < lam_fpc:
                print('Found solution!')
                self.supp = np.where(~(self.l1.coef_==0))[0]
                self.l1 = LogisticRegression('l2',C=1000,fit_intercept=True,solver='lbfgs',max_iter=1000)
                self.l1.fit(Xtil[:,self.supp],y)
                break
            # else:
            #     print('ii: %i, lam1: %0.3f, lam2: %0.3f' % (ii, ll, ll2[ii]))

    def predict(self,X):
        return(self.l1.predict_log_proba(self.enc.transform(X)[:,self.supp])[:,1])

