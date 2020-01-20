import sys
import numpy as np

def stopifnot(stmt):
    if not stmt:
        sys.exit('error! Statement is not True')


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

# from sklearn.naive_bayes import BernoulliNB
# xx1 = np.where( np.random.rand(100,3) > 0.5, 1, 0)
# yy1 = np.random.choice(['a','b','c'],100)
# xx2 = np.where( np.random.rand(100,4) > 0.5, 1, 0)
# yy2 = np.random.choice(['a','b','c'],100)
# nb = BernoulliNB().fit(X=xx1,y=yy1)
# nb.partial_fit(X=xx1,y=yy1,classes=np.unique(yy1))
# nb.partial_fit(X=xx2,y=yy2,classes=np.unique(yy1))
#
# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# xx, yy = load_iris(True)
#
# mlogit = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs',multi_class='multinomial')
# mlogit2 = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs',multi_class='multinomial')
# mlogit3 = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs',multi_class='multinomial')
# mlogit.fit(xx,yy)
# Eta = xx.dot(mlogit.coef_.T)
# Eta2 = Eta[:,1:]
# mlogit2.fit(Eta, yy)
# mlogit3.fit(Eta2, yy)
# rr = 5
# np.max(mlogit3.predict_proba(Eta2) - mlogit2.predict_proba(Eta))
# np.round(mlogit.predict_proba(xx),3)==np.round(mlogit2.predict_proba(Eta),3)

