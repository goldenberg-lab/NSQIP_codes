
"""
SCRIPT TO ANALYZE MULTITASK APPROACHES
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from support.support_funs import rand_split, gridsearch

def sigmoid(x):
    return(1 / (1 + np.exp(-x)))

def yXmulti(n,p,k,rho,ss):
    np.random.seed(ss)
    X = np.random.randn(n,p)
    b0 = np.where(np.random.rand(p) < 0.5,1,-1)
    eta = X.dot(b0).reshape([n,1])
    sig2_eta = sum(b0**2)
    sig2_noise = sig2_eta*(1/rho**2 - 1)
    noise = np.random.randn(n,k) * np.sqrt(sig2_noise)
    latent = noise + eta
    Y = np.where(sigmoid(latent) > np.random.rand(n,k), 1, 0)
    return(Y,X,latent,b0,eta)

from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

n=200
p=10
k=10
rho = 0.75
nsim = 250
df_store = pd.DataFrame(np.zeros([nsim,2]),columns=['mtl','indep'])
for ii in range(nsim):
    if (ii+1) % 1 == 0:
        print('Iteration %i of %i' % (ii+1, nsim))
    Y, X, latent, b0, eta = yXmulti(2*n,p,k,rho,ii)
    X = np.c_[X, np.random.randn(2*n,100)]

    Ytrain, Ytest = Y[:n], Y[n:]
    Xtrain, Xtest = X[:n], X[n:]
    enc = StandardScaler(copy=False).fit(Xtrain)
    Xtrain_norm = enc.transform(Xtrain)
    Xtest_norm = enc.transform(Xtest)

    # --- (i) Do multitask training --- #
    # see: https://www.cs.cmu.edu/~ggordon/10725-F12/slides/16-kkt.pdf [no sqrt(k) though]
    lammax = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)),1,Xtrain.T.dot(Ytrain) / n).max()
    mtl = MultiTaskElasticNet(l1_ratio=1,copy_X=False)
    lamseq = np.exp(np.linspace(np.log(1e-3),np.log(lammax*0.99),50))
    gs_mtl = gridsearch(param_dict={'alpha': lamseq}, estimator=mtl,
                    metric=roc_auc_score, splitter=rand_split)
    gs_mtl.fit(X=Xtrain, y=Ytrain, ncv=3, ss=1234)
    auc_mtl = roc_auc_score(Ytest, gs_mtl.estimator.predict(Xtest))

    # --- (ii) Do columnwise training --- #
    auc_indep = np.repeat(np.NaN, k)
    for jj in range(k):
        ytrain_jj, ytest_jj = Ytrain[:, jj], Ytest[:, jj]
        lam_jj = np.max(np.abs(Xtrain.T.dot(ytrain_jj)/n))
        elnet = ElasticNet(l1_ratio=1,copy_X=False)
        lamseq_jj = np.exp(np.linspace(np.log(1e-3),np.log(lam_jj*0.99),50))
        gs = gridsearch(param_dict={'alpha':lamseq_jj},estimator=elnet,
                        metric=roc_auc_score,splitter=rand_split)
        gs.fit(X=Xtrain, y=ytrain_jj, ncv=3, ss=1234)
        auc_indep[jj] = roc_auc_score(ytest_jj, gs.estimator.predict(Xtest))
    auc_indep = auc_indep.mean()
    df_store.loc[ii] = [auc_mtl, auc_indep]


# g = sns.scatterplot(x='indep',y='mtl',data=df_store)
# mi = np.round(df_store.min().min(),2)-0.02
# mx = np.round(df_store.max().max(),2)+0.02
# g.set_xlim([mi,mx])
# g.set_ylim([mi,mx])
# plt.plot([mi, mx], [mi, mx],color='red')



