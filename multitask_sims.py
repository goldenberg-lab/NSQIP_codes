"""
SCRIPT TO ANALYZE MULTITASK APPROACHES
"""

import os
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy import stats
import seaborn as sns

# Support functions
def sigmoid(x):
    return(1 / (1 + np.exp(-x)))
def corrfun(x,y):
    return(np.corrcoef(x,y)[0,1])

# n: sample size
# p: number of features
# k: number of tasks
# rho: correlation between latent task space
# ss: seeding
def yXmulti(n,p,k,rho,ss):
    np.random.seed(ss)
    X = np.random.randn(n,p)
    b0 = np.where(np.random.rand(p) < 0.5,1,-1)
    eta = X.dot(b0).reshape([n,1])
    sig2_eta = sum(b0**2)
    sig2_noise = sig2_eta*(1/rho**2 - 1)
    noise = np.random.randn(n,k) * np.sqrt(sig2_noise)
    latent = noise + eta
    Y = np.where(sigmoid(latent) < np.random.rand(n,1), 1, 0)
    return(Y,X,latent,b0,eta)

def loss_l2(b,y,X,lam,w=None):
    if w is None:
        w = np.repeat(1,len(y))
    eta = X.dot(b)
    l2 = np.sum(b**2)
    return(np.mean(w * (y - eta) ** 2) + lam*l2 / 2)

def grad_l2(b,y,X,lam,w=None):
    res = y - X.dot(b)
    return(-X.T.dot(res)/X.shape[0] + lam*b)

def loss_logit(y,X,b):
    eta = X.dot(b)
    return(-1*np.mean( y*eta - np.log(1+np.exp(eta)) ))

def grad_logit(y,X,b):
    py = sigmoid(X.dot(b))
    return(-X.T.dot(y - py)/X.shape[0] + b)

def OLS(y,x,add_intercept=True):
    if add_intercept:
        x = np.c_[np.ones([x.shape[0],1]),x]
    lhs = np.linalg.inv(x.T.dot(x))
    rhs = x.T.dot(y)
    bhat = lhs.dot(rhs)
    res = y - x.dot(bhat)
    se = np.std(res) * np.sqrt(np.diag(lhs))
    z = bhat / se
    return(bhat, se, z)

# x=Z.copy();y=latent[:,0];standardize=True; add_intercept=True;lam=0
def least_squares(y, x, lam=0, w=None, standardize=True, add_intercept=True):
    p = x.shape[1]
    if standardize:
        mu_x = x.mean(axis=0).reshape([1, p])
        sd_x = x.std(axis=0).reshape([1, p])
        x = (x - mu_x) / sd_x
    else:
        mu_x = np.repeat(0, p).reshape([1, p])
        sd_x = mu_x + 1
    if add_intercept:
        x = np.insert(arr=x, obj=0, values=1, axis=1)
    # Run optimize
    w = minimize(fun=loss_l2, x0=np.append(np.mean(y), np.repeat(0, p)),
                   args=(y, x, lam), jac=grad_l2).x
    # re-normalize
    w[1:] = w[1:] / sd_x.flatten()
    w[0] = w[0] - sum(w[1:]*mu_x.flatten())
    return(w)


# # Fast logistic regression fit
# def fast_logit(y,X):


n=50
p=5
k=10
rho=0.75
nsim = 100
holder = []
for ii in range(nsim):
    if (ii+1) % 25 == 0:
        print('Iteration %i of %i' % (ii+1, nsim))
    Y, X, latent, b0, eta = yXmulti(n,p,k,rho,ii)
    Z = np.c_[X, np.random.randn(n,p)]
    # df = pd.DataFrame({'tt':np.append('int',np.repeat(['true','false'], p)),
    #               'z':OLS(y=latent[:,0],x=Z)[2]})
    # holder.append(df)

# df_sim = pd.concat(holder).reset_index(drop=True)
# df_sim['pval'] = 2*(1-stats.norm.cdf(np.abs(df_sim.z)))
#
# g = sns.FacetGrid(data=df_sim,col='tt',hue='tt',sharey=False)
# g.map(sns.distplot,'pval')
#
# df_sim.head()


