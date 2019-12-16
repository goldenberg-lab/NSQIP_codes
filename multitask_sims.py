"""
SCRIPT TO ANALYZE MULTITASK APPROACHES
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns

from modelling_funs import multitask_logistic, sigmoid

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
    Y = np.where(sigmoid(latent) > np.random.rand(n,1), 1, 0)
    return(Y,X,latent,b0,eta)


n=1000
p=50
k=10
rho=0.75
nsim = 100
holder = []
for ii in range(nsim):
    if (ii+1) % 25 == 0:
        print('Iteration %i of %i' % (ii+1, nsim))
    Y, X, latent, b0, eta = yXmulti(n,p,k,rho,ii)
    Z = np.c_[X, np.random.randn(n,p)]
    # Fit model
    mdl = multitask_logistic(add_intercept=True,standardize=True)
    mdl.fit(X=Z,Y=Y,lam1=0,lam2=0.0)
    df = pd.DataFrame(np.concatenate((mdl.intercept,mdl.weights)).T,
                      columns=np.append('int',np.repeat(['true','false'], p)))
    holder.append(df)
    # g = sns.FacetGrid(data=pd.DataFrame({'y': Y[:, 1], 'eta': sigmoid(mdl.predict(Z))[:, 1]}), hue='y')
    # g.map(sns.distplot, 'eta'); g.add_legend()

df_sim = pd.concat(holder).reset_index(drop=True)
# df_sim['pval'] = 2*(1-stats.norm.cdf(np.abs(df_sim.z)))
g = sns.FacetGrid(data=df_sim.melt(),col='variable',hue='variable',sharey=False)
g.map(sns.distplot,'value')


