import numpy as np
import pandas as pd
import matplotlib, os
matplotlib.use('Agg') # no print-outs
from matplotlib import pyplot as plt
import seaborn as sns
dir_base = os.getcwd()
dir_figures = os.path.join(dir_base,'..','figures')

###########################################################

from scipy.stats import norm

def sens(t, mu1, se1):
    return 1 - norm.cdf((t - mu1) / se1)

def isens(sens, mu1, se1):
    return se1 * norm.ppf(1-sens) + mu1

def fpr(t, mu0, se0):
    return 1 - norm.cdf((t - mu0) / se0)

def calc_sens(x, y, t):
    x1 = x[y == 1]
    return np.mean(x1 > t)

def calc_fpr(x, y, t):
    x0 = x[y == 0]
    return np.mean(x0 > t)

def calc_ppv(x, y, t):
    ymu = np.mean(y)
    tp, fp = calc_sens(x, y, t), calc_fpr(x, y, t)
    ppv = (tp * ymu) / ((tp * ymu) + (fp*(1-ymu)))
    return ppv

def dgp_yeta(n, mu1, prop=0.5, se1=1, mu0=0, se0=1):
    n = int(n)
    n1 = int(n * prop)
    n0 = n - n1
    x0 = se0 * np.random.randn(n0) + mu0
    x1 = se1 * np.random.randn(n1) + mu1
    y = np.append(np.repeat(0, n0), np.repeat(1, n1))
    x = np.append(x0, x1)
    return x, y

###########################################################

# Empirical terms
mu0, mu1 = -3.73, -2.99
se0, se1 = 1.00, 0.93

# For target of 60% sensitivity:
sens_s60 = 0.6
t_s60 = isens(sens_s60, mu1, se1)
fpr_s60 = fpr(t_s60, mu0, se0)
ybar = 40 / 440 #0.03034
ppv_s60 = (sens_s60 * ybar) / ( (sens_s60 * ybar) + (fpr_s60 * (1-ybar)) )

print('Sensitivity: %0.3f, FPR: %0.3f, and PPV: %0.3f' %
      (sens_s60, fpr_s60, ppv_s60))

n_tot = 440
nsim = 10000
mat = np.zeros([nsim,2])
for ii in range(nsim):
    if (ii+1) % 1000 == 0:
        print('Simulation %i of %i' % (ii+1, nsim))
    x, y = dgp_yeta(n=n_tot, mu1=mu1, prop=ybar, se1=se1, mu0=mu0, se0=se0)
    ii_sens, ii_ppv = calc_sens(x, y, t_s60), calc_ppv(x, y, t_s60)
    mat[ii] = ii_sens, ii_ppv

df_sim = pd.DataFrame(mat,columns=['Sensitivity','PPV']).melt(None,None,'metric','value')

def hist_with_95CI(*args,**kwargs):
    data = kwargs.pop('data')
    lb, ub = data.value.quantile(0.025), data.value.quantile(0.975)
    mi, mx = data.value.min(), data.value.max()
    sns.distplot(data.value,bins=np.linspace(mi, mx+1e-5,num=22),**kwargs)
    plt.axvline(x=lb, c='black', linestyle='--')
    plt.axvline(x=ub, c='black', linestyle='--')
    plt.text(x=lb*0.65, y=1/lb, s=np.round(lb, 3))
    plt.text(x=ub * 1.03, y=1 / lb, s=np.round(ub, 3))

plt.close()
g = sns.FacetGrid(df_sim,col='metric',sharex=False,sharey=False)
g.map_dataframe(hist_with_95CI,'value')
g.savefig(os.path.join(dir_figures,'SSI_sim_440.png'))





