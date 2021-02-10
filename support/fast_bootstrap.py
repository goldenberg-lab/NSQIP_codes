import numpy as np
import pandas as pd
from scipy import stats

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rvec(x):
    return np.atleast_2d(x)

def cvec(x):
    return rvec(x).T

def to_3d(mat):
    return np.atleast_3d(mat).transpose(2,0,1)

def srho(x,y):
    return stats.spearmanr(x,y)[0]

"""
VECTORIZED PAIRWISE CORRELATION
A : (n x p x s)
n: sample size (rows of the data)
p: columns of the data (could be bootstrapped columns)
s: copies of the n x p matrix (could be studentized copies)
"""

def pairwise_cor(A, B):
    assert A.shape == B.shape
    n = A.shape[0]
    if (len(A.shape) == 2):
        mu_A, mu_B = rvec(A.mean(0)), rvec(B.mean(0))
        se_A, se_B = A.std(axis=0,ddof=1), B.std(axis=0,ddof=1)
    else:
        mu_A, mu_B = to_3d(A.mean(0)), to_3d(B.mean(axis=0))
        se_A, se_B = A.std(axis=0,ddof=1), B.std(axis=0,ddof=1)
    D = np.sum((A - mu_A) * (B - mu_B),0) / (n-1)
    return D / (se_A*se_B)

def bs_student_spearman(x, y, n_bs, n_s, alpha=0.05):
    # alpha = rvec([0.05, 0.1, 0.2])
    tt = ['student','normal','quant']
    if isinstance(alpha, float) | isinstance(alpha,list):
        alpha = np.array([alpha])
    alpha = rvec(alpha)
    assert len(x) == len(y)
    assert np.all(alpha > 0) & np.all(alpha < 0.5)
    # (i) Get baseline statistic
    rho = stats.spearmanr(x, y)[0]
    n = len(x)
    pvals = np.r_[alpha/2,1-alpha/2].T
    # (ii) Transform data into ranks and sample with replacement
    x_r, y_r = stats.rankdata(x), stats.rankdata(y)
    x_bs = pd.Series(x_r).sample(frac=n_bs,replace=True)
    y_bs = pd.Series(y_r).iloc[x_bs.index]
    x_bs = x_bs.values.reshape([n,n_bs])
    y_bs = y_bs.values.reshape([n,n_bs])
    rho_bs = pairwise_cor(x_bs, y_bs)
    se_bs = rho_bs.std(ddof=1)
    # (iii) Bootstrap the bootstraps (studentize) to get standard error
    x_s = pd.DataFrame(x_bs).sample(frac=n_s,replace=True)
    y_s = pd.DataFrame(y_bs).iloc[x_s.index]
    x_s = x_s.values.reshape([n_s,n,n_bs]).transpose(1,2,0)
    y_s = y_s.values.reshape([n_s,n,n_bs]).transpose(1,2,0)
    se_s = pairwise_cor(x_s, y_s).std(axis=1,ddof=1)
    del x_s, y_s
    # Get the confidence intervals for the different approaches
    z_q = np.quantile(rho_bs,pvals.flat).reshape(pvals.shape)
    z_n = stats.norm.ppf(pvals)
    t_s = (rho_bs-rho)/se_s
    z_s = np.quantile(t_s,pvals.flat).reshape(pvals.shape)
    df = pd.DataFrame(np.r_[rho - se_bs*z_s[:,[1,0]], rho - se_bs*z_n[:,[1,0]], z_q],columns=['lb','ub'])
    df.insert(0,'rho',rho)
    df = df.assign(tt=np.repeat(tt,len(pvals)),alpha=np.tile(2*pvals[:,0],len(tt)))
    return df
