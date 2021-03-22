import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from scipy.special import comb

def n_star_sens(sens, dsens, beta, alpha):
    """
    sens: the target sensitivity
    dens: the difference in the trial sensitivity
    beta/alpha: type-II and type-I error rate
    """
    assert (dsens > 0) & (dsens < sens) & (sens > 0) & (sens < 1)
    l = sens - dsens
    term1 = np.sqrt(sens*(1-sens))*stats.norm.ppf(beta)
    term2 = np.sqrt(l*(1-l))*stats.norm.ppf(1-alpha)
    term3 = l - sens 
    stat = ((term1 - term2)/term3)**2
    return stat

# n=100; k=0.05; j=10
# del n, k, j
def umbrella_thresh(n, k, j, ret_df=False):
    """
    1 - k: the sensitivity target
    1 - j: the generalization error (coverage target)
    """
    assert (k > 0) & (k <= 0.5)
    assert (j > 0) & (j <= 0.5)
    rank_seq = np.arange(n+1)
    rank_pdf = np.array([comb(n, l, True)*((1-k)**l)*((k)**(n-l)) for l in rank_seq])
    rank_cdf = np.array([rank_pdf[l:].sum() for l in rank_seq])
    res = pd.DataFrame({'rank':rank_seq, 'pdf':rank_pdf, 'cdf':rank_cdf, 'delt':1-rank_cdf})
    if ret_df:
        return res
    r_max = max(res[res.delt <= 1-j]['rank'].max(),1)
    return r_max

# Function to carry out bootstrap
# x=df_sk.preds.values; fun=np.sum; n_bs=1000; k=6
def bootstrap_x(x, fun, n_bs, k):
    assert len(x.shape) == 1
    n = len(x)
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    n_k = n_bs // k
    k_seq = np.append(np.repeat(n_k,k),[n_bs - n_k*k])
    k_seq = k_seq[k_seq != 0]
    i_seq = np.append([0],np.cumsum(k_seq))
    holder = np.repeat(np.NaN, n_bs)
    for i, k in enumerate(k_seq):
        res = np.apply_over_axes(fun, x.sample(frac=k,replace=True,random_state=i).values.reshape([n,k]), [0])
        holder[i_seq[i]:i_seq[i+1]] = res.flat
    return holder



def gen_CI(x,se,alpha):
    pvals = np.array([1-alpha/2, alpha/2])
    critvals = stats.norm.ppf(pvals)
    return pd.DataFrame({'lb':x - critvals[0]*se, 'ub':x - critvals[1]*se})

def auc2se(x,normal=False):
    assert isinstance(x,pd.DataFrame)
    assert x.columns.isin(['auc','n1','n0']).sum() == 3
    if normal:
        x = x.assign(se = lambda x: np.sqrt((x.n1+x.n0+1)/(12*x.n1*x.n0)))
    else:
        x = x.assign(q0 = lambda x: x.auc*(1-x.auc),
                    q1 = lambda x: x.auc/(2-x.auc) - x.auc**2,
                    q2 = lambda x: 2*x.auc**2/(1+x.auc) - x.auc**2,
                    n1n0=lambda x: x.n1*x.n0)
        x = x.assign(se=lambda x: np.sqrt( (x.q0 + (x.n1-1)*x.q1 + (x.n0-1)*x.q2 ) / x.n1n0 ) )
    return x.se.values
