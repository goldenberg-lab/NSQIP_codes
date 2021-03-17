import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd

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
