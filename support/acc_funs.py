import numpy as np
import scipy as sp
import pandas as pd
import sys

import matplotlib
# matplotlib.use('Agg')
import seaborn as sns

def stopifnot(stmt):
    if not stmt:
        sys.exit('error! Statement is not True')

# X = X_df[cX].copy();
# tt = tt_X.copy()
def process_X_dicts(X, tt):
    import sklearn.preprocessing as pp
    n, p = X.shape
    if not (p == len(tt)):
        sys.exit('Error! Lengths do not line up')
    holder = []
    for jj in range(p):
        x_jj = X.iloc[:, jj].values.reshape([n, 1])
        if (tt[jj] == 'int64') | (tt[jj] == 'float64'):
            holder.append(pp.StandardScaler().fit(x_jj))
        elif tt[jj] == 'object':
            holder.append(pp.OneHotEncoder(drop='first').fit(x_jj))
        else:
            sys.exit('error! Data type does not conform to expectations')
    return (holder)


def sens_spec_fun(thresh, y, score, method):
    y_pred = np.where(score > thresh, 1, 0)
    if method == 'sens':
        y0 = 1
    elif method == 'spec':
        y0 = 0
    else:
        print('Error! Please choose a method=["sens","spec"]')
    metric = np.mean(y_pred[y == y0] == y0)
    return (metric)


def sens_spec_fun2(thresh, y, score, method, target):
    return ((sens_spec_fun(thresh, y, score, method) - target) ** 2)


# y=y_mdl; score=p_mdl;method='spec';target=0.98
def thresh_finder(target, y, score, method):
    thresh = sp.optimize.minimize_scalar(fun=sens_spec_fun2, bounds=(0, 1),
                                         method='bounded', args=(y, score, method, target)).x
    return (thresh)

# General AUC function
def auc(y, score, rand=False, nsamp=10000, both=False):
    if not all((y == 0) | (y == 1)):
        print('error, y has non-0/1'); return(None)
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    den = len(idx1) * len(idx0)
    if (len(idx1) == 0) | (len(idx0) == 0):
        if both:
            return(np.nan, 0)
        else:
            return (np.nan)
    if not rand:
        num = 0
        score0 = score[idx0]
        for ii in idx1:
            num += np.sum(score[ii] > score0)
            num += np.sum(score[ii] == score0)/2
        ratio = num / den
        if both:
            return(ratio, den)
        else:
            return (ratio)
    else:
        auc_rand = np.mean(score[np.random.choice(idx1, nsamp)] > score[np.random.choice(idx0, nsamp)])
        if both:
            return(auc_rand, den)
        else:
            return(auc_rand)

def auc_ordinal(y,score):
  score = score[~np.isnan(y)]
  y = y[~np.isnan(y)]
  uy = np.sort(np.unique(y))
  tmp = []
  for yy in uy:
      tmp.append(score[y == yy])
  num, den = 0, 0
  for ii in np.arange(1,len(uy)):
      score_other = np.concatenate(tmp[:ii])
      score_ii = tmp[ii]
      score_broad = score_ii.repeat(len(score_other)).reshape(len(score_ii),len(score_other)).T
      num += np.sum(score_broad > score_other.reshape([len(score_other), 1]))
      num += np.sum(score_broad == score_other.reshape([len(score_other), 1]))/2
      den += np.cumprod(list(score_broad.shape))[1]
  return(num / den)


# Decomposition function
# y = np.array([0,1,0,1,1,1])
# score = np.array([0,1,1.5,3,2,10])
# group = np.array(['a','a','b','b','b','c'])
#print(pd.DataFrame({'y':y,'score':score,'group':group}))
#y=x.y.values;score=x.phat.values;group=x.cpt.values
def auc_decomp(y,score,group,rand=False):
    stopifnot(len(y)==len(score)==len(group))
    auc_tot, n_tot = auc(y,score, both=True) # will also check for errors
    ugroup = np.unique(group)
    ugroup1 = ugroup[np.where([sum(y[group==gg]==1)>=1 for gg in ugroup])[0]]
    #group0 = ugroup[np.where([sum(y[group == gg] == 0) >= 1 for gg in ugroup])[0]]
    gwise_auc = [auc(y[group==gg],score[group == gg],both=True,rand=rand) for gg in ugroup]
    df_gwise = pd.DataFrame(np.array(gwise_auc),columns=['auc','den'])
    df_gwise.den = df_gwise.den.astype(int)
    df_gwise.insert(0,'group',ugroup)
    n_within = df_gwise.den.sum()
    n_between = n_tot - n_within
    stopifnot(n_between>=0)
    df_gwise = df_gwise.assign(num=lambda x: (x.den * x.auc)) #.astype(int)
    auc_within = df_gwise.num.sum() / n_within
    holder = []
    for gg in ugroup1:
        idx_ingroup = (group == gg) & (y==1)
        idx_outgroup = ~(group == gg) & (y == 0)
        holder.append(auc(np.append(y[idx_ingroup],y[idx_outgroup]),
                np.append(score[idx_ingroup], score[idx_outgroup],rand=rand),both=True))
    df_between = pd.DataFrame(np.array(holder),columns=['auc','den'])
    stopifnot(df_between.den.sum() == n_between)
    df_between = df_between.assign(num=lambda x: (x.den * x.auc)) #.astype(int)
    auc_between = df_between.num.sum() / n_between
    auc_weighted = (auc_between * n_between / n_tot) + (auc_within * n_within / n_tot)
    stopifnot(np.round(auc_tot - auc_weighted,5)==0) # Ensure high precision
    df_ret = pd.DataFrame({'tt':['total','within','between'],
                 'auc':[auc_tot, auc_within, auc_between],
                  'den':[n_tot, n_within, n_between]})
    return(df_ret)

# The columns of score need to match unique(y)
def pairwise_auc(y,score,average=True):
    if len(y.shape) > 1:
        y = y.argmax(axis=1)
    uy = np.unique(y)
    tmp = []
    for i1 in np.arange(0,len(uy)-1):
        for i2 in np.arange(i1+1,len(uy)):
            l0, l1 = uy[i1], uy[i2]
            si1 = score[((y == l1) | (y == l0)),l1]
            yi1 = y[((y == l1) | (y == l0))]
            yi1 = np.where(yi1 == l1, 1, 0)
            tmp.append(pd.Series({'y1': l1, 'y0': l0, 'auc': auc(y=yi1, score=si1)}))
    df = pd.concat(tmp,axis=1).T
    if average:
        return(df.auc.mean())
    else:
        return (df)


# Plotting function for AUC
def plot_auc(lbl,score,num=100,figure=True):
    if not all((lbl == 0) | (lbl == 1)):
        print('error, lbl has non-0/1'); return (None)
    sidx = np.argsort(score)
    lbl, score = lbl[sidx], score[sidx]
    s1 = score[lbl == 1]
    uscore = np.linspace(start=s1.min()*0.99, stop=s1.max()*1.01, num=num)
    tmp = np.ones([uscore.shape[0],2]) * np.NaN
    for ii, tt in enumerate(uscore):
        tpr = sens_spec_fun(thresh=tt,y=lbl,score=score,method='sens')
        fpr = 1 - sens_spec_fun(thresh=tt,y=lbl,score=score,method='spec')
        tmp[ii,:] = [tpr, fpr]
    df = pd.DataFrame(tmp,columns=['tpr','fpr'])
    df.insert(0,'thresh',uscore)
    if figure:
        auc_tot = auc(y=lbl,score=score)
        fig = sns.lineplot(x='fpr', y='tpr', data=df)
        fig.set_ylabel('TPR'); fig.set_xlabel('FPR')
        fig.text(0.01, 0.98, 'AUC: %0.3f' % auc_tot)
        return((fig, df))
    else:
        return(df)

# Function to calculate positive predictive value ~
def ppv(thresh, y, score):
    yhat = np.where(score >= thresh, 1, 0)  # predicted label
    ntp = np.sum(yhat[y == 1])  # number of true positive
    nfp = np.sum(yhat[y == 0] == 1)  # number of false positive
    ppv = ntp / (ntp + nfp)
    return (ppv)

def plot_ppv(lbl,score,figure=True,num=int(1e3)):
    if not all((lbl == 0) | (lbl == 1)):
        print('error, lbl has non-0/1'); return (None)
    sidx = np.argsort(score)
    lbl, score = lbl[sidx], score[sidx]
    s1 = score[lbl == 1]
    uscore = np.linspace(start=s1.min()*0.99, stop=s1.max()*0.99, num=num)
    tmp = np.ones([uscore.shape[0],2]) * np.NaN
    for ii, tt in enumerate(uscore):
        tpr = sens_spec_fun(thresh=tt,y=lbl,score=score,method='sens')
        precision = ppv(thresh=tt,y=lbl,score=score)
        tmp[ii,:] = [tpr, precision]
    df = pd.DataFrame(tmp,columns=['tpr','precision'])
    df.insert(0,'thresh',uscore)
    if figure:
        fig = sns.lineplot(x='tpr', y='precision', data=df)
        fig.set_ylabel('PPV'); fig.set_xlabel('TPR')
        return((fig, df))
    else:
        return(df)


# Function to find threshold closest to target ppv
def thresh_find_ppv(target, y, score, ttype):
    s1 = score[y == 1]  # scores for positive class
    t_seq = np.linspace(s1.min(), s1.max(), num=100)
    t_seq = np.sort(np.concatenate((t_seq, s1)))
    ppv_seq = np.array([ppv(x, y, score) for x in t_seq])
    tmp = (ppv_seq - target) ** 2
    idx_min = np.where(tmp == tmp.min())[0]
    if ttype == 'inf':
        idx_star = idx_min.min()
    elif ttype == 'sup':
        idx_star = idx_min.max()
    else:
        print('error!');
        return (np.nan)
    t_star = t_seq[idx_star]
    return (t_star)

