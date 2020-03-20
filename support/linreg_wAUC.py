import numpy as np
import pandas as pd
from support.mdl_funs import col_encoder
from scipy.optimize import minimize
from sklearn.naive_bayes import BernoulliNB
from support.support_funs import strat_split
from support.mdl_funs import stopifnot

def sigmoid(x):
  return( 1 / (1 + np.exp(-x)) )

def idx_I0I1(y):
  return( (np.where(y == 0)[0], np.where(y == 1)[0] ) )

# y=ytest.copy();group=cpt_test.copy()
def idx_wb(y,group):
    tab = pd.crosstab(index=group,columns=y)
    ugroup0 = list(tab[tab.iloc[:,0]>0].index)
    ugroup1 = list(tab[tab.iloc[:,1]>0].index)
    ugroup10 = list(tab[tab.apply(lambda x: all(x > 0), 1)].index)
    # within-group indexing
    di_idx_w = dict(zip(ugroup10,[np.where(group == gg)[0] for gg in ugroup10]))
    for gg in di_idx_w:
        i0, i1 = idx_I0I1(y[di_idx_w[gg]])
        di_idx_w[gg] = dict(zip(['idx0','idx1'],[di_idx_w[gg][i0],di_idx_w[gg][i1]]))
    # between-group indexing
    di_idx_b1 = dict(zip(ugroup1, [np.where((group == gg) & (y == 1) )[0] for gg in ugroup1]))
    di_idx_b0 = dict(zip(ugroup0, [np.where((group == gg) & (y == 0))[0] for gg in ugroup0]))
    di_idx_b = {'idx1':di_idx_b1, 'idx0':di_idx_b0}
    return((di_idx_w, di_idx_b))

# Calculate non-convex version
# w=x0.copy();X=xx.copy();di_idx=idx_all.copy();eta=None
def wAUC(w,X,di_idx,offset=None,eta=None, den2=int(1e4)):
    if eta is None:
        eta = X.dot(w) + offset
    num, den = 0, 0
    is_between = all([z in di_idx for z in ['idx1','idx0']])
    if is_between:  # between
        for g1 in di_idx['idx1']:
            idx1 = di_idx['idx1'][g1]
            g0 = np.setdiff1d(list(di_idx['idx0'].keys()), g1)
            idx0 = np.concatenate([di_idx['idx0'][gg] for gg in g0])
            n, d = AUC_eta(eta[idx1], eta[idx0], den2)
            num += n; den += d
    else:  # within
        for gg in di_idx:
            idx0, idx1 = di_idx[gg]['idx0'],di_idx[gg]['idx1']
            n, d = AUC_eta(eta[idx1], eta[idx0], den2)
            num += n; den += d
    auc = num / den
    return auc

def AUC_eta(eta1, eta0,den2=int(1e4)):
    num = 0
    den = len(eta1) * len(eta0)
    if den <= den2:
        #print('exact')
        for e1 in eta1:
            num += sum( e1 > eta0 )
            num += sum( e1 == eta0 ) / 2
    else:
        #print('stochastic')
        np.random.seed(den)
        eta1b, eta0b = np.random.choice(eta1, den2),\
                       np.random.choice(eta0,den2)
        num = sum(eta1b > eta0b) + 0.5*sum(eta1b == eta0b)
        den = den2
    return (num, den)

def cAUC_eta(eta1, eta0, den2=int(1e4)):
    num = 0
    den = len(eta1) * len(eta0)
    if den <= den2:
        #print('exact')
        for e1 in eta1:
            num += sum( np.log(sigmoid(e1 - eta0)) )
    else:
        #print('stochastic')
        np.random.seed(den)
        eta1b, eta0b = np.random.choice(eta1, den2),\
                       np.random.choice(eta0,den2)
        num = sum(np.log(sigmoid(eta1b - eta0b)))
        den = den2
    return (-num, den)

def dcAUC_eta(idx1, idx0, eta, X, den2=int(1e4)):
    n1, n0 = len(idx1), len(idx0)
    den = n1 * n0
    grad = np.repeat(0.0, X.shape[1])
    if den <= den2:
        #print('exact')
        eta0, X0 = eta[idx0], X[idx0]
        for ii in idx1:
            grad += ((1 - sigmoid(eta[ii] - eta0)).reshape([n0, 1]) *
                    (X[[ii]] - X0)).sum(axis=0)
    else:
        #print('stochastic')
        np.random.seed(den)
        idx1b, idx0b = np.random.choice(idx1, den2), \
                       np.random.choice(idx0, den2)
        grad += ((1 - sigmoid(eta[idx1b] - eta[idx0b])).reshape([den2,1]) *
                    (X[idx1b] - X[idx0b])).sum(axis=0)
        den = den2
    return grad, den

# w=x0.copy(); X=xx.copy();di_idx=idx_w;lam=0
def cAUC(w,X,di_idx,offset,lam=0, den2=int(1e4)):
    eta = X.dot(w) + offset
    num, den = 0, 0
    is_between = all([z in di_idx for z in ['idx1','idx0']])
    if is_between: # between
        for g1 in di_idx['idx1']:
            idx1 = di_idx['idx1'][g1]
            g0 = np.setdiff1d(list(di_idx['idx0'].keys()), g1)
            idx0 = np.concatenate([di_idx['idx0'][gg] for gg in g0])
            n, d = cAUC_eta(eta[idx1], eta[idx0], den2)
            num += n; den += d
    else: # within
        for gg in di_idx:
            idx0, idx1 = di_idx[gg]['idx0'],di_idx[gg]['idx1']
            n, d = cAUC_eta(eta[idx1], eta[idx0], den2)
            num += n; den += d
    nll = (num + 0.5*lam*sum(w**2)) / den
    return nll

def dcAUC(w, X, di_idx, offset, lam=0, den2=int(1e4)):
    eta = X.dot(w) + offset
    grad = np.repeat(0.0,len(w))
    den = 0
    is_between = all([z in di_idx for z in ['idx1','idx0']])
    if is_between:
        for rr, g1 in enumerate(di_idx['idx1']):
            print('Group: %s (%i of %i)' % (g1,rr+1,len(di_idx['idx1'])))
            idx1 = di_idx['idx1'][g1]
            g0 = np.setdiff1d(list(di_idx['idx0'].keys()), g1)
            idx0 = np.concatenate([di_idx['idx0'][gg] for gg in g0])
            g, d = dcAUC_eta(idx1, idx0, eta, X, den2)
            grad += g
            den += d
    else:
        for gg in di_idx:
            idx0, idx1 = di_idx[gg]['idx0'], di_idx[gg]['idx1']
            g, d = dcAUC_eta(idx1, idx0, eta, X, den2)
            grad += g
            den += d
    grad = (-grad + lam*w) / den
    return grad

# self=linreg_wAUC();data=Xtrain.copy();lbls=ytrain.copy();fctr=cpt_train.copy()
# val=0.3;ss=1; tt=None; lam_seq=None;nlam=10
class linreg_wAUC():
    def __init__(self,standardize=True):
        self.standarize = standardize

    def transform_fctr(self,ff):
        if not isinstance(ff, pd.DataFrame):
            ff = pd.DataFrame(ff.copy())
        return(self.enc_fctr.transform(ff))

    def predict_fctr(self,fctr):
        phat = self.mdl_fctr.predict_proba(self.transform_fctr(fctr))[:,1]
        logodds = np.log(phat/(1-phat))
        return(logodds)

    def predict(self,data,fctr):
        xx = self.enc_X.transform(data)
        offset = self.predict_fctr(fctr)
        eta = {t: 'eta' for t in self.mdl}
        for tt in self.mdl:
            if tt == 'within':
                eta[tt] = xx.dot(self.mdl[tt]['bhat'])
            else:
                eta[tt] = xx.dot(self.mdl[tt]['bhat']) + offset
        return(eta)

    def AUC(self,eta,fctr,lbls,tt=None):
        if tt is None:
            tt = ['total', 'within', 'between']
        idx_w, idx_b = idx_wb(lbls,fctr)
        idx_all, _ = idx_wb(lbls, np.repeat(1,len(lbls)))
        di_idx = dict(zip(['total','within','between'],(idx_all,idx_w,idx_b)))
        di_idx = {z: di_idx[z] for z in tt if z in di_idx}
        auc = dict(zip(di_idx,
            [wAUC(eta=eta,di_idx=di_idx[z],X=None,offset=None,w=None) for z in di_idx]))
        return auc

    def fit(self,data,lbls,fctr,tt=None, nlam=50, den2=int(1e5),
                    lam_seq=None,val=None,ss=1234):
        if not isinstance(data,pd.DataFrame):
            data = pd.DataFrame(data)
        if not isinstance(fctr,pd.Series):
            fctr = pd.Series(fctr)
        if not isinstance(lbls,np.ndarray):
            lbls = np.array(lbls)
        stopifnot(len(np.setdiff1d(np.unique(lbls),[0,1]))==0)
        if val is not None:
            print('Splitting data into validation set')
            stopifnot(isinstance(val,float) & (val<1) & (val>0))
            np.random.seed(ss)
            idx_val = strat_split(fctr, int(np.round(1/val)))
            idx_train = np.concatenate(idx_val[1:])
            idx_val = idx_val[0]
            self.idx = {'train':idx_train,'val':idx_val}
            data_val, data = data.iloc[idx_val], data.iloc[idx_train]
            lbls_val, lbls = lbls[idx_val], lbls[idx_train]
            fctr_val, fctr = fctr.iloc[idx_val], fctr.iloc[idx_train]

        self.enc_X = col_encoder(dropfirst=True,sparse=False)
        self.enc_X.fit(data)
        self.enc_fctr = col_encoder(dropfirst=False,sparse=True,dtype=np.int8)
        self.enc_fctr.fit(pd.DataFrame(fctr))
        # Train naive-bayes on sparse-factor
        self.mdl_fctr = BernoulliNB(alpha=1,binarize=None)
        self.mdl_fctr.fit(self.transform_fctr(fctr),lbls)

        # Create design matrices
        if val is not None:
            if tt is None:
                tt = ['total', 'within', 'between']
            self.tune(data, lbls, fctr, tt=tt, val=val, nlam=nlam, lam_seq=lam_seq,
                      den2=den2, data_val=data_val, lbls_val=lbls_val, fctr_val=fctr_val)
        else:
            self.tune(data, lbls, fctr, tt=tt, val=val, den2=den2,
                      nlam=nlam, lam_seq=lam_seq)

    # nlam=50; lam_seq=None; tt=['total','within','between']
    def tune(self, data, lbls, fctr, tt, den2, val=None,
             nlam=50, lam_seq=None, data_val=None, lbls_val=None, fctr_val=None):
        if lam_seq is None:
            self.lam_seq = np.flip(np.exp(np.linspace(np.log(1), np.log(1e5), nlam)))
        else:
            self.lam_seq = np.flip(np.sort(lam_seq))
            nlam = len(self.lam_seq)
        xx = self.enc_X.transform(data)
        n_tt = len(tt)
        offset = self.predict_fctr(fctr)# Offset is log-odds
        # Get indexes
        idx_w, idx_b = idx_wb(lbls,fctr)
        idx_all, _ = idx_wb(lbls, np.repeat(1,len(lbls)))
        di_idx = dict(zip(['total','within','between'],(idx_all,idx_w,idx_b)))
        di_idx = {z: di_idx[z] for z in tt if z in di_idx}
        print('Learning coefficients across lambdas')
        x0 = np.array([np.corrcoef(xx[:, jj], lbls)[0, 1] for jj in range(xx.shape[1])])
        Bhat = np.zeros([xx.shape[1],nlam,n_tt])
        for k, t in enumerate(tt):
            print('Itetration %i of %i: %s' % (k+1,n_tt,t))
            idx = di_idx[t]
            for i, l in enumerate(self.lam_seq):
                if (i + 1) % 1 == 0: print('lambda %i of %i' % (i+1,nlam))
                if t == 'between':
                    offs = 0*offset
                else:
                    offs = offset.copy()
                import time as ti
                from timeit import timeit
                dcAUC(x0,xx,idx_all,offset,1,int(1e5))
                Bhat[:, i, k] = minimize(fun=cAUC, jac=dcAUC, x0=x0,
                         args=(xx, idx, offs, l, den2), method='l-bfgs-b').x
                x0 = Bhat[:, i, k].copy()
        self.mdl = {z: {'bhat':Bhat[:,:,k],'lam':self.lam_seq} for
                                    z,k in zip(tt,range(n_tt))}

        # Tune
        if val is not None:
            xx_val = self.enc_X.transform(data_val)
            print('Selecting Bhat based on validation')
            val_scores = np.zeros([nlam,n_tt])
            offset_val = self.predict_fctr(fctr_val)
            idx_w_val, idx_b_val = idx_wb(lbls_val, fctr_val)
            idx_all_val, _ = idx_wb(lbls_val, np.repeat(1, len(lbls_val)))
            di_idx_val = dict(zip(['total', 'within', 'between'],
                                  (idx_all_val, idx_w_val, idx_b_val)))
            di_idx_val = {z: di_idx_val[z] for z in tt if z in di_idx_val}
            for k, t in enumerate(tt):
                idx_val = di_idx_val[t]
                if t == 'between':
                    offs = 0*offset_val
                else:
                    offs = offset_val.copy()
                val_scores[:,k] = [wAUC(Bhat[:,i,k], xx_val, idx_val, offs) for i in range(nlam)]
            idx_star = val_scores.argmax(axis=0)
            lam_star = {t: self.lam_seq[i] for t, i in zip(tt,idx_star)}
            score_star = {t: val_scores[i,k] for t,i,k in zip(tt,idx_star,range(n_tt))}
            self.mdl = {t: {'cls':linreg_wAUC()} for t in tt}
            for t in self.mdl:
                self.mdl[t]['cls'].fit(data=pd.concat([data,data_val],axis=0),
                    lbls=np.append(lbls,lbls_val), fctr=pd.concat([fctr,fctr_val]),
                                       tt=[t], lam_seq=[lam_star[t]])
            # Keep only "optimal" coefficients
            self.mdl = {t: {'bhat':self.mdl[t]['cls'].mdl[t]['bhat'].flatten(),
                'lam':lam_star[t],'score':score_star[t]} for t in self.mdl}





