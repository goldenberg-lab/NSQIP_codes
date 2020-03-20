#############################################################
## ---- SCRIPT TO COMPARE WITHIN-AUC FOR SIMPLE MODEL ---- ##
#############################################################

import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot
import matplotlib
matplotlib.use('Agg') # no print-outs

###############################
# ---- STEP 1: LOAD DATA ---- #

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_figures = os.path.join(dir_base,'..','figures')
[stopifnot(os.path.exists(z),'Path does not exist: '+z) for z in [dir_base, dir_output]]
for pp in [dir_figures]:
    if not os.path.exists(pp):
        print('making directory %s' % pp); os.mkdir(pp)

fn_X = 'X_imputed.csv'
fn_Y = 'y_agg.csv'
# Keep only specific aggregated
use_Y = ['caseid','operyr','agg_adv1','agg_nsi4','agg_ssi2','agg_unplan2']
dat_Y = pd.read_csv(os.path.join(dir_output,fn_Y),usecols=use_Y)
dat_X = pd.read_csv(os.path.join(dir_output,fn_X))

print(dat_X.shape); print(dat_Y.shape)
stopifnot(all(dat_X.caseid == dat_Y.caseid))
u_years = dat_X.operyr.unique()
dat_X['cpt'] = 'c'+dat_X.cpt.astype(str) # !! ENCODE CPT AS CATEGORICAL !! #
cn_X = np.setdiff1d(list(dat_X.columns[2:]),'cpt')
cn_Y = list(dat_Y.columns[2:])
map_Y = dict(zip(cn_Y,['Adverse','non-SSI','SSI','Unplanned']))
stopifnot(all(dat_Y.iloc[:,2:].apply(lambda x: sum(x==-1),0)==0),
          'Aggregate outcomes have missing values!')

#####################################
# ---- STEP 2: TRAIN THE MODEL ---- #

dir_auc = os.path.join(dir_output,'linreg_wAUC')
if not os.path.exists(dir_auc):
    os.mkdir(dir_auc)

from support.linreg_wAUC import linreg_wAUC
import time as ti

train_years = [2012]
test_years = np.setdiff1d(u_years, train_years)
tt = ['total','within']
n_bs = 2

yy=test_years[0]
for yy in test_years[0:1]:
    print('-------- Training years: %s, test year: %i --------' %
          (', '.join([str(x) for x in train_years]),yy))
    idx_train = dat_X.operyr.isin(train_years)
    idx_test = (dat_X.operyr == yy)
    Xtrain, Xtest = dat_X.loc[idx_train, cn_X], dat_X.loc[idx_test, cn_X]
    Ytrain, Ytest = dat_Y.loc[idx_train, cn_Y], dat_Y.loc[idx_test, cn_Y]
    cpt_train, cpt_test = dat_X.cpt.loc[idx_train], dat_X.cpt.loc[idx_test]
    # Train model for each label
    di_mdls = {l: [] for l in cn_Y}
    t_now = ti.time()
    for l in di_mdls:
        print('------ Model for outcome: %s -------' % l)
        ytrain, ytest = Ytrain[l].values, Ytest[l].values
        di_mdls[l] = linreg_wAUC()
        di_mdls[l].fit(data=Xtrain, lbls=ytrain, fctr=cpt_train,
                       nlam=2, val=0.1, ss=1234, tt=tt)
        cn_l = di_mdls[l].enc_X.cn_transform
        # Get next-year's performance
        eta_test = di_mdls[l].predict(Xtest, cpt_test)
        dat_eta = pd.DataFrame(np.vstack([eta_test[z] for z in eta_test]).T,
               columns=eta_test).assign(y=ytest,cpt=cpt_test.values,lbl=l,operyr=yy)
        # Save
        dat_eta.to_csv(os.path.join(dir_auc, 'eta_y' + str(yy)+'_lbl'+l+'.csv'),index=False)

        # Get boot-strapped coefficientes
        Bhat = np.zeros([n_bs,len(cn_l),len(tt)])
        for jj in range(n_bs):
            np.random.seed(jj)
            print('Bootstrap iteration %i of %i' % (jj+1, n_bs))
            idx_bs = np.random.choice(range(len(ytrain)), len(ytrain))
            for k,t in enumerate(tt):
                lam_t = di_mdls[l].mdl[t]['lam']
                tmp = linreg_wAUC()
                tmp.fit(Xtrain.iloc[idx_bs], ytrain[idx_bs],
                        cpt_train.iloc[idx_bs], tt=[t], lam_seq=[lam_t])
                Bhat[jj, :, k] = tmp.mdl[t]['bhat'].flatten()
        dat_l = pd.DataFrame(np.vstack([di_mdls[l].mdl[t]['bhat'] for t in tt]).T,
                 columns=tt).assign(cn=cn_l,lbl=l)
        dat_l = dat_l.melt(['cn','lbl'],None,'tt','bhat')
        dat_bs = pd.concat([pd.DataFrame(Bhat[:, :, z], columns=cn_l).assign(tt=t,bs=range(n_bs))
         for z,t in zip(range(len(tt)),tt)])
        dat_bs = dat_bs.melt(['tt','bs'],None,'cn','v').pivot_table('v',['tt','cn'],'bs').reset_index()
        dat_l = dat_l.merge(dat_bs,on=['cn','tt'])
        dat_l.to_csv(os.path.join(dir_auc, 'bhat_y' + str(yy)+'_lbl'+l+'.csv'),index=False)
    # Collect information
    tdiff = ti.time() - t_now
    print('------ Script took %i seconds for one label ----- ' % tdiff)

# # Do some bootstrapped coefficient simulations
# holder_score, holder_bhat = [], []
# for ii in range(nsim):
#     print('--------- Simulation %i of %i ----------' % (ii + 1, nsim))
#     df_train, df_test, y_train, y_test, group_train, \
#         group_test = train_test_split(df,y,group,test_size=0.2,stratify=group,random_state=ii)
#     mdl_wAUC = linreg_wAUC()
#     mdl_wAUC.fit(data=df_train, lbls=y_train, fctr=group_train, val=0.25, ss=ii, nlam=10)
#     tmp_bhat = pd.DataFrame(np.vstack([mdl_wAUC.mdl[z]['bhat'] for z in mdl_wAUC.mdl]).T,
#                  columns=mdl_wAUC.mdl).assign(cn=mdl_wAUC.enc_X.cn_transform)
#     holder_bhat.append(tmp_bhat)
#     # compare performance across dimensions
#     idx_val = mdl_wAUC.idx['val']
#     di_eta = mdl_wAUC.predict(df_test, group_test)
#     for t1 in di_eta:
#         tmp_metric = mdl_wAUC.AUC(eta=di_eta[t1], fctr=group_test,lbls=y_test, tt=di_eta)
#         tmp_df = pd.DataFrame(tmp_metric.items(), columns=['metric', 'score']).assign(eta=t1)
#         holder_score.append(tmp_df)

# score = pd.concat(holder_score).reset_index(drop=True)
# score_summary = score.groupby(['metric','eta']).score.mean().reset_index().pivot('metric','eta','score').reset_index()
# bhat = pd.concat(holder_bhat).melt('cn',None,'metric','bhat')
# bhat_summary = bhat.groupby(['cn','metric']).bhat.apply(lambda x:
#      pd.Series({'mu':x.mean(),'se':x.std(),'lb':x.quantile(0.025),'ub':x.quantile(0.975)})).reset_index()
# bhat_summary = bhat_summary.pivot_table('bhat',['cn','metric'],'level_2',lambda x: x).reset_index()
# bhat_summary.columns = list(bhat_summary.columns)
# bhat_summary = bhat_summary.assign(sig1=lambda x: np.sign(x.lb)==np.sign(x.ub))
# print(bhat_summary.pivot('cn','metric','sig1').reset_index())
#
# import matplotlib
# matplotlib.use('Agg') #TkAgg
# from matplotlib import pyplot as plt
# import seaborn as sns
# g = sns.FacetGrid(bhat,col='cn',col_wrap=4,sharey=False,sharex=False,height=4,aspect=1.7,hue='metric')
# g.map(sns.distplot,'bhat')

