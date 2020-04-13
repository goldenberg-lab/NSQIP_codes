#############################################################
## ---- SCRIPT TO COMPARE WITHIN-AUC FOR SIMPLE MODEL ---- ##
#############################################################

import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot
import matplotlib
matplotlib.use('Agg') # no print-outs
matplotlib.rcParams['figure.max_open_warning'] = 25
from matplotlib import pyplot as plt
import seaborn as sns

from support.linreg_wAUC import linreg_wAUC, stochastc_wb_auc
import time as ti

###############################
# ---- STEP 1: LOAD DATA ---- #

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_figures = os.path.join(dir_base,'..','figures')
[stopifnot(os.path.exists(z),'Path does not exist: '+z) for z in [dir_base, dir_output]]
for pp in [dir_figures]:
    if not os.path.exists(pp):
        print('making directory %s' % pp); os.mkdir(pp)

dir_auc = os.path.join(dir_output,'linreg_wAUC')
if not os.path.exists(dir_auc):
    print('making AUC output folder');os.mkdir(dir_auc)

dir_weights = os.path.join(dir_output,'weights')

# Labels
fn_Y = 'y_agg.csv'
use_Y = ['caseid','operyr','agg_adv1','agg_nsi4','agg_ssi2','agg_unplan2']
dat_Y = pd.read_csv(os.path.join(dir_output,fn_Y),usecols=use_Y)
cn_Y = list(dat_Y.columns[2:])
map_Y = dict(zip(cn_Y,['Adverse','non-SSI','SSI','Unplanned']))

# Data
fn_X = 'X_imputed.csv'
dat_X = pd.read_csv(os.path.join(dir_output,fn_X))
stopifnot(all(dat_X.caseid == dat_Y.caseid))
u_years = dat_X.operyr.unique()
dat_X['cpt'] = 'c'+dat_X.cpt.astype(str) # !! ENCODE CPT AS CATEGORICAL !! #
cn_X = np.setdiff1d(list(dat_X.columns[2:]),'cpt')
stopifnot(all(dat_Y.iloc[:,2:].apply(lambda x: sum(x==-1),0)==0),
          'Aggregate outcomes have missing values!')

def two2one(z):
    z1, z2 = z
    ret = pd.Series({'z1':z1,'z2':z2})
    return ret

#####################################
# ---- STEP 2: TRAIN THE MODEL ---- #

# train_years = [2012]
# test_years = np.setdiff1d(u_years, train_years)
# tt = ['total','within']
# n_bs = 101
#
# yy=test_years[0]
# for yy in test_years:
#     print('-------- Training years: %s, test year: %i --------' %
#           (', '.join([str(x) for x in train_years]),yy))
#     idx_train = dat_X.operyr.isin(train_years)
#     idx_test = (dat_X.operyr == yy)
#     Xtrain, Xtest = dat_X.loc[idx_train, cn_X], dat_X.loc[idx_test, cn_X]
#     Ytrain, Ytest = dat_Y.loc[idx_train, cn_Y], dat_Y.loc[idx_test, cn_Y]
#     cpt_train, cpt_test = dat_X.cpt.loc[idx_train], dat_X.cpt.loc[idx_test]
#     # Train model for each label
#     di_mdls = {l: [] for l in cn_Y}
#     t_now = ti.time()
#     for l in di_mdls:
#         print('------ Model for outcome: %s -------' % l)
#         ytrain, ytest = Ytrain[l].values, Ytest[l].values
#         di_mdls[l] = linreg_wAUC()
#         di_mdls[l].fit(data=Xtrain, lbls=ytrain, fctr=cpt_train,
#                        nlam=1, val=0.1, ss=1234, tt=tt)
#         cn_l = di_mdls[l].enc_X.cn_transform
#         # Get next-year's performance
#         eta_test = di_mdls[l].predict(Xtest, cpt_test)
#         dat_eta = pd.DataFrame(np.vstack([eta_test[z] for z in eta_test]).T,
#                columns=eta_test).assign(y=ytest,cpt=cpt_test.values,lbl=l,operyr=yy)
#         # Save
#         dat_eta.to_csv(os.path.join(dir_auc, 'eta_y' + str(yy)+'_lbl'+l+'.csv'),index=False)
#
#         # Get boot-strapped coefficientes
#         Bhat = np.zeros([n_bs,len(cn_l),len(tt)])
#         for jj in range(n_bs):
#             np.random.seed(jj)
#             print('Bootstrap iteration %i of %i' % (jj+1, n_bs))
#             idx_bs = np.random.choice(range(len(ytrain)), len(ytrain))
#             for k,t in enumerate(tt):
#                 lam_t = di_mdls[l].mdl[t]['lam']
#                 tmp = linreg_wAUC()
#                 tmp.fit(Xtrain.iloc[idx_bs], ytrain[idx_bs],
#                         cpt_train.iloc[idx_bs],
#                         x0=di_mdls[l].mdl[t]['bhat'],
#                         tt=[t], lam_seq=[lam_t])
#                 Bhat[jj, :, k] = tmp.mdl[t]['bhat'].flatten()
#         dat_l = pd.DataFrame(np.vstack([di_mdls[l].mdl[t]['bhat'] for t in tt]).T,
#                  columns=tt).assign(cn=cn_l,lbl=l)
#         dat_l = dat_l.melt(['cn','lbl'],None,'tt','bhat')
#         dat_bs = pd.concat([pd.DataFrame(Bhat[:, :, z], columns=cn_l).assign(tt=t,bs=range(n_bs))
#          for z,t in zip(range(len(tt)),tt)])
#         dat_bs = dat_bs.melt(['tt','bs'],None,'cn','v').pivot_table('v',['tt','cn'],'bs').reset_index()
#         dat_l = dat_l.merge(dat_bs,on=['cn','tt'])
#         dat_l.to_csv(os.path.join(dir_auc, 'bhat_y' + str(yy)+'_lbl'+l+'.csv'),index=False)
#     # Collect information
#     tdiff = ti.time() - t_now
#     print('------ Script took %i seconds for all labels ----- ' % tdiff)
#     # Update the training years
#     train_years.append(yy)

#####################################
# ---- STEP 3: AUGMENT RESULTS ---- #

from support.mdl_funs import col_encoder

fn_weights = pd.Series(os.listdir(dir_weights))
fn_weights = fn_weights[fn_weights.str.contains('df_test')].reset_index(drop=True)
fn_auc = pd.Series(os.listdir(dir_auc))
fn_bhat = fn_auc[fn_auc.str.contains('bhat')]

train_years = [2012]
test_years = np.setdiff1d(u_years, train_years)

cn_nnet = ['operyr','lbl','cpt','y','phat']

yy=test_years[0]
holder_nn, holder_auc = [], []
for yy in test_years:
    print('-------- Training years: %s, test year: %i --------' %
          (', '.join([str(x) for x in train_years]),yy))

    # --- STEP 2: GET NNET SCORES AND ACCURACY --- #
    # Get the neural network models
    fn_nn = fn_weights[fn_weights.str.contains(str(yy))].to_list()
    if len(fn_nn) > 0:
        print('--nnet--')
        fn_nn = fn_nn[0]
        tmp_nn = pd.read_csv(os.path.join(dir_weights,fn_nn),usecols=cn_nnet)
        tmp_nn = tmp_nn[tmp_nn.lbl.isin(cn_Y)].reset_index(None,True)
        tmp_nn = tmp_nn.groupby(['operyr','lbl']).apply(lambda x:
            two2one(stochastc_wb_auc(y=x['y'].values,
             score=x['phat'].values,group=x['cpt'].values))).reset_index()
        holder_nn.append(tmp_nn)

    # --- STEP 1: GET WITHIN-WEIGHT TRAINING/TEST DIFFERENCE --- #
    # Load the weights
    ff = fn_bhat[fn_bhat.str.contains(str(yy))].to_list()
    if len(ff) > 0:
        print('training-test diff')
        idx_train = dat_X.operyr.isin(train_years)
        idx_test = (dat_X.operyr == yy)
        Xtrain, Xtest = dat_X.loc[idx_train, cn_X], dat_X.loc[idx_test, cn_X]
        Ytrain, Ytest = dat_Y.loc[idx_train, cn_Y], dat_Y.loc[idx_test, cn_Y]
        cpt_train, cpt_test = dat_X.cpt.loc[idx_train], dat_X.cpt.loc[idx_test]
        # Fit encoding
        enc = col_encoder()
        enc.fit(Xtrain)
        xx = enc.transform(Xtrain)
        xx_test = enc.transform(Xtest)
        dat_weights = pd.concat([pd.read_csv(os.path.join(dir_auc, z),
                                             usecols=['cn', 'lbl', 'tt', 'bhat']) for z in ff])
        dat_weights = dat_weights.pivot_table('bhat', ['tt', 'lbl'], 'cn').reset_index()
        cn_xx = pd.Series(enc.cn_transform)
        cn_weights = pd.Series(dat_weights.columns[2:].to_list())
        cidx = np.append([0, 1], np.array([np.where(z == cn_weights)[0][0] for z in cn_xx]) + 2)
        dat_weights = dat_weights.iloc[:, cidx]
        stopifnot(all(dat_weights.columns[2:] == cn_xx))
        tt = 'within'
        for lbl in cn_Y:
            ylbl = Ytrain.loc[:, lbl].values
            ylbl_test = Ytest.loc[:, lbl].values
            print('lbl: %s, type: %s' % (lbl, tt))
            ww = dat_weights.loc[(dat_weights.lbl == lbl) & (dat_weights.tt == tt)]
            ww = ww.iloc[:, 2:].values.flatten()
            eta = xx.dot(ww)
            eta_test = xx_test.dot(ww)
            auc_train, cpt_auc_train = stochastc_wb_auc(ylbl, eta, cpt_train.values)
            auc_test, cpt_auc_test = stochastc_wb_auc(ylbl_test, eta_test, cpt_test.values)
            auc_both = pd.concat([pd.DataFrame(auc_train).T.assign(oos='train'),
                                  pd.DataFrame(auc_test).T.assign(oos='test')])
            auc_both = auc_both.assign(mdl=tt, lbl=lbl, yy=yy)
            cpt_both = pd.concat([cpt_auc_train.assign(oos='train'),
                                  cpt_auc_test.assign(oos='test')])
            cpt_both = cpt_both.assign(mdl=tt, lbl=lbl, yy=yy)
            # save
            holder_auc.append(auc_both)
    # Update the years
    train_years.append(yy)

# Tidy up
df_auc_within = pd.concat(holder_auc).melt(['yy','lbl','oos','mdl'],None,'tt','auc')
df_nnet = pd.concat(holder_nn).reset_index(None,True)

#####################################
# ---- STEP 4: ANALYZE RESULTS ---- #

# Load in the stored eta's
holder_eta, holder_bhat = [], []
for fn in fn_auc:
    tmp = pd.read_csv(os.path.join(dir_auc,fn))
    if fn.split('_')[0]=='eta':
        holder_eta.append(tmp)
    else:
        tmp = tmp.assign(operyr=int(fn.split('_')[1].replace('y','')))
        holder_bhat.append(tmp)
# Merge
df_eta = pd.concat(holder_eta).melt(['y','cpt','lbl','operyr'],None,'mdl','eta')
df_bhat = pd.concat(holder_bhat).reset_index().melt(['index','operyr','cn','lbl',
             'bhat','tt'],None,'bs_iter').drop(columns=['index'])
# qq = df_eta[(df_eta.lbl == 'agg_ssi2') & (df_eta.mdl == 'total')]
# plt.close()
# g = sns.FacetGrid(qq,hue='y')
# g.map(sns.distplot,'eta')
# g.savefig(os.path.join(dir_figures,'phat_dist.png'))
# print(np.round(qq.groupby('y').eta.apply(lambda x: pd.Series({'mu':x.mean(),'se':x.std()})).reset_index(),2))

# Get summary statistics
cidx_eta = ['lbl','operyr','mdl']
df_eta = df_eta.groupby(cidx_eta).apply(lambda x:
      two2one(stochastc_wb_auc(x.y.values,x.eta.values,x.cpt.values))).reset_index()
df_eta_tt = pd.concat([df_eta[cidx_eta],
   pd.DataFrame(np.vstack(df_eta.z1), columns=df_eta.z1[0].index)],axis=1)
df_eta_tt = df_eta_tt.melt(cidx_eta,None,'tt','auc')
# CPT-based
df_auc_cpt = pd.concat([df_eta.z2[ii].assign(lbl=df_eta.loc[ii,'lbl'],
                     operyr=df_eta.loc[ii,'operyr'],
                     mdl=df_eta.loc[ii,'mdl']) for ii in range(df_eta.shape[0])])
df_auc_cpt.reset_index(None,True,True)
# Calculate for NNet data
df_nnet_tt = pd.concat([df_nnet[['operyr','lbl']],
   pd.DataFrame(np.vstack(df_nnet.z1), columns=df_nnet.z1[0].index)],axis=1)
df_nnet_cpt = pd.concat([df_nnet.z2[ii].assign(lbl=df_nnet.loc[ii,'lbl'],
         operyr=df_nnet.loc[ii,'operyr']) for ii in range(df_nnet.shape[0])]).reset_index(None,True)


# Get the confidence intervals
cn_melt = ['operyr','cn','lbl','tt']
df_bhat_sum = df_bhat.groupby(cn_melt).value.apply(lambda x:
    pd.Series({'lb':x.quantile(0.025),'ub':x.quantile(0.975)})).reset_index()
df_bhat_sum = df_bhat_sum.pivot_table('value',cn_melt,'level_'+str(len(cn_melt)),
                                      lambda x: x).reset_index()
df_bhat_sum = df_bhat_sum.merge(pd.concat(holder_bhat)[cn_melt+['bhat']],on=cn_melt,how='left')
df_bhat_sum = df_bhat_sum.assign(sig= lambda x: np.sign(x.lb)==np.sign(x.ub))

# Q1: Does the within-coefficients do a better job that aggregate?
tmp = df_eta_tt[~(df_eta_tt.tt=='between')].assign(lbl=lambda x: x.lbl.map(map_Y))
g = sns.FacetGrid(data=tmp,row='tt',col='lbl',hue='mdl')
g.map(plt.plot,'operyr','auc',marker='o')
g.add_legend()
g.set_xlabels('Test-year'); g.set_ylabels('AUROC')
g._legend.set_title('Model')
for ax in g.axes.flat:
    ax.set_title(ax.get_title().replace('tt','AUC-type'))
g.savefig(os.path.join(dir_figures,'intra_auc_scores.png'))
# Look at within-scatter plot
df_auc_cpt_w = df_auc_cpt.pivot_table('auc',['group','lbl','operyr','npair_w'],'mdl').reset_index()
g = sns.FacetGrid(data=df_auc_cpt_w.assign(lbl=lambda x: x.lbl.map(map_Y)),col='operyr',row='lbl')
g.map(plt.scatter,'total','within')
for ax in g.axes.flatten():
    ax.plot(np.arange(0,1,0.1),np.arange(0,1,0.1),c='black')
g.set_xlabels('Total'); g.set_ylabels('Within')
g.savefig(os.path.join(dir_figures,'within_scatter_auc.png'))
# A1: There is no meaningful difference

# Q2: Are the coefficients systematically different in coefficients?
df_bhat_sig = df_bhat_sum.groupby(['operyr','lbl','tt']).sig.value_counts().reset_index(name='n')
g = sns.FacetGrid(data=df_bhat_sig.assign(lbl=lambda x: x.lbl.map(map_Y)),
                  hue='sig',col='lbl',row='tt')
g.map(plt.plot,'operyr','n',marker='o')
g.add_legend()
g.set_xlabels('Test-year'); g.set_ylabels('# of coefficients')
g._legend.set_title('Significant')
g.set(xticks=[2013,2014,2015],xticklabels=[2013,2014,2015])
g.savefig(os.path.join(dir_figures,'bhat_sig.png'))

# Scatter
tmp = df_bhat_sum.pivot_table('bhat',['operyr','cn'],'tt').reset_index()
g = sns.FacetGrid(data=tmp,hue='operyr')
g.map(sns.scatterplot,'total','within',alpha=0.5)
g.add_legend()
xx = np.arange(-1.5,1.5+1e-4,0.5)
g.set(xticks=xx,xticklabels=xx)
g.set_ylabels('Total (coefficient)');g.set_xlabels('Within (coefficients)')
g.savefig(os.path.join(dir_figures,'bhat_scatter.png'))
# A2: Coefficients appear to be fundamentally similar

# Q3: Does the NN do any better?
tmp1 = df_eta_tt[df_eta_tt.mdl == df_eta_tt.tt].assign(mdl = 'Linear')
tmp2 = df_nnet_tt.drop(columns=['between']).melt(['operyr','lbl'],None,'tt','auc').assign(mdl='NeuralNet')
dat_comp_tt = pd.concat([tmp1,tmp2],axis=0).reset_index(None,True)
tmp3 = df_auc_cpt[df_auc_cpt.mdl == 'within'].assign(mdl = 'Linear')
dat_comp_cpt = df_nnet_cpt.merge(tmp3,'left',['group','npair_w','lbl','operyr'],suffixes=('_nnet','_linear'))
print('Correlation between NeuralNet and linear model: %0.3f' %
      (np.corrcoef(dat_comp_cpt.auc_nnet,dat_comp_cpt.auc_linear)[0,1]))

# Point estimate
g = sns.FacetGrid(data=dat_comp_tt,col='lbl',row='tt',hue='mdl')
g.map(plt.plot,'operyr','auc',marker='o')
g.add_legend()
g.set_xlabels('Test-year'); g.set_ylabels('AUROC')
g._legend.set_title('Model')
# g.set(xticks=[2014, 2015],xticklabels=[2014,2015])
g.savefig(os.path.join(dir_figures,'auc_comp_tt.png'))

# Scatter plot on AUC level
g = sns.FacetGrid(data=dat_comp_cpt,row='lbl',col='operyr')
g.map(plt.scatter,'auc_linear','auc_nnet')
g.set_ylabels('AUROC (NeuralNet)');g.set_xlabels('AURUC (Linear Model)')
for ax in g.axes.flatten():
    ax.plot(np.arange(0,1,0.1),np.arange(0,1,0.1),c='black')
g.savefig(os.path.join(dir_figures,'auc_comp_cpt.png'))
# A3: Linear model with offset actually does better

# Q4: How does the distribution of performance change within CPT?
# Do more npairs or higher rates determine performance?
dat_cpt = dat_Y.assign(cpt=dat_X.cpt).drop(columns='caseid').melt(['operyr','cpt'],None,'lbl','y')
dat_cpt = dat_cpt.groupby(['operyr','cpt','lbl']).y.mean().reset_index().rename(columns={'y':'rate'})
dat_cpt = df_auc_cpt.merge(dat_cpt.rename(columns={'cpt':'group'}),'left',['group','operyr','lbl'])
dat_cpt = dat_cpt.assign(lbl=lambda x: x.lbl.map(map_Y),log_npair=lambda x: np.log(x.npair_w))
dat_cpt = dat_cpt[dat_cpt.mdl == 'within'].reset_index(None,True)

g = sns.FacetGrid(data=dat_cpt,row='lbl',col='operyr')
g.map(plt.scatter,'rate','auc')
g.set_ylabels('AUROC');g.set_xlabels('Event rate')
g.savefig(os.path.join(dir_figures,'auc_vs_eventrate.png'))

g = sns.FacetGrid(data=dat_cpt,row='lbl',col='operyr')
g.map(plt.scatter,'log_npair','auc')
g.set_ylabels('AUROC');g.set_xlabels('log(# pairwise comparisons)')
g.savefig(os.path.join(dir_figures,'auc_vs_npair.png'))

# Is the AUC consistent across years
from scipy import stats
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

tmp = dat_cpt.pivot_table('auc',['group','lbl'],'operyr').reset_index()
g = sns.PairGrid(tmp, palette=["red"])
g.map_upper(plt.scatter, s=10)
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_lower(corrfunc)
g.savefig(os.path.join(dir_figures,'auc_consistency_years.png'))
# A4: within-CPT AUC is noisy over the years, generally better for more pairs, rather than event rate


# Q5: How does training/test performance differ?
tmp = df_auc_within.assign(lbl = lambda x: x.lbl.map(map_Y))
tmp = tmp[tmp.tt == 'within'].drop(columns=['mdl','tt']).reset_index(None,True)
g = sns.FacetGrid(data=tmp,col='lbl',hue='oos')
g.map(plt.plot,'yy','auc',marker='o')
g.add_legend()
g.set_xlabels('Test-year'); g.set_ylabels('AUROC')
g._legend.set_title('OOS')
g.set(xticks=[2014, 2015],xticklabels=[2014,2015])
g.savefig(os.path.join(dir_figures,'train_test_withinAUC.png'))
# A5: Only for SSI



