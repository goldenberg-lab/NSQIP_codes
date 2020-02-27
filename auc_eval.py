"""
SRIPT TO EVALUATE THE PERFORMANCE OF THE DIFFERENT MODELS
"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score as ppv
from support.acc_funs import plot_ppv, auc_decomp, plot_auc


import seaborn as sns
from matplotlib import pyplot as plt

###############################
# ---- STEP 1: LOAD DATA ---- #

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_figures = os.path.join(dir_base,'..','figures')
dir_weights = os.path.join(dir_base,'..','weights')
for pp in [dir_figures, dir_weights]:
    if not os.path.exists(pp):
        print('making directory %s' % pp); os.mkdir(pp)

# (1) Load in the NaiveBayes data
auc_NB = pd.read_csv(os.path.join(dir_output, 'naivebayes_results.csv'))
auc_NB['tt'] = auc_NB.tt.map({'cpt':'CPT','all':'NaiveBayes'})

# (2) Load in the Mtask NNet data
fn_weights = pd.Series(os.listdir(dir_weights))
holder = []
for fn in fn_weights[fn_weights.str.contains('df_test')]:
    print(fn)
    holder.append(pd.read_csv(os.path.join(dir_weights, fn)))
dat_nnet = pd.concat(holder).drop(columns='index').sort_values(['operyr','lbl']).reset_index(drop=True)
del holder
dat_nnet = dat_nnet[dat_nnet.y >= 0].reset_index(drop=True) # Remove misisng years (i.e. sdehis)
# Merge with CPT
df_cpt = pd.read_csv(os.path.join(dir_output,'X_imputed.csv'),usecols=['operyr','cpt'])
df_cpt['idx'] = df_cpt.groupby('operyr').cumcount()
dat_nnet['idx'] = dat_nnet.groupby(['operyr','lbl']).cumcount()
dat_nnet = dat_nnet.merge(df_cpt,how='left',on=['idx','operyr'])

# (3) Load in Delvin's results
auc_delvin = pd.read_csv(os.path.join(dir_output, 'auc_clean.csv'))
auc_delvin = auc_delvin.drop(columns=['unq_set','set']).rename(columns={'validation_year':'operyr'}).melt(
    ['operyr','outcome'],var_name='tt',value_name='auc')
auc_delvin.tt = auc_delvin.tt.str.replace('_auc','').map({'glmnet':'Lasso','rf':'RForest'})

# (4) Load the CPT-NB
dat_NB = pd.read_csv(os.path.join(dir_output, 'nbayes_phat.csv'))

# # (4) Load Ben's feature importance
# tmp = pd.Series(os.listdir(dir_weights))
# fn_ben = tmp[tmp.str.contains('sdehis')].sort_values().to_list()
# holder = []
# for fn in fn_ben:
#     holder.append(pd.read_csv(os.path.join(dir_weights, fn)).drop(columns='Unnamed: 0').assign(operyr=2000+int(fn.split('_')[1].split('.')[0])))
# fi_ben = pd.concat(holder).reset_index(drop=True)
# vv_order = fi_ben.groupby('variable').score.mean().sort_values(ascending=False).reset_index()
# fi_ben.variable = pd.Categorical(values=fi_ben.variable,categories=vv_order.variable)

############################################
# ---- STEP 2: CACULATE AUCs/PR curve ---- #

cn_agg = pd.Series(dat_nnet.lbl.unique())
cn_agg = cn_agg[cn_agg.str.contains('agg')].to_list()


# Calculate within/between AUC
decomp_nnet = dat_nnet.groupby(['operyr','lbl']).apply(lambda x:
    pd.Series({'decomp':auc_decomp(x['y'].values,x['phat'].values,x['cpt'].values)})).reset_index()
decomp_nnet = decomp_nnet.drop(columns='decomp').reset_index().merge(
    pd.concat([df.assign(**{'index':ii}) for df, ii in zip(decomp_nnet.decomp.to_list(),decomp_nnet.index)]),
    how='left',on=['index'])
decomp_nnet['cn'] = np.where(decomp_nnet.lbl.isin(cn_agg),'Aggregate','Outcome')

# Calculate AUCs by years
auc_nnet = dat_nnet.groupby(['operyr','lbl']).apply(lambda x:
        pd.Series({'auc':auc(x['y'],x['phat'])})).reset_index()
auc_nnet = auc_nnet.pivot('operyr','lbl','auc').reset_index().melt('operyr',value_name='auc',var_name='outcome')
auc_nnet = auc_nnet.sort_values(['outcome','operyr']).reset_index(drop=True)
auc_nnet.insert(0,'tt','MultiTask')
auc_nnet['cn'] = np.where(auc_nnet.outcome.isin(cn_agg),'Aggregate','Outcome')

# Calculate the PR-Curves for the nnet model
ppv_nnet = dat_nnet.groupby(['operyr','lbl']).apply(lambda x:
        pd.Series({'df':plot_ppv(lbl=x['y'].values,score=x['phat'].values,figure=False,num=1000),
                   'ppv':ppv(x['y'],x['phat'])})).reset_index()
ppv_net = ppv_nnet[['operyr','lbl','ppv']].reset_index().merge(
    pd.concat([df.assign(**{'index':ii}) for df, ii in zip(ppv_nnet.df.to_list(),ppv_nnet.index)]),
    on='index')
ppv_net['cn'] = np.where(ppv_net.lbl.isin(cn_agg),'Aggregate','Outcome')

# Merge with other data
auc_all = pd.concat([auc_NB, auc_nnet,auc_delvin],sort=True).reset_index(drop=True)

# Save data for later
decomp_nnet.drop(columns='index').to_csv(os.path.join(dir_output,'df_decomp_nnet.csv'),index=False)

# Calculate overall between NB and NN
cal_tt = pd.concat([dat_nnet.drop(columns=['idx','cpt']).assign(tt='nnet'),
   dat_NB.assign(tt='cpt').rename(columns={'outcome':'lbl'})],axis=0).groupby(['lbl','tt']).apply(lambda x:
        pd.Series({'df':plot_ppv(lbl=x['y'].values,score=x['phat'].values,figure=False,num=1000),
                   'ppv':ppv(x['y'].values,x['phat'].values)})).reset_index()
cal_tt = cal_tt[['tt','lbl','ppv']].reset_index().merge(
    pd.concat([df.assign(**{'index':ii}) for df, ii in zip(cal_tt.df.to_list(),cal_tt.index)]),
    on='index')
cal_tt['cn'] = np.where(cal_tt.lbl.isin(cn_agg),'Aggregate','Outcome')
cal_tt.tt = cal_tt.tt.map({'cpt':'CPT-only','nnet':'MultiTask'})

################################
# ---- STEP 3: MAKE PLOTS ---- #

# # --- (iv) Feature importance for RF Sdehis --- #
# g = sns.catplot(y='variable',x='score',col='operyr',data=fi_ben,col_wrap=2,margin_titles=True)
# g.set_ylabels('')
# g.set_xlabels('Importance Score')
# g.fig.suptitle('RForest importance scores for sdehis',**{'x':0.5,'y':1.00})
# g.savefig(os.path.join(dir_figures,'RForest_importance.png'))

# --- (iii) AUC DECOMPOSITION --- #
tmp = decomp_nnet.copy()
tmp.operyr = np.where(tmp.tt == 'within', tmp.operyr-0.1, np.where(tmp.tt == 'between',tmp.operyr +0.1, tmp.operyr))
tmp['lden'] = np.log10(tmp.den)
tmp['tt'] = tmp.tt.map({'tot':'Total','within':'Within','between':'Between'})

for cn in tmp.cn.unique():
    g = sns.FacetGrid(data=tmp[tmp.cn == cn], col='lbl', hue='tt', col_wrap=5)
    g.map(sns.scatterplot, 'operyr', 'auc')  # lden
    g.map(plt.axhline, y=0.5, ls='--', c='black')
    g.add_legend(title='AUC Decomposition')
    g.set_ylabels('AUROC')  # g.set_ylabels('log10( # pairwise )')
    g.set_xlabels('Test year')
    g.savefig(os.path.join(dir_figures, 'auc_decomp_n_'+cn+'.png'))


# --- (ii) PR CURVE by model --- #
for cn in tmp.cn.unique():
    g = sns.FacetGrid(data=ppv_net[ppv_net.cn == cn],col='lbl',hue='operyr',col_wrap=5)
    g.map(plt.plot,'tpr','precision')
    g.add_legend(title='Test year')
    g.set_ylabels('Precision')
    g.set_xlabels('Recall')
    #g.fig.suptitle('AUPRC for MultiTask Model')
    for ax, ppv in zip(g.axes.flat,ppv_net[ppv_net.cn == cn].groupby(['lbl']).ppv.mean()):
        ax.text(0.2,0.8,'Average PPV = ' + str(np.round(ppv,4)))
    g.savefig(os.path.join(dir_figures,'auprc_multitask_'+cn+'.png'))

for cn in ['Aggregate','Outcome']:
    g = sns.FacetGrid(data=cal_tt[cal_tt.cn == cn],col='lbl',hue='tt',col_wrap=5,sharey=False)
    g.map(plt.plot,'tpr','precision')
    g.add_legend(title='Model')
    g.set_ylabels('Precision')
    g.set_xlabels('Recall')
    g.map(plt.axhline, y=0.1, ls='--', c='black')
    #g.fig.suptitle('AUPRC for MultiTask Model')
    for ax, ppv, ppv2 in zip(g.axes.flat,cal_tt[cal_tt.cn == cn].groupby(['lbl']).ppv.mean(),
                       cal_tt[cal_tt.cn == cn].groupby(['lbl']).precision.max()):
        ax.text(0.2,0.8*ppv2,'Average PPV = ' + str(np.round(ppv,4)))
    g.savefig(os.path.join(dir_figures,'auprc_comp_'+cn+'.png'))

# --- (i) AUC by model --- #
g = sns.FacetGrid(data=auc_all,col='outcome',hue='tt',col_wrap=5)
g.map(sns.stripplot,'operyr','auc',**{'jitter':1})
g.add_legend(title='Models')
g._legend.loc = 'lower right'
g.set_ylabels('AUC')
g.set_xlabels('Test year')
g.savefig(os.path.join(dir_figures,'auc_all_models.png'))



