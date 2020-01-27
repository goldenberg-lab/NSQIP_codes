"""
SRIPT TO EVALUATE THE PERFORMANCE OF THE DIFFERENT MODELS
"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score as ppv
from support.acc_funs import plot_ppv

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

# (3) Load in Delvin's results


############################################
# ---- STEP 2: CACULATE AUCs/PR curve ---- #

# Calculate AUCs by years
auc_nnet = dat_nnet.groupby(['operyr','lbl']).apply(lambda x:
        pd.Series({'auc':auc(x['y'],x['phat'])})).reset_index()
auc_nnet = auc_nnet.pivot('operyr','lbl','auc').reset_index().melt('operyr',value_name='auc',var_name='outcome')
auc_nnet = auc_nnet.sort_values(['outcome','operyr']).reset_index(drop=True)
auc_nnet.insert(0,'tt','MultiTask')

# Calculate the PR-Curves for the nnet model
ppv_nnet = dat_nnet.groupby(['operyr','lbl']).apply(lambda x:
        pd.Series({'df':plot_ppv(lbl=x['y'].values,score=x['phat'].values,figure=False,num=1000),
                   'ppv':ppv(x['y'],x['phat'])})).reset_index()
ppv_net = ppv_nnet[['operyr','lbl','ppv']].reset_index().merge(
    pd.concat([df.assign(**{'index':ii}) for df, ii in zip(ppv_nnet.df.to_list(),ppv_nnet.index)]),
    on='index')

# Merge with other data
auc_all = pd.concat([auc_NB, auc_nnet],sort=True).reset_index(drop=True)

################################
# ---- STEP 3: MAKE PLOTS ---- #

# --- (ii) PR CURVE by model --- #
g = sns.FacetGrid(data=ppv_net,col='lbl',hue='operyr',col_wrap=5)
g.map(plt.plot,'tpr','precision')
g.add_legend(title='Test year')
g.set_ylabels('Precision')
g.set_xlabels('Recall')
g.fig.suptitle('AUPRC for MultiTask Model')
for ax, ppv in zip(g.axes.flat,ppv_net.groupby(['lbl']).ppv.mean()):
    ax.text(0.2,0.8,'Average PPV = ' + str(np.round(ppv,4)))
g.savefig(os.path.join(dir_figures,'auprc_multitask.png'))


# --- (i) AUC by model --- #
g = sns.FacetGrid(data=auc_all,col='outcome',hue='tt',col_wrap=5)
g.map(sns.scatterplot,'operyr','auc')
g.add_legend(title='Models')
g._legend.loc = 'lower right'
g.set_ylabels('AUC')
g.set_xlabels('Test year')
g.savefig(os.path.join(dir_figures,'auc_all_models.png'))



