# load necessary modules
import numpy as np
import pandas as pd
import os
import gc
from support.support_funs import stopifnot, gg_color_hue
from plotnine import *

import seaborn as sns
from matplotlib import pyplot as plt

# set up directories
dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_figures = os.path.join(dir_base,'..','figures')
stopifnot(all([os.path.exists(x) for x in [dir_output,dir_figures]]))

# Load in the accuracy functions
from support.acc_funs import auc, plot_ppv

##############################################
### ---- (1) load in and process data ---- ###

# load delvin's output
df_scores = pd.read_csv(os.path.join(dir_output,'maizlin_res_df.csv')).rename(columns={'actual_y':'y','model_y':'yhat'})
# load in associated x_data for each
cn_X=['caseid','operyr','cpt','race','sex','age_days']
# Merge
df_scores = df_scores.merge(pd.read_csv(os.path.join(dir_output,'X_preop.csv'),usecols=cn_X), on='caseid',how='left')

# Get CPT counts
dat_n_cpt = df_scores[df_scores.outcome == 'orgspcssi'].cpt.value_counts().reset_index().rename(columns={'cpt':'n','index':'cpt'})

# How many CPT codes have "outcomes"
dat_n_y_cpt = df_scores.groupby(['outcome','y','cpt']).size().reset_index().rename(columns={0:'n'})
dat_n_y_cpt = dat_n_y_cpt.pivot_table(values='n',index=['outcome','cpt'],columns='y').reset_index().melt(id_vars=['outcome','cpt'],value_name='n').fillna(0)
dat_n_y_cpt.n = dat_n_y_cpt.n.astype(int)
dat_n_y_cpt['exists'] = np.where(dat_n_y_cpt.n > 0 ,'yes','no')
dat_outcome_share = dat_n_y_cpt.groupby(['outcome','y','exists']).size().reset_index().rename(columns={0:'n'})
dat_outcome_share = dat_outcome_share[(dat_outcome_share.y == 1) & (dat_outcome_share.exists == 'yes')].reset_index(drop=True)
dat_outcome_share['cpt_share'] = dat_outcome_share.n / dat_n_y_cpt.cpt.unique().shape[0]
print(np.round(dat_outcome_share,2))

#################################
### ---- (2) Naive Bayes ---- ###

from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import OneHotEncoder
cn_nb = ['y','cpt']
ylbls = df_scores.outcome.value_counts().index.to_list()
auc_nb = np.zeros(len(ylbls))
for i, y in enumerate(ylbls):
    print(y)
    tmp_y = df_scores.query('outcome==@y')
    tmp_y_train = tmp_y.query('set=="training"')[cn_nb]
    tmp_y_val = tmp_y.query('set=="validation"')[cn_nb]
    ytrain, yval = tmp_y_train.y.values, tmp_y_val.y.values
    xtrain, xval = np.atleast_2d(tmp_y_train.cpt).T, np.atleast_2d(tmp_y_val.cpt).T
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore', dtype=int)
    enc.fit(xtrain)
    clf = BernoulliNB().fit(X=enc.transform(xtrain),y=ytrain)
    pval_val = clf.predict_proba(X=enc.transform(xval))[:,1]
    auc_val = auc(y=yval,score=pval_val)
    auc_nb[i] = auc_val

df_nb = pd.DataFrame({'outcome':ylbls, 'auc_nb':auc_nb})

#############################
### ---- (3) Metrics ---- ###

# (i) AUC by outcome
dat_auc = df_scores.groupby('outcome').apply(lambda x: pd.Series({'auc':auc(y=x['y'].values,score=x['yhat'].values)})).reset_index()
dat_auc = dat_auc.sort_values('auc',ascending=False).reset_index(drop=True)
dat_auc = dat_auc.merge(df_scores.groupby('outcome').y.sum().reset_index())

print(dat_auc)
# Dictionary
di_lbls = {'orgspcssi':'Organ/Space SSI','dehis':'Deep Wound Disruption',
 'wndinfd':'Deep Incisional SSI','sdehis':'Superficial Wound Disruption',
 'supinfec':'Superficial Incisional SSI'}
# AUC within each outcome
dat_auc_cpt = df_scores.groupby(['outcome','cpt']).apply(lambda x:
            pd.Series({'auc':auc(y=x['y'].values,score=x['yhat'].values)})).reset_index()
dat_auc_cpt = dat_auc_cpt[dat_auc_cpt.auc.notnull()].merge(dat_auc.rename(columns={'auc':'tot'}),on='outcome',how='left')
dat_auc_cpt = dat_auc_cpt.merge(dat_n_y_cpt[dat_n_y_cpt.y == 1].drop(columns='y'),on=['outcome','cpt'])
dat_auc_cpt['share'] = dat_auc_cpt.n / dat_auc_cpt.y
dat_auc_cpt_agg = dat_auc_cpt.groupby('outcome').apply(lambda x:
        pd.Series({'mu':np.mean(x['auc']),'wmu':np.average(x['auc'],weights=x['share'])})).reset_index()

# Combine
dat_auc_both = dat_auc.merge(dat_auc_cpt_agg).drop(columns=['y','wmu'])
dat_auc_both.rename(columns={'auc':'auc_agg','mu':'auc_within'},inplace=True)
dat_auc_both = dat_auc_both.assign(lbl=lambda x: x.outcome.map(di_lbls))
dat_auc_both = dat_auc_both.merge(df_nb.assign(lbl=lambda x: x.outcome.map(di_lbls)))


# Compare the Maizlin results
di_maizlin = {'Deep Wound Disruption':0.754, 'Superficial Incisional SSI':0.701,'Deep Incisional SSI':0.755, 'Organ/Space SSI':0.839}
df_maizlin = pd.DataFrame.from_dict(di_maizlin,orient='index').reset_index().rename(columns={'index':'lbl',0:'auc_maizlin'})
df_maizlin = dat_auc_both.merge(df_maizlin).drop(columns='outcome').melt('lbl',None,'tt')
# Make a plot
tit = 'Decomposition of within vs overall AUROC score\nWithin AUROC is average over all CPT codes'
di_lblz = {'auc_agg':'AUROC (ours)', 'auc_within':'within-AUROC (ours)',
           'auc_nb':'AUROC-NB', 'auc_maizlin':'AUROC (Maizlin)'}
colz = gg_color_hue(2) + ['black']
tmp_df = df_maizlin.assign(tt=lambda x: x.tt.map(di_lblz)).query('tt!="AUROC-NB"')
gg_maizlin = (ggplot(tmp_df, aes(x='lbl',y='value',color='tt')) +
              theme_bw() + geom_point(size=2,position=position_dodge(0.5)) +
              ggtitle(tit) + labs(y='AUROC') +
              theme(axis_text_x=element_text(angle=45),axis_title_x=element_blank()) +
              scale_color_manual(name='AUROC', values=colz) +
              geom_hline(yintercept=0.5, linetype='--') +
              scale_y_continuous(limits=[0.49,1]))
gg_maizlin.save(os.path.join(dir_figures,'gg_maizlin.png'),height=4,width=6)

qq = df_maizlin.assign(value=lambda x: x.value-0.5).pivot('lbl','tt','value')
qq.assign(share=lambda x: x.auc_within/x.auc_agg).share.mean()

# # AUC by year
# dat_auc_yr
# df_scores.groupby(['outcome','operyr']).apply(lambda x:
#             pd.Series({'auc':auc(y=x['y'].values,score=x['yhat'].values)})).reset_index()

#############################
### ---- (4) FIGURES ---- ###

# Detioration in AUC by CPT code
tmp = dat_auc_cpt_agg.merge(dat_auc).melt(id_vars='outcome',value_vars=['auc','wmu']).rename(columns={'variable':'type'})
tmp.type = tmp.type.map({'auc':'Overall','wmu':'CPT-Weighted'})
tmp2 = dat_auc_cpt_agg.merge(dat_auc)
tmp2['gain_cpt'] = tmp2.auc - tmp2.wmu
tmp2['gain_other'] = (tmp2.auc-0.5) - tmp2.gain_cpt
print(tmp2)
fig = sns.catplot(x='outcome',y='value',hue='type',legend=True,data=tmp,s=7,legend_out=False)
(fig.set(ylim=(0.5,1.0))
    .set_axis_labels(x_var='',y_var='AUC'))
plt.legend(title='')
fig.savefig(os.path.join(dir_figures,'cpt_auc_decomp.png'))

# Share of CPT codes that even have outcomes
fig = sns.scatterplot(x='outcome',y='cpt_share',hue=None,data=dat_outcome_share,s=50)
fig.set_title('Share of CPT with outcome')
fig.set_ylabel('Percent');fig.set_xlabel('')
fig.figure.savefig(os.path.join(dir_figures,'cpt_outcome_share.png'))















