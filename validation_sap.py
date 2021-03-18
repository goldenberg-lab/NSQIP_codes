"""
STATISTICAL ANALYSIS PLAN FOR SSI
i) Compare SSI @SK compared to @NSQIP
-> Raw vs Propensity score adjusted rates
-> CPT codes
ii) How many CPTs do we have "tailored" models for? How many do we have baseline risk for? (As in how many using SK data)
iii) What threshold do we need for a 5% PPV? Show the lower-bound (BCA). What sensitivity/PPV/FPR would we have obtained on SK?
iv) Power analysis
"""

import os
import dill
import pickle
from joblib import dump, load
import numpy as np
import pandas as pd
from support.stats_funs import gen_CI, auc2se, bootstrap_x
from support.support_funs import find_dir_nsqip, gg_save, gg_color_hue, cvec
from support.dict import di_agg

import plotnine
from plotnine import *
from plydata.cat_tools import *

# Set directories
dir_NSQIP = find_dir_nsqip()
dir_figures = os.path.join(dir_NSQIP, 'figures')
dir_output = os.path.join(dir_NSQIP, 'output')
dir_models = os.path.join(dir_output, 'models')
lst_dir = [dir_figures, dir_output, dir_models]
assert all([os.path.exists(fold) for fold in lst_dir])

di_outcome = {'adv':'ADV', 'aki':'AKI', 'cns':'CNS',
              'nsi':'nSSIs', 'ssi':'SSIs', 'unplan':'UPLN'}

#############################
# ----- (1) LOAD DATA ----- #

# (i) Load SK predicted risk scores
df_sk = pd.read_csv(os.path.join(dir_output,'dat_sk_score.csv'))

# (ii) Load NSQIP predicted risk scores
df_ns = pd.read_csv(os.path.join(dir_output, 'best_eta.csv'))
# The method/version/model are irrelavant as these are the "winning"
df_ns.drop(columns=['model','method','version'],inplace=True)

# (iii) Low caseid/cpt dictionary
df_idt = pd.read_csv(os.path.join(dir_output, 'X_imputed.csv'), usecols=['caseid','cpt'])
df_idt.cpt = 'c'+ df_idt.cpt.astype(str)

# (iv) Load the NaiveBayes baseline
df_nb = pd.read_csv(os.path.join(dir_output, 'nbayes_phat.csv'))
# Map to the aggregate (best SSI is version 1)
cn_ssi = di_agg['ssi1']
cn_idx = ['operyr','caseid']
df_nb = df_nb[df_nb.outcome.isin(cn_ssi)]
df_nb = df_nb.pivot_table(['y','phat'],cn_idx,'outcome')
df_nb = df_nb['phat'].sum(1).reset_index().merge(df_nb['y'].sum(1).clip(0,1).reset_index(),'left',cn_idx)
df_nb.rename(columns={'0_x':'phat','0_y':'y'}, inplace=True)
# Add on CPT
df_nb = df_nb.merge(df_idt,'left','caseid')

# Subset down to SSI only
df_sk = df_sk.query('outcome == "ssi"').reset_index(None, True).drop(columns='outcome')
df_ns = df_ns.query('outcome == "ssi"').reset_index(None, True).drop(columns='outcome')

# Check that CPTs overlap
cpt_sk = pd.Series(df_sk.cpt.unique())
cpt_ns = pd.Series(df_ns.cpt.unique())
cpt_nb = pd.Series(df_nb.cpt.unique())
assert cpt_sk.isin(cpt_nb).all()  # All CPT codes should be in here
print('SK data has %i unique CPT codes, NSQIP has %i' % (len(cpt_sk),len(cpt_nb)))

# Get number of observations for later
dat_n = pd.DataFrame({'ds':['SK','NSQIP'],'n':[len(df_sk),len(df_ns)]})

#################################
# ----- (2) LOAD NB MODEL ----- #

path_mdl = os.path.join(dir_models,'NB_cpt_agg_ssi1.pickle')
assert os.path.exists(path_mdl)
with open(path_mdl, 'rb') as file:
   mdl_nb = dill.load(file)
print(mdl_nb.predict(df_sk[['cpt']])[:,1].sum())

#################################
# ----- (3) SUMMARY STATS ----- #

di_tt = {'yhat':'Predicted','y':'Actual'}

n_sk, n_ns = df_sk.shape[0], df_ns.shape[0]

alpha=0.05
n_bs, k = 1000, 5
# Compare calibration
se_yhat_sk = bootstrap_x(df_sk.preds.values, np.sum, n_bs, k=k).std(ddof=1)
se_yhat_ns = bootstrap_x(df_ns.preds.values, np.sum, n_bs, k=k).std(ddof=1)
se_y_sk = bootstrap_x(df_sk.y.values, np.sum, n_bs, k=k).std(ddof=1)
se_y_ns = bootstrap_x(df_ns.y.values, np.sum, n_bs, k=k).std(ddof=1)
se_yhat = np.array([se_yhat_sk, se_yhat_ns, se_y_sk, se_y_ns])
y_count = np.array([df_sk.preds.sum(), df_ns.preds.sum(), df_sk.y.sum(), df_ns.y.sum()])
dat_calib = pd.DataFrame({'ds':np.tile(['SK','NSQIP'],2), 'tt':np.repeat(['yhat','y'],2), 
'y':y_count,'se':se_yhat})
dat_calib = pd.concat([dat_calib,gen_CI(dat_calib.y,dat_calib.se,alpha=alpha)],1)
dat_calib.tt = dat_calib.tt.map(di_tt)

# COMPARE PROBABILITY SCORES TO LEAVE-ONE-YEAR OUT NAIVE BAYES....
di_agg

###########################
# ----- (X) FIGURES ----- #


colz = gg_color_hue(2)

# Show predicted versus actual amount
gg_calib_ssi = (ggplot(dat_calib,aes(x='tt',y='y')) + theme_bw() + 
    geom_point(size=2) + 
    geom_linerange(aes(ymin='lb',ymax='ub')) + 
    labs(y='# of SSIs') + 
    facet_wrap('~ds',scales='free_y') + 
    theme(axis_title_x=element_blank(),subplots_adjust={'wspace': 0.25}))
gg_save('gg_calib_ssi.png',dir_figures,gg_calib_ssi,8,4)


