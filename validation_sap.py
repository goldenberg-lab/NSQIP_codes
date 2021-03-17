"""
STATISTICAL ANALYSIS PLAN FOR SSI
i) Compare SSI @SK compared to @NSQIP
-> Raw vs Propensity score adjusted rates
-> CPT codes
"""

import os
import pickle
from joblib import dump, load
import numpy as np
import pandas as pd
from support.stats_funs import gen_CI, auc2se, bootstrap_x
from support.support_funs import find_dir_nsqip, gg_save, gg_color_hue

import plotnine
from plotnine import *
from plydata.cat_tools import *

# Set directories
dir_NSQIP = find_dir_nsqip()
dir_figures = os.path.join(dir_NSQIP, 'figures')
dir_output = os.path.join(dir_NSQIP, 'output')
lst_dir = [dir_figures, dir_output]
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

# Subset down to SSI only
df_sk = df_sk.query('outcome == "ssi"').reset_index(None, True).drop(columns='outcome')
df_ns = df_ns.query('outcome == "ssi"').reset_index(None, True).drop(columns='outcome')

dat_n = pd.DataFrame({'ds':['SK','NSQIP'],'n':[len(df_sk),len(df_ns)]})

#################################
# ----- (2) SUMMARY STATS ----- #

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


