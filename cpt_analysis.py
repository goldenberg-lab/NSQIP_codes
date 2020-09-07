import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
pd.set_option('display.max_columns', None)

###############################
# ---- STEP 1: LOAD DATA ---- #
dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_base, '..', 'figures')

fn_X = 'X_imputed.csv'
fn_Y = 'y_agg.csv'

dat_X = pd.read_csv(os.path.join(dir_output, fn_X))
dat_Y = pd.read_csv(os.path.join(dir_output, fn_Y))

# !! ENCODE CPT AS CATEGORICAL !! #
dat_X['cpt'] = 'c' + dat_X.cpt.astype(str)

# only keep cpt
dat_X = dat_X[['cpt']]

cn_Y = list(dat_Y.columns[25:37])

# DELETE NON AGG LABELS
dat_Y.drop(dat_Y.columns[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
           axis=1, inplace=True)

# join
dat = pd.merge(dat_X, dat_Y, left_index=True, right_index=True)
dat = dat[dat['operyr']!=2012].reset_index(drop=True)

# group by year and cpt code and get sum of positive labels for each outcome
dat_labels = dat.groupby(['operyr', 'cpt'])[cn_Y].apply(np.sum).reset_index()
dat_labels = dat_labels.melt(id_vars=['operyr', 'cpt'], var_name='outcome', value_name='value')

# function that loads sig cpt and gets sum of positive labels
# load significant cpts
logit_results = pd.read_csv(os.path.join(dir_output, 'logit_sig_cpts.csv'))
logit_results = logit_results.rename(columns={'test_year': 'operyr'}, inplace=False)

# subset each by outcome
logit_results = logit_results[logit_results['outcome']=='agg_nsi1'].reset_index(drop=True)
dat_labels = dat_labels[dat_labels['outcome']=='agg_nsi1'].reset_index(drop=True)

# join data
logit_results = pd.merge(dat_labels, logit_results, how='inner', on=['operyr', 'cpt', 'outcome'])

# plot value (number of positive values for year/outcome/cpt combination) vs p values
sns.set()
sns.scatterplot(x='value',y= 'diff_p_value', data=logit_results)

# subset by agg p value significant (<0.05)
logit_sig = logit_results[logit_results['agg_p_value'] <=0.05].reset_index(drop=False)
sns.scatterplot(x='value',y= 'diff_p_value', data=logit_sig)

# read in cpt_anno and examin significant cpts
cpt_anno = pd.read_csv(os.path.join(dir_output, 'cpt_anno.csv'))
cpt_anno['cpt'] = 'c' + cpt_anno.cpt.astype(str)

# get unique cpts that have a signficant diff between aucs and agg over 0.5
cpt_sig = logit_results[(logit_results['agg_p_value'] <=0.05) & (logit_results['diff_p_value'] <=0.05)].reset_index(drop=False)
cpt_sig = cpt_sig.cpt.unique()

# subset annotation by signifcant cpts
cpt_anno_sig = cpt_anno[cpt_anno.cpt.isin(cpt_sig)].reset_index(drop=False)