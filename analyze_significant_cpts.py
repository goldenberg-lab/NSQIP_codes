import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
pd.set_option('display.max_columns', None)

# PART 1 DESCRIPTION: READ IN SIGNIFICANT CPT RESULTS AND JOIN WITH DATASET THAT HAS THE SUM OF POSITIVE VALUES FOR EACH
# CPT/OUTCOME/YEAR COMBINATION. THE IDEA IS TO SEE IF THE NUMBER OF POSITIVE LABELS HAS AN IMPACT ON IF THE SUB MODEL
# OUT PERFORMS THE AGG MODEL FOR A GIVEN CPT.

# PART 2 DESCRIPTION: READ IN CPT ANNOTATION DATA AND SUBSET BY SIGNIFICANT CPTS TO SEE DESCRIPTION

###############################
# ---- PART 1: PLOT NUMBER OF POSTIVE VALUE AGAINST P VALUE ---- #
dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_base, '..', 'figures')

fn_X = 'X_imputed.csv'
fn_Y = 'y_agg.csv'

dat_X = pd.read_csv(os.path.join(dir_output, fn_X))
dat_Y = pd.read_csv(os.path.join(dir_output, fn_Y))

# !! ENCODE CPT AS CATEGORICAL !! #
dat_X['cpt'] = 'c' + dat_X.cpt.astype(str)

# ONLY KEEP CPT COLUMN
dat_X = dat_X[['cpt']]

# SUBSET Y BY AGG OUTCOME
cn_Y = list(dat_Y.columns[25:37])

# DELETE NON AGG LABELS
dat_Y.drop(dat_Y.columns[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
           axis=1, inplace=True)

# JOIN OUTCOME WITH CPT DATA AND REMOVE 2012
dat = pd.merge(dat_X, dat_Y, left_index=True, right_index=True)
dat = dat[dat['operyr']!=2012].reset_index(drop=True)

# GROUP BY YEAR AND CPT AND GET SUM OF POSITIVE LABELS FOR EACH OUTCOME
dat_labels = dat.groupby(['operyr', 'cpt'])[cn_Y].apply(np.sum).reset_index()

# PUT DATA IN LONG FROMAT FOR PLOTTING
dat_labels = dat_labels.melt(id_vars=['operyr', 'cpt'], var_name='outcome', value_name='value')

# LOAD SIGNIFICANT CPTS
logit_results = pd.read_csv(os.path.join(dir_output, 'logit_sig_cpts.csv'))
logit_results = logit_results.rename(columns={'test_year': 'operyr'}, inplace=False)

# SUBSET BY OUTCOME, LATER PUT INTO LOOP
logit_results = logit_results[logit_results['outcome']=='agg_nsi1'].reset_index(drop=True)
dat_labels = dat_labels[dat_labels['outcome']=='agg_nsi1'].reset_index(drop=True)

# JOIN POSITIVE LABEL COUNTS WITH SIGNIFICANT CPTS
logit_results = pd.merge(dat_labels, logit_results, how='inner', on=['operyr', 'cpt', 'outcome'])

# PLOT THE NUMBER OF POSITIVE VALUES AGAINST AUC DIFF PVALUE
sns.set()
sns.scatterplot(x='value',y= 'diff_p_value', data=logit_results)

# SUBSET DATA BY SIGNIFICANT AGG P VALUE (AGGREGATE MODEL FOR THAT CPT WAS SIGNIFICANTLY GREATER THAN 0.5)
logit_sig = logit_results[logit_results['agg_p_value'] <=0.05].reset_index(drop=False)
sns.scatterplot(x='value',y= 'diff_p_value', data=logit_sig)

###############################
# ---- PART 2: LOOK AT CPT DESCRIPTION FOR THE SIGNIFICANT CPTS ---- #

# LOAD CPT ANNOTATION
cpt_anno = pd.read_csv(os.path.join(dir_output, 'cpt_anno.csv'))
cpt_anno['cpt'] = 'c' + cpt_anno.cpt.astype(str)

# GET SIGNICIANT CPTS - BOTH AGG AND AUC DIFF P VALUE
cpt_sig = logit_results[(logit_results['agg_p_value'] <=0.05) & (logit_results['diff_p_value'] <=0.05)].reset_index(drop=False)
cpt_sig = cpt_sig.cpt.unique()

# SUBSET ANNOTATION DATA BY SIGNIFICANT CPTS TO SEE DESCRIPTIONSS
cpt_anno_sig = cpt_anno[cpt_anno.cpt.isin(cpt_sig)].reset_index(drop=False)