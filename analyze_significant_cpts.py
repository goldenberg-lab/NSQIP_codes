import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
pd.set_option('display.max_columns', None)

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_base, '..', 'figures')

#########################################
# ---- PART 1 WHICH CPT CODES PERFORM BEST ON AGG AND SUB ---- #

# READ IN LOGIT SUB AND AGG MODELS
logit_agg = pd.read_csv(os.path.join(dir_output, 'logit_agg.csv'))
logit_sub = pd.read_csv(os.path.join(dir_output, 'logit_sub.csv'))

# COMBINE DATA BY STACKING
logit_sub = logit_sub.dropna().reset_index(drop=True)
logit_agg = logit_agg.dropna().reset_index(drop=True)
logit_sub.insert(0, 'model', 'sub')
logit_agg.insert(0, 'model', 'agg')
dat = pd.concat([logit_agg, logit_sub], axis=0).reset_index(drop=True)

# FILTER AUC OVER .8
dat =dat[dat['auc']>0.8].reset_index(drop=True)

# GROUP BY OUTCOME, MODEL, AND GET COUNTS
dat_outcome = dat.groupby(['model', 'outcome']).size().reset_index(name="counts")

# GROUP BY CPT, MODEL, AND GET COUNTS
dat_cpt = dat.groupby(['model', 'cpt']).size().reset_index(name="counts")






# LOAD SIGNIFICANT CPTS
logit_results = pd.read_csv(os.path.join(dir_output, 'logit_sig_cpts.csv'))
logit_results = logit_results.rename(columns={'test_year': 'operyr'}, inplace=False)

# SUBSET BY OUTCOME, LATER PUT INTO LOOP
logit_results = logit_results[logit_results['outcome']=='agg_nsi1'].reset_index(drop=True)

#########################################
# ---- PART 2 WHICH CPT CODES HAVE BETTER SUB AUC (SIGNIFICANT AND INSIGNIFICANT). ARE THEY BEATING A GOOD AGG AUC? ---- #




#########################################
# ----  PART 3 ARE THE BEST CPTS CONSISTENT ACROSS OUTCOMES (SSI IN PARTICULAR) ---- #





#########################################
# ---- PART 4 WHAT DRIVES THESE RESULTS? SAMPLE SIZE? AVAILABILITY OF POSITIVE OUTCOMES IN A GIVEN YEAR (CHECK SAME CPTS ACROSS YEARS) ---- #




#########################################
# ---- PART 5 DESCRIPTION: READ IN CPT ANNOTATION DATA AND SUBSET BY SIGNIFICANT CPTS TO SEE DESCRIPTION ---- #



#########################################
# ---- PART 6 RERUN LOGIT WITHOUT REMOVING SO MANY CPTS AND COMPARE TO CURRENT RESULTS WHERE ONLY ONES WITH 1000 WERE KEPT ---- #


#########################################

# ---- PART 2: PLOT NUMBER OF POSTIVE VALUE AGAINST P VALUE ---- #
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

# PUT DATA IN LONG FROMAT FOR hePLOTTING
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
# ---- PART 3: LOOK AT CPT DESCRIPTION FOR THE SIGNIFICANT CPTS ---- #

# LOAD CPT ANNOTATION
cpt_anno = pd.read_csv(os.path.join(dir_output, 'cpt_anno.csv'))
cpt_anno['cpt'] = 'c' + cpt_anno.cpt.astype(str)

# GET SIGNICIANT CPTS - BOTH AGG AND AUC DIFF P VALUE
cpt_sig = logit_results[(logit_results['agg_p_value'] <=0.05) & (logit_results['diff_p_value'] <=0.05)].reset_index(drop=False)
cpt_sig = cpt_sig.cpt.unique()

# SUBSET ANNOTATION DATA BY SIGNIFICANT CPTS TO SEE DESCRIPTIONSS
cpt_anno_sig = cpt_anno[cpt_anno.cpt.isin(cpt_sig)].reset_index(drop=False)