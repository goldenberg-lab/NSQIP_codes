import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_base, '..', 'figures')

# DESCRIPTION: THIS SCRIPT READS IN RESULTS FROM ALL MODELS AND PLOTS AUC COMPARISONS

# SAVES TO OUTPUT:
# --- logit_results/auc_compare.png
# --- logit_results/auc_compare_phat.png
# --- logit_results/auc_compare_title.png
# --- logit_results/auc_compare_organ.png
# --- logit_results/auc_quin_{outcome}.png

# --- rf_results/auc_compare.png
# --- xgb_results/auc_compare.png

# -----------------------------------------------
# FUNCTIONS FOR PLOTTING
def plot_auc(read_file_1, read_file_2,  plot_dir, save_file):
    # read in data
    temp_sub = pd.read_csv(os.path.join(dir_output, read_file_1))
    temp_agg = pd.read_csv(os.path.join(dir_output, read_file_2))

    # remove NA
    temp_sub = temp_sub.dropna().reset_index(drop=True)
    temp_agg = temp_agg.dropna().reset_index(drop=True)

    # create new variable to indicate if agg or sub data
    temp_sub.insert(0, 'model', 'sub')
    temp_agg.insert(0, 'model', 'agg')

    # get outpult file
    plot_output = os.path.join(dir_figures, plot_dir)
    # combine data
    dat = pd.concat([temp_agg, temp_sub], axis=0).reset_index(drop=True)
    sns.catplot(x='test_year', y='auc', hue='model',
                kind='violin', col='outcome', col_wrap=5, data=dat).savefig(os.path.join(plot_output, save_file))

def clean_quin(temp_quin):
    del temp_quin['num_obs']
    # group by outcome, test_year, bin and inner quartile range
    temp = temp_quin.groupby(['outcome', 'test_year', 'bin']).describe().stack(level=0)[['25%', 'mean', '75%']].reset_index()
    del temp['level_3']
    temp = temp.rename(columns={'25%': 'X_1', 'mean': 'X_2', '75%': 'X_3'}, inplace=False)
    temp = pd.wide_to_long(temp, stubnames='X_', i=['test_year', 'outcome', 'bin'], j="iqr").reset_index().rename(
        columns={'X_': 'auc'})
    temp['iqr'].replace([1, 2, 3], ['25%', 'mean', '75%'], inplace=True)
    return temp

def plot_auc_quin(read_file_1, read_file_2,  plot_dir):
    # read in data
    temp_sub = pd.read_csv(os.path.join(dir_output, read_file_1))
    temp_agg = pd.read_csv(os.path.join(dir_output, read_file_2))

    # clean use this to do inter quartile range - but not finished
    #temp_sub = clean_quin(temp_quin=temp_sub)
    #temp_agg = clean_quin(temp_quin=temp_agg)

    # remove NA
    temp_sub = temp_sub.dropna().reset_index(drop=True)
    temp_agg = temp_agg.dropna().reset_index(drop=True)

    # create new variable to indicate if agg or sub data
    temp_sub.insert(0, 'model', 'sub')
    temp_agg.insert(0, 'model', 'agg')

    # get outpult file
    plot_output = os.path.join(dir_figures, plot_dir) # HERE MAYBE?
    # combine data
    dat = pd.concat([temp_agg, temp_sub], axis=0).reset_index(drop=True)
    outcome_names = dat.outcome.unique()
    for i in outcome_names:
        temp = dat[dat['outcome']==i].reset_index(drop=True)
        sns.catplot(x='test_year', y='auc', hue='model',
                kind='violin', col='bin', col_wrap=3, data=temp).savefig(os.path.join(plot_output, "auc_quin_{}.png".format(i)))

# -----------------------------------------------

# PLOT AUC COMPARISON FOR LOGIT, RANDOMFOREST, AND XGB BOOST
plot_auc(read_file_1='logit_sub.csv', read_file_2='logit_agg.csv', plot_dir='logit_results', save_file='auc_compare.png')
plot_auc(read_file_1='logit_sub_phat.csv', read_file_2='logit_agg_phat.csv', plot_dir='logit_results', save_file='auc_compare_phat.png')
plot_auc(read_file_1='rf_sub.csv', read_file_2='rf_agg.csv', plot_dir='rf_results', save_file='auc_compare.png')
plot_auc(read_file_1='xgb_sub.csv', read_file_2='xgb_agg.csv', plot_dir='xgb_results', save_file='auc_compare.png')

# PLOT AUC COMPARISON FOR LOGIT MODELS ON CPT TITLE GROUPS AND ORGANS
plot_auc(read_file_1='sub_cpt_title.csv', read_file_2='agg_cpt_title.csv', plot_dir='logit_results', save_file='auc_compare_title.png')
plot_auc(read_file_1='sub_cpt_organ.csv', read_file_2='agg_cpt_organ.csv', plot_dir='logit_results', save_file='auc_compare_organ.png')

# PLOT AUC COMPARISON FOR LOGIT RISK QUINTILES
plot_auc_quin(read_file_1='logit_sub_quin_cpt.csv',read_file_2='logit_agg_quin_cpt.csv',plot_dir='logit_results')

