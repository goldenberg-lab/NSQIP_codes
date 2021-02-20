import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from sklearn import metrics


#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#matplotlib.style.use('ggplot')

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_base, '..', 'figures')

# DESCRIPTION: THIS SCRIPT READS IN RESULTS FROM ALL MODELS AND PLOTS AUC COMPARISONS

# SAVES TO Figures:
# --- logit_results/auc_compare.png
# --- logit_results/auc_compare_phat.png
# --- logit_results/auc_compare_title.png
# --- logit_results/auc_compare_organ.png
# --- logit_results/auc_quin_{outcome}.png

# --- rf_results/auc_compare.png
# --- xgb_results/auc_compare.png

def plot_outcome_counts(read_file_1, read_file_2, save_file, plot_dir):
    temp_sub = pd.read_csv(os.path.join(dir_output, read_file_1))
    temp_agg = pd.read_csv(os.path.join(dir_output, read_file_2))
    temp_sub = recode_outcome(temp_sub)
    temp_agg = recode_outcome(temp_agg)
    plot_output = os.path.join(dir_figures, plot_dir)
    dat = pd.concat([temp_agg, temp_sub], axis=0).reset_index(drop=True)
    dat = dat.groupby(['outcome', 'model']).size().reset_index(name='counts')
    img = sns.barplot(x='outcome', y='counts', hue='model', data=dat).get_figure()
    img.tight_layout()
    img.axes[0].yaxis.get_major_formatter().set_scientific(False)
    img.savefig(os.path.join(plot_output, save_file))
    img.clf()


def subset_agg(temp_sub, temp_agg):

    # loop through outcome, year, cpt and fill agg with NA outcome/year/cpt dont exist in sub models
    outcome_list = ['agg_nsi1', 'agg_nsi2', 'agg_nsi3', 'agg_nsi4', 'agg_ssi1', 'agg_ssi2', 'agg_aki', 'agg_adv1', 'agg_adv2', 'agg_unplan1', 'agg_unplan2', 'agg_cns']
    # outcome_list = ['nSSIs', 'SSIs', 'ADV', 'UPLN', 'CNS', 'AKI']
    year_list = [2014, 2015, 2016, 2017, 2018]

    # here try to figure out why after removing cpts from agg that it's still not the same size as sub.
    outcome_data=[]
    for o in outcome_list:
        sub_outcome = temp_sub[temp_sub['outcome']== o]
        agg_outcome = temp_agg[temp_agg['outcome']==o]
        year_data = []
        for y in year_list:
            sub_year = sub_outcome[sub_outcome['test_year']==y]
            agg_year = agg_outcome[agg_outcome['test_year']==y]

            # get unique cpt from each dataset
            sub_cpt = sub_year.caseid.unique()
            agg_year = agg_year[agg_year.caseid.isin(sub_cpt)].reset_index(drop=True)
            year_data.append(agg_year)
        year_data = pd.concat(year_data)
        outcome_data.append(year_data)
    new_agg = pd.concat(outcome_data)
    return new_agg


def recode_outcome(temp_dat):
    # recode outcome
    temp_dat['outcome'] = np.where(temp_dat['outcome'] == "agg_ssi1", 'SSIs',
                                   np.where(temp_dat['outcome'] == 'agg_ssi2', 'SSIs',
                                            np.where(temp_dat['outcome'] == 'agg_adv1', 'ADV',
                                                     np.where(temp_dat['outcome'] == 'agg_adv2', 'ADV',
                                                              np.where(temp_dat['outcome'] == 'agg_nsi1', 'nSSIs',
                                                                       np.where(temp_dat['outcome'] == 'agg_nsi2', 'nSSIs',
                                                                                np.where(temp_dat['outcome'] == 'agg_nsi3', 'nSSIs',
                                                                                         np.where(temp_dat['outcome']== 'agg_nsi4', 'nSSIs',
                                                                                                  np.where(temp_dat['outcome']  == 'agg_unplan1', 'UPLN',
                                                                                                           np.where(temp_dat['outcome'] == 'agg_unplan2', 'UPLN',
                                                                                                                    np.where(temp_dat['outcome']  == 'agg_cns', 'CNS', 'AKI')))))))))))
    return temp_dat


def plot_auc_decomp(read_file_1, read_file_2, plot_dir, save_file):
    temp_sub = pd.read_csv(os.path.join(dir_output, read_file_1))
    temp_agg = pd.read_csv(os.path.join(dir_output, read_file_2))

    # create new variable to indicate if agg or sub data
    temp_sub.insert(0, 'model', 'sub')
    temp_agg.insert(0, 'model', 'agg')

    # get outpult file
    plot_output = os.path.join(dir_figures, plot_dir)
    # combine data
    dat = pd.concat([temp_agg, temp_sub], axis=0).reset_index(drop=True)

    sns.catplot(x='tt', y='auc', hue='model', col='outcome', col_wrap=5, s=12, data=dat).savefig(
        os.path.join(plot_output, save_file))


# first make bar plot with number of cpt with 3 or more occurences. function can take outcome as input
# second make plot with pct
# thrid combine table and save
def sig_threshold(file_name, plot_dir):
    df = pd.read_csv(os.path.join(dir_output, file_name))
    # filter by ssi
    outcome_names = df.outcome.unique()
    plot_output = os.path.join(dir_figures, plot_dir)
    #
    for i in outcome_names:
        ssi_df = df[df['outcome'] == i].reset_index(drop=True)
       # add 0.5 back to sig_value_agg and sig_value_sub to get their auc value at 2.5%
        ssi_df['sig_value_agg'] = ssi_df['sig_value_agg'] + 0.5
        ssi_df['sig_value_sub'] = ssi_df['sig_value_sub'] + 0.5
       # only keep observations that have sig values greater than 0 for either of the agg or sub
        ssi_df = ssi_df.loc[(ssi_df['sig_value_agg'] > 0.7) | (ssi_df['sig_value_sub'] > 0.7)].reset_index(drop=False)
        ssi_df['per_greater'] =np.nan
        for ind in ssi_df.index:
            ind_value = ssi_df.sig_value_agg[ind]
            num_greater = sum(j > ind_value for j in ssi_df.sig_value_agg)
            ssi_df['per_greater'][ind] = num_greater/len(ssi_df.index)


        plot_data = ssi_df[['sig_value_agg', 'per_greater']]
       # get outpult file
        img = sns.scatterplot(x='sig_value_agg', y='per_greater', data=plot_data).get_figure()
        img.savefig(os.path.join(plot_output, "sig_greater_{}.png".format(i)))
        img.clf()
        print(i)
#
#
def sig_year_plot(file_name, plot_dir):
    df = pd.read_csv(os.path.join(dir_output, file_name))
    # filter by ssi
    outcome_names = df.outcome.unique()
    plot_output = os.path.join(dir_figures, plot_dir)

    for i in outcome_names:
        ssi_df = df[df['outcome'] == i].reset_index(drop=True)

        # only keep observations that have sig values greater than 0 for either of the agg or sub
        ssi_df = ssi_df.loc[(ssi_df['sig_value_agg'] > 0) | (ssi_df['sig_value_sub'] > 0)].reset_index(drop=False)

        # add 0.5 back to sig_value_agg and sig_value_sub to get their auc value at 2.5%
        ssi_df['sig_value_agg'] = ssi_df['sig_value_agg'] + 0.5
        ssi_df['sig_value_sub'] = ssi_df['sig_value_sub'] + 0.5
        # tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})

        ssi_df = ssi_df.groupby(['cpt', 'test_year']).agg(
            {'sig_value_agg': 'max', 'sig_value_sub': 'max', 'outcome': 'size'}).reset_index(drop=False)
        ssi_df = ssi_df.groupby(['cpt'])['outcome'].apply(lambda x: sum(x)).reset_index(drop=False).rename(
            columns={'outcome': 'sum_n'})

        # create data frame that gives number cpt greater than 1, 2, 3...
        one = ssi_df[ssi_df['sum_n'] >= 1]['sum_n'].count()
        two = ssi_df[ssi_df['sum_n'] >= 2]['sum_n'].count()
        three = ssi_df[ssi_df['sum_n'] >= 3]['sum_n'].count()
        four = ssi_df[ssi_df['sum_n'] >= 4]['sum_n'].count()
        five = ssi_df[ssi_df['sum_n'] >= 5]['sum_n'].count()
        six = ssi_df[ssi_df['sum_n'] >= 6]['sum_n'].count()

        plot_data = pd.DataFrame({'key': ['one', 'two', 'three', 'four', 'five', 'six'],
                                  'value': [one, two, three, four, five, six]})
        # get outpult file
        img = sns.barplot(x='key', y='value', data=plot_data).get_figure()
        img.savefig(os.path.join(plot_output, "sig_years_{}.png".format(i)))
        img.clf()
        print(i)

def auc_group(df):
    y = df.y
    preds = df.preds
    if all(y==0):
        auc= np.nan
    else:
        auc =metrics.roc_auc_score(y, preds)
    return auc

def get_auc(df):
    df = df.dropna().reset_index(drop=True)
    #df['y'] = df['y'].str.strip('[]').astype(int)
    #df = df.groupby(['outcome', 'test_year', 'cpt']).apply(auc_group).reset_index().rename(columns={0: 'auc'})
    df = df.groupby(['outcome', 'cpt']).apply(auc_group).reset_index().rename(columns={0: 'auc'})

    return df

# -----------------------------------------------
# FUNCTIONS FOR PLOTTING
def plot_auc(read_file_1, read_file_2, plot_dir, save_file, generate_auc):
    # read in data
    temp_sub = pd.read_csv(os.path.join(dir_output, read_file_1))
    temp_agg = pd.read_csv(os.path.join(dir_output, read_file_2))

    #subset agg model to match sub models
    temp_agg = subset_agg(temp_sub=temp_sub, temp_agg=temp_agg)

    # recode outcome
    temp_agg= recode_outcome(temp_dat=temp_agg)
    temp_sub = recode_outcome(temp_dat=temp_sub)

    if generate_auc:
        # get auc
        temp_sub = get_auc(temp_sub)
        temp_agg = get_auc(temp_agg)

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
    sns.catplot(x='outcome', y='auc', hue='model',
                kind='violin',  data=dat).savefig(os.path.join(plot_output, save_file))

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
        plt.figure()
        img=sns.catplot(x='test_year', y='auc', hue='model',
                kind='violin', col='bin', col_wrap=3, data=temp)
        img.savefig(os.path.join(plot_output, "auc_quin_{}.png".format(i)))
        plt.close()

# -----------------------------------------------

# PLOT AUC COMPARISON FOR LOGIT, RANDOMFOREST, AND XGB BOOST
plot_auc(read_file_1='best_sub_logit.csv', read_file_2='best_agg_logit.csv', plot_dir='logit_results', save_file='auc_compare.png', generate_auc=True)
plot_auc(read_file_1='best_sub_rf.csv', read_file_2='best_agg_rf.csv', plot_dir='rf_results', save_file='auc_compare.png',generate_auc=True)
plot_auc(read_file_1='best_sub_xgb.csv', read_file_2='best_agg_xgb.csv', plot_dir='xgb_results', save_file='auc_compare.png',generate_auc=True)

# PLOT AUC DECOMPOSITION
plot_auc_decomp(read_file_1='logit_sub_model_auc_decomposed.csv', read_file_2='logit_agg_model_auc_decomposed.csv', plot_dir='logit_results', save_file='logit_auc_decomp_compare.png')
plot_auc_decomp(read_file_1='rf_sub_model_auc_decomposed.csv', read_file_2='rf_agg_model_auc_decomposed.csv', plot_dir='rf_results', save_file='rf_auc_decomp_compare.png')
plot_auc_decomp(read_file_1='xgb_sub_model_auc_decomposed.csv', read_file_2='xgb_agg_model_auc_decomposed.csv', plot_dir='xgb_results', save_file='xgb_auc_decomp_compare.png')

# PLOT AUC COMPARISON FOR LOGIT MODELS ON CPT TITLE GROUPS AND ORGANS
plot_auc(read_file_1='sub_cpt_title.csv', read_file_2='agg_cpt_title.csv', plot_dir='logit_results', save_file='auc_compare_title.png', generate_auc=False)
plot_auc(read_file_1='logit_sub_main.csv', read_file_2='logit_agg_main.csv', plot_dir='logit_results', save_file='auc_compare_main.png', generate_auc=False)
plot_auc(read_file_1='sub_cpt_organ.csv', read_file_2='agg_cpt_organ.csv', plot_dir='logit_results', save_file='auc_compare_organ.png', generate_auc=False)

# PLOT AUC COMPARISON FOR LOGIT RISK QUINTILES
plot_auc_quin(read_file_1='logit_sub_quin_cpt.csv',read_file_2='logit_agg_quin_cpt.csv',plot_dir='logit_results')

# plots for bootstrap data
sig_year_plot(file_name = 'logit_sig_cpts.csv', plot_dir='logit_results')
sig_threshold(file_name = 'logit_sig_cpts.csv', plot_dir='logit_results')

# get outcome counts between agg and submodels (for paper)
plot_outcome_counts(read_file_1='best_sub_logit.csv', read_file_2='best_agg_logit.csv', plot_dir='logit_results', save_file='outcome_counts.png')