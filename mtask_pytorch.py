"""
SCRIPT TO IMPLEMENT DIFFERENT MULTITASK ARCHITECTURES
"""

import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot
from sklearn import metrics

import seaborn as sns

###############################
# ---- STEP 1: LOAD DATA ---- #

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_figures = os.path.join(dir_base,'..','figures')
dir_weights = os.path.join(dir_base,'..','weights')
for pp in [dir_figures, dir_weights]:
    if not os.path.exists(pp):
        print('making directory %s' % pp); os.mkdir(pp)

fn_X = 'X_imputed.csv'
fn_Y = 'y_bin.csv'
dat_X = pd.read_csv(os.path.join(dir_output,fn_X))
dat_Y = pd.read_csv(os.path.join(dir_output,fn_Y))
print(dat_X.shape); print(dat_Y.shape)
stopifnot(all(dat_X.caseid == dat_Y.caseid))
u_years = dat_X.operyr.unique()
# !! ENCODE CPT AS CATEGORICAL !! #
dat_X['cpt'] = 'c'+dat_X.cpt.astype(str)
cn_X = list(dat_X.columns[2:])

cn_Y = list(dat_Y.columns[2:])
missing_Y = dat_Y.melt('operyr',cn_Y)
missing_Y['value'] = (missing_Y.value==-1)
missing_Y = missing_Y.groupby(list(missing_Y.columns)).size().reset_index()
missing_Y = missing_Y.pivot_table(values=0,index=['operyr','variable'],columns='value').reset_index().fillna(0)
missing_Y.columns = ['operyr','cn','complete','missing']
missing_Y[['complete','missing']] = missing_Y[['complete','missing']].astype(int)
missing_Y['prop'] = missing_Y.missing / missing_Y[['complete','missing']].sum(axis=1)
print(missing_Y[missing_Y.prop > 0].sort_values(['cn','operyr']).reset_index(drop=True))
tmp = missing_Y[missing_Y.prop > 0].cn.value_counts().reset_index()
tmp_drop = tmp[tmp.cn > 3]['index'].to_list()
# Remove outcomes missing is two or more years
dat_Y.drop(columns=tmp_drop,inplace=True)
# Remove any Y's that have less than 100 events in 6 yeras
tmp = dat_Y.iloc[:,2:].apply(lambda x: x[~(x==-1)].sum() ,axis=0).reset_index().rename(columns={0:'n'})
tmp_drop = tmp[tmp.n < 100]['index'].to_list()
dat_Y.drop(columns=tmp_drop,inplace=True)
cn_Y = list(dat_Y.columns[2:])

# If we use 2012/13 as baseline years, what is the y-prop?
prop_Y = dat_Y.groupby('operyr')[cn_Y].apply(lambda x: x[~(x==-1)].mean()).reset_index()
prop_Y = prop_Y.melt('operyr',var_name='outcome')
tmp = dat_Y.groupby('operyr')[cn_Y].apply(lambda x: (x==-1).sum()).reset_index().melt('operyr',
                                                                    value_name='n',var_name='outcome')
prop_Y = prop_Y.merge(tmp[tmp.n > 0],how='left',on=['operyr','outcome'])
prop_Y = prop_Y[prop_Y.n.isnull()].reset_index(drop=True).drop(columns='n')
prop_Y['l10'] = -np.log10(prop_Y.value)

# g = sns.FacetGrid(data=prop_Y,col='outcome',col_wrap=5,sharey=True,sharex=True)
# g.map(sns.scatterplot,'operyr','l10')
# g.savefig(os.path.join(dir_figures,'outcome_prop.png'))

#####################################
# ---- STEP 2: TRAIN THE MODEL ---- #

import torch
from sklearn.metrics import roc_auc_score, average_precision_score
# Initialize the model

from support.mtask_network import mtask_nn

train_years = [2012, 2013]
test_years = np.setdiff1d(u_years, train_years)

#yy=test_years[0]
for yy in test_years:
    print('Training years: %s, test year: %i' % (', '.join([str(x) for x in train_years]),yy))
    idx_train = dat_X.operyr.isin(train_years)
    idx_test = (dat_X.operyr == yy)
    Xtrain, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                    dat_X.loc[idx_test, cn_X].reset_index(drop=True)
    Ytrain, Ytest = dat_Y.loc[idx_train, cn_Y].reset_index(drop=True), \
                    dat_Y.loc[idx_test, cn_Y].reset_index(drop=True)
    # Initialize NN model
    mdl = mtask_nn()
    # fn_weights = pd.Series(os.listdir(dir_weights))
    # fn_weights = fn_weights[fn_weights.str.contains(str(yy-1)+'.pt$')].to_list()
    # if len(fn_weights)==1:
    #     mdl.load_state_dict(os.path.join(dir_weights, fn_weights[0]))
    # Fit model
    mdl.fit(data=Xtrain,lbls=Ytrain,nepochs=1000,mbatch=1000,val_prop=0.1,lr=0.001)
    # Save network weights
    torch.save(mdl.state_dict(),os.path.join(dir_weights,'mtask5_' + str(yy) + '.pt'))
    # Evaluate model on test-year
    phat_test = pd.DataFrame(mdl.predict(Xtest,True),columns=Ytest.columns).melt(value_name='phat',var_name='lbl').reset_index()
    df_test = Ytest.melt(value_name='y', var_name='lbl').reset_index().merge(phat_test,on=['index','lbl'])
    df_test.insert(0,'operyr',yy)
    df_test.to_csv(os.path.join(dir_weights,'df_test_' + str(yy) + '.csv'),index=False)
    # Update the training years
    train_years.append(yy)










