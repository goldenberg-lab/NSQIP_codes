"""
SCRIPT TO IMPLEMENT DIFFERENT MULTITASK ARCHITECTURES
"""

import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot, makeifnot, find_dir_nsqip
from sklearn import metrics

import torch

from support.mtask_network import mtask_nn
from support.fpc_lasso import FPC

###############################
# ---- STEP 1: LOAD DATA ---- #

# Set directories
dir_NSQIP = find_dir_nsqip()
dir_output = os.path.join(dir_NSQIP, 'output')
assert os.path.exists(dir_output)
dir_figures = os.path.join(dir_NSQIP, 'figures')
makeifnot(dir_figures)
dir_weights = os.path.join(dir_output, 'weights')
makeifnot(dir_weights)

fn_X = 'X_imputed.csv'
fn_Y = 'y_agg.csv'
dat_X = pd.read_csv(os.path.join(dir_output,fn_X))
dat_Y = pd.read_csv(os.path.join(dir_output,fn_Y))
print(dat_X.shape); print(dat_Y.shape)
stopifnot(all(dat_X.caseid == dat_Y.caseid))
u_years = dat_X.operyr.unique()
# !! ENCODE CPT AS CATEGORICAL !! #
dat_X['cpt'] = 'c'+dat_X.cpt.astype(str)
cn_X = list(dat_X.columns[2:])

# Split Y into the agg vs not
dat_agg = dat_Y.loc[:,dat_Y.columns.str.contains('^agg|caseid|operyr')]
dat_Y = dat_Y.loc[:,~dat_Y.columns.str.contains('^agg')]
cn_Y = list(dat_Y.columns[2:])
cn_agg = list(dat_agg.columns[2:])

# # If we use 2012/13 as baseline years, what is the y-prop?
# prop_Y = dat_Y.groupby('operyr')[cn_Y].apply(lambda x: x[~(x==-1)].mean()).reset_index()
# prop_Y = prop_Y.melt('operyr',var_name='outcome')
# tmp = dat_Y.groupby('operyr')[cn_Y].apply(lambda x: (x==-1).sum()).reset_index().melt('operyr',
#                                                                     value_name='n',var_name='outcome')
# prop_Y = prop_Y.merge(tmp[tmp.n > 0],how='left',on=['operyr','outcome'])
# prop_Y = prop_Y[prop_Y.n.isnull()].reset_index(drop=True).drop(columns='n')
# prop_Y['l10'] = -np.log10(prop_Y.value)

# g = sns.FacetGrid(data=prop_Y,col='outcome',col_wrap=5,sharey=True,sharex=True)
# g.map(sns.scatterplot,'operyr','l10')
# g.savefig(os.path.join(dir_figures,'outcome_prop.png'))

#####################################
# ---- STEP 2: TRAIN THE MODEL ---- #

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

train_years = [2012, 2013]
test_years = np.setdiff1d(u_years, train_years)

yy=test_years[0]
for yy in test_years:
    print('Training years: %s, test year: %i' % (', '.join([str(x) for x in train_years]),yy))
    idx_train = dat_X.operyr.isin(train_years)
    idx_test = (dat_X.operyr == yy)
    Xtrain, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                    dat_X.loc[idx_test, cn_X].reset_index(drop=True)
    Ytrain, Ytest = dat_Y.loc[idx_train, cn_Y].reset_index(drop=True), \
                    dat_Y.loc[idx_test, cn_Y].reset_index(drop=True)
    YAggtrain, YAggtest = dat_agg.loc[idx_train,cn_agg].reset_index(drop=True), \
                          dat_agg.loc[idx_test,cn_agg].reset_index(drop=True)
    cpt_train , cpt_test = dat_X.loc[idx_train,'cpt'].reset_index(drop=True), \
                        dat_X.loc[idx_test, 'cpt'].reset_index(drop=True)
    caseid_train, caseid_test = dat_X.loc[idx_train,'caseid'].values, \
                        dat_X.loc[idx_test,'caseid']
    # Initialize NN model
    mdl = mtask_nn()
    # Fit model
    mdl.fit(data=Xtrain,lbls=Ytrain,nepochs=2000,mbatch=1000,val_prop=0.1,lr=0.001)
    # fn_weights = pd.Series(os.listdir(dir_weights))
    # fn_weights = fn_weights[fn_weights.str.contains(str(yy)+'.pt$')].to_list()
    # if len(fn_weights)==1:
    #     mdl.load_state_dict(torch.load(os.path.join(dir_weights, fn_weights[0])))
    # Save network weights
    torch.save(mdl.nnet.state_dict(),
               os.path.join(dir_weights, 'mtask5_' + str(yy) + '.pt'))

    # Train sparse model on top for aggregated outcomes
    phat_train = mdl.predict(data=Xtrain, mbatch=10000)
    mdl_FPC = dict(zip(cn_agg,[FPC(standardize=True) for ii in range(len(cn_agg))]))
    for ii, cc in enumerate(cn_agg):
        print('Aggregated column: %s (%i of %i)' % (cc, ii+1, len(cn_agg)))
        y_cc = YAggtrain[cc].values
        idx_cc = np.where((y_cc == 0) | (y_cc == 1))[0]
        mdl_FPC[cc].fit(phat_train[idx_cc],y_cc[idx_cc],2)

    # Get test probabilities
    phat_test = mdl.predict(data=Xtest, mbatch=10000)
    fpc_test = np.vstack([mdl_FPC[cc].predict(phat_test) for cc in cn_agg]).T
    y_test = pd.concat([Ytest, YAggtest], axis=1)
    df_test = pd.DataFrame(np.c_[phat_test, fpc_test], columns=cn_Y + cn_agg)
    stopifnot(all(df_test.columns == y_test.columns))
    holder = []
    for cc in df_test.columns:
        df_cc = pd.DataFrame({'lbl':cc,'y':y_test[cc],'phat':df_test[cc],
        'cpt':cpt_test,'caseid':caseid_test})
        holder.append(df_cc)
    df_test = pd.concat(holder)
    df_test.insert(0,'operyr',yy)
    df_test.to_csv(os.path.join(dir_weights,'df_test_' + str(yy) + '.csv'),index=True)
    # Print the average test performance
    print(df_test[~(df_test.y == -1)].groupby('lbl').apply(lambda x: pd.Series({'auc': metrics.roc_auc_score(x['y'], x['phat']),
                                              'pr':metrics.average_precision_score(x['y'], x['phat'])})))
    # Update the training years
    train_years.append(yy)




