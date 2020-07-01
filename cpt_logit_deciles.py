import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot
from support.naive_bayes import mbatch_NB
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns
from sklearn import preprocessing
from support.support_funs import stopifnot
from support.mdl_funs import normalize, idx_iter

###############################
# ---- STEP 1: LOAD DATA ---- #
dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_base, '..', 'figures')

fn_X = 'X_imputed.csv'
fn_Y = 'y_agg.csv'

dat_X = pd.read_csv(os.path.join(dir_output, fn_X))
dat_Y = pd.read_csv(os.path.join(dir_output, fn_Y))

# CREATE DUMMY VARIABLES FOR NON NUMERIC
dat_X = pd.get_dummies(dat_X)

# !! ENCODE CPT AS CATEGORICAL !! #
dat_X['cpt'] = 'c' + dat_X.cpt.astype(str)

# GET COLUMNS
cn_X = list(dat_X.columns[2:])
cn_Y = list(dat_Y.columns[25:37])


# deleted non aggregate labels
dat_Y.drop(dat_Y.columns[[1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]], axis=1, inplace=True)

# merge dat_Y and dat_X
dat = pd.merge(dat_Y, dat_X, on = 'caseid')

####################################################
# ---- STEP 3: LEAVE-ONE-YEAR - ALL VARIABLES, FOR DECICLES---- #

holder_y_all = []
for ii, vv in enumerate(cn_Y):
    # group by cpt and get mean of outcome
    cpt_groups = pd.DataFrame(dat.groupby('cpt')[vv].apply(np.mean).reset_index().rename(columns={vv: 'outcome_mean'}))

    # remove ctp codes with all zeroes in outcome vv
    cpt_groups = cpt_groups[cpt_groups['outcome_mean'] > 0].reset_index(drop=False)

    # get deciles
    cpt_groups['decile'] = pd.qcut(cpt_groups['outcome_mean'], 10, labels=False)

    # get train years
    tmp_ii = pd.concat([dat.operyr, dat[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]

    holder_y = []
    for yy in tmp_train_years:
        print('Train Year %i' % (yy))
        idx_train = dat.operyr.isin(tmp_years) & (dat.operyr < yy)
        idx_test = dat.operyr.isin(tmp_years) & (dat.operyr == yy)

        # get dummies
        Xtrain, Xtest = dat.loc[idx_train, cn_X].reset_index(drop=True), \
                        dat.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = dat.loc[idx_train, [vv]].reset_index(drop=True), \
                        dat.loc[idx_test, [vv]].reset_index(drop=True)

        # get deciles
        cpt_deciles = cpt_groups.decile.sort_values().unique()
        within_holder = []
        for cc in cpt_deciles:
            print('decile %s' % (cc))
            # get cpts from cpt_group for decile cc
            tmp_cpts = np.array(cpt_groups.cpt[cpt_groups['decile']==cc])

            # SUBSET XTRAIN AND XTEST BY CPTS
            sub_xtrain = Xtrain[Xtrain['cpt'].isin(tmp_cpts)]
            sub_xtest = Xtest[Xtest['cpt'].isin(tmp_cpts)]

            # SUBSET YTRAIN AND YTEST BY THE CORRESPONDING INDICES IN SUBSETTED XDATA
            sub_ytrain = ytrain[ytrain.index.isin(sub_xtrain.index)]
            sub_ytest = ytest[ytest.index.isin(sub_xtest.index)]

            # remove cpt column
            del sub_xtrain['cpt']
            del sub_xtest['cpt']

            # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
            if all(np.unique(sub_ytrain.values) == 0) or all(np.unique(sub_ytest.values) == 0):
                within_holder.append(pd.DataFrame({'auc': 'NA',
                                                   'cpt': cc,
                                                   'num_cpts': len(tmp_cpts)}, index=[0]))
            else:
                # TRAIN MODEL
                logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
                logit_fit = logisticreg.fit(sub_xtrain, sub_ytrain.values.ravel())

                # TEST MODEL
                logit_preds = logit_fit.predict_proba(sub_xtest)[:, 1]

                within_holder.append(
                    pd.DataFrame({'auc': metrics.roc_auc_score(sub_ytest.values, logit_preds),
                                  'cpt': cc,
                                  'num_cpts': len(tmp_cpts)}, index=[0]))

        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'logit_auc_cpt_within_deciles.csv'), index=False)
