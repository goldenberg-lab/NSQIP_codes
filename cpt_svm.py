import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot
from support.naive_bayes import mbatch_NB
from sklearn import metrics
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
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

# GROUPBY CPT AND GET NUMBER OF OBSERVATIONS
top_cpts = dat_X.groupby('cpt').size().sort_values(ascending=False)
top_cpts = pd.DataFrame({'cpt': top_cpts.index, 'count': top_cpts.values})

# KEEP ONLY CPT CODES WITH OVER 1000
top_cpts = top_cpts[top_cpts['count'] > 1000]
top_cpts = top_cpts.cpt.unique()

# SUBET BY DATA FRAMES BY CPT CODES
dat_X = dat_X[dat_X.cpt.isin(top_cpts)].reset_index(drop=True)
dat_Y = dat_Y[dat_Y.caseid.isin(dat_X.caseid)].reset_index(drop=True)

# GET COLUMNS
cn_X = list(dat_X.columns[2:])
cn_Y = list(dat_Y.columns[25:37])

# DELETE NON AGG LABELS
dat_Y.drop(dat_Y.columns[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
           axis=1, inplace=True)

#So if they take too long don't worry about it. Only other model I'd do is boosting (XGBoost with SKLearn is quite fast).
# In addition to the agg/sub comparisons, can you also see whether the significant y/cpt sub-models are different between the models?

###############################################
# ---- STEP 2: LEAVE-ONE-YEAR - ALL VARIABLES  ---- #

# START LOOP
holder_y_all = []

for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))
    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]

    holder_y = []
    for yy in tmp_train_years:
        print('Train Year %i' % (yy))
        idx_train = dat_X.operyr.isin(tmp_years) & (dat_X.operyr < yy)
        idx_test = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy)
        # get dummies
        Xtrain, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                        dat_X.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = dat_Y.loc[idx_train, [vv]].reset_index(drop=True), \
                        dat_Y.loc[idx_test, [vv]].reset_index(drop=True)

        # store cpt code
        tmp_cpt = Xtest.cpt
        # remove cpt code
        del Xtrain['cpt']
        del Xtest['cpt']
        xgb_mod = xgb.XGBClassifier()

        xgb_mod.fit(Xtrain, ytrain.values.ravel())

        xgb_preds = xgb_mod.predict_proba(Xtest)[:,1]



        # STORE RESULTS FROM AGGREGATE MODEL
        tmp_holder = pd.DataFrame({'y_preds': list(xgb_preds), 'y_values': list(ytest.values), 'cpt': list(tmp_cpt)})
        within_holder = []
        # LOOP THROUGH EACH CPT CODE
        for cc in top_cpts:
            #print('cpt %s' % (cc))
            sub_tmp_holder = tmp_holder[tmp_holder['cpt'] == cc].reset_index(drop=True)
            if all(sub_tmp_holder.y_values.values == 0):
                within_holder.append(pd.DataFrame({'auc': 'NA',
                                                   'cpt': cc}, index=[0]))
            else:
                within_holder.append(pd.DataFrame({'auc': metrics.roc_auc_score(list(sub_tmp_holder.y_values.values),
                                                                                list(sub_tmp_holder.y_preds.values)),
                                                   'cpt': cc}, index=[0]))

        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'xgb_agg.csv'), index=False)

####################################################
# ---- STEP 3: LEAVE-ONE-YEAR - ALL VARIABLES, FOR EACH CPT CODE, SUB MODELS---- #

holder_y_all = []
for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))
    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]

    holder_y = []
    for yy in tmp_train_years:
        print('Train Year %i' % (yy))
        idx_train = dat_X.operyr.isin(tmp_years) & (dat_X.operyr < yy)
        idx_test = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy)
        # get dummies
        Xtrain, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                        dat_X.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = dat_Y.loc[idx_train, [vv]].reset_index(drop=True), \
                        dat_Y.loc[idx_test, [vv]].reset_index(drop=True)

        within_holder = []
        for cc in top_cpts:
            #print('cpt %s' % (cc))
            # SUBSET XTRAIN AND XTEST BY CPT CODE
            sub_xtrain = Xtrain[Xtrain['cpt'] == cc]
            sub_xtest = Xtest[Xtest['cpt'] == cc]

            # SUBSET YTRAIN AND YTEST BY THE CORRESPONDING INDICES IN SUBSETTED XDATA
            sub_ytrain = ytrain[ytrain.index.isin(sub_xtrain.index)]
            sub_ytest = ytest[ytest.index.isin(sub_xtest.index)]

            # remove cpt column
            del sub_xtrain['cpt']
            del sub_xtest['cpt']

            # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
            if all(np.unique(sub_ytrain.values) == 0) or all(np.unique(sub_ytest.values) == 0):
                within_holder.append(pd.DataFrame({'auc': 'NA',
                                                   'cpt': cc}, index=[0]))
            else:
                # TRAIN MODEL
                xgb_mod = xgb.XGBClassifier()

                xgb_mod.fit(sub_xtrain, sub_ytrain.values.ravel())

                xgb_preds = xgb_mod.predict_proba(sub_xtest)[:, 1]


                within_holder.append(
                    pd.DataFrame({'auc': metrics.roc_auc_score(sub_ytest.values, xgb_preds), 'cpt': cc}, index=[0]))

        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'xgb_sub.csv'), index=False)

