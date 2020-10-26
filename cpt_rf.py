import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from support.acc_funs import auc_decomp
from sklearn.model_selection import GridSearchCV

# DESCRIPTION: THIS SCRIPT GENERATES AUC SCORES FOR THE AGGREGATE AND SUB MODELS.
# SAVES TO OUTPUT:
# --- rf_agg.csv
# --- rf_sub.csv

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

###############################################
# ---- STEP 2: LEAVE-ONE-YEAR - ALL VARIABLES  ---- #

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
        Xtrain, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                        dat_X.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = dat_Y.loc[idx_train, [vv]].reset_index(drop=True), \
                       dat_Y.loc[idx_test, [vv]].reset_index(drop=True)

        # STORE CPT CODES AND DELETE FROM DATA
        # scale data, code from statsmodels
        tmp_cpt = Xtest.cpt
        del Xtrain['cpt']
        del Xtest['cpt']

        # HERE implement Random forest grid search
        # grid search
        param_grid = {
            'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'n_estimators': [100, 200, 300, 1000]
        }

        clf = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=6)

        random_forest = clf.fit(Xtrain, ytrain.values.ravel())
        rf_preds = random_forest.predict_proba(Xtest)[:, 1]

        # STORE RESULTS FROM AGGREGATE MODEL
        within_holder = []
        tmp_holder = pd.DataFrame({'y_preds': list(rf_preds), 'y_values': list(ytest.values), 'cpt': list(tmp_cpt)})
        within_holder.append(pd.DataFrame({'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,'cpt': tmp_holder.cpt}))        # LOOP THROUGH EACH CPT CODE

        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'rf_agg.csv'), index=False)

###############################################
# decompose auc and save
#read_file_1 = 'rf_agg.csv'
#res_y_all = pd.read_csv(os.path.join(dir_output, read_file_1))
res_y_all = res_y_all.dropna().reset_index(drop=True)
res_y_all['y'] = res_y_all['y'].str.strip('[]').astype(int)

result_list = []
for i in cn_Y:
    print(i)
    temp_list = []
    sub_res = res_y_all[res_y_all['outcome']==i].reset_index(drop=True)
    y_values = sub_res.y.values
    pred_values = sub_res.preds.values
    group_values = sub_res.cpt.values
    agg_auc_decomp = auc_decomp(y=y_values, score=pred_values, group=group_values, rand=False)
    temp_list.append(pd.DataFrame({'tt': agg_auc_decomp.tt, 'auc': agg_auc_decomp.auc,
                                       'den': agg_auc_decomp.den}))
    result_list.append(pd.concat(temp_list).assign(outcome=i))

agg_model_auc = pd.concat(result_list).reset_index(drop=True)
agg_model_auc.to_csv(os.path.join(dir_output, 'rf_agg_model_auc_decomposed.csv'), index=False)

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
                within_holder.append(pd.DataFrame({'y': 'NA',
                                                   'preds':'NA',
                                                   'cpt':'NA'}, index=[0]))
            else:

                # grid search
                param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                clf = GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear', max_iter=200), param_grid, n_jobs=6)

                logisiticreg = clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                logit_preds = logisiticreg.predict_proba(sub_xtest)[:, 1]


                # create a vector of cc, that repeats so its the same length as the other columns in the data frame
                cc_name =np.repeat(cc, logit_preds.shape[0])

                tmp_holder = pd.DataFrame({'y_preds': list(logit_preds), 'y_values': list(sub_ytest.values), 'cpt': list(cc_name)})
                within_holder.append(pd.DataFrame({'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,
                                                   'cpt': tmp_holder.cpt}))  # LOOP THROUGH EACH CPT CODE

        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'rf_sub.csv'), index=False)

###############################################
# decompose auc and save

#read_file_1 = 'rf_sub.csv'
#res_y_all = pd.read_csv(os.path.join(dir_output, read_file_1))
res_y_all = res_y_all.dropna().reset_index(drop=True)
res_y_all['y'] = res_y_all['y'].str.strip('[]').astype(int)

result_list = []
for i in cn_Y:
    print(i)
    temp_list = []
    sub_res = res_y_all[res_y_all['outcome']==i].reset_index(drop=True)
    y_values = sub_res.y.values
    pred_values = sub_res.preds.values
    group_values = sub_res.cpt.values
    sub_auc_decomp = auc_decomp(y=y_values, score=pred_values, group=group_values, rand=False)
    temp_list.append(pd.DataFrame({'tt': sub_auc_decomp.tt, 'auc': sub_auc_decomp.auc,
                                       'den': sub_auc_decomp.den}))
    result_list.append(pd.concat(temp_list).assign(outcome=i))

sub_model_auc = pd.concat(result_list).reset_index(drop=True)
sub_model_auc.to_csv(os.path.join(dir_output, 'rf_sub_model_auc_decomposed.csv'), index=False)
