#!/hpf/tools/centos6/python/3.7.6_benbrew/bin/python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-mdepth', '--m_depth', type=int, help='max_depth_rf', default=2)
parser.add_argument('-nest', '--n_est', type=float, help='column sample by tree', default=0.3)

args = parser.parse_args()
m_depth= args.m_depth
n_est = args.n_est
n_est = 50
m_depth = 4

import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

###############################
# ---- STEP 1: LOAD DATA ---- #
dir_base = '/hpf/largeprojects/agoldenb/ben/Projects/nsqip/NSQIP_codes'
dir_base = os.getcwd()
dir_output_test = os.path.join(dir_base, '..', 'rf_results/test_auc')
dir_output_validation = os.path.join(dir_base, '..', 'rf_results/validation_auc')
dir_data =os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_base, '..', 'figures')
fn_X = 'X_imputed.csv'
fn_Y = 'y_agg.csv'
dat_X = pd.read_csv(os.path.join(dir_data, fn_X))
dat_Y = pd.read_csv(os.path.join(dir_data, fn_Y))
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
cn_Y = list(dat_Y.columns[36:37])
# DELETE NON AGG LABELS
dat_Y.drop(dat_Y.columns[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
           axis=1, inplace=True)
###############################################
# ---- STEP 2: LEAVE-ONE-YEAR - ALL VARIABLES  ---- #
holder_y_all = []
holder_y_all_valid=[]
for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))
    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]
    holder_y = []
    holder_y_valid = []
    for yy in tmp_train_years:
        # FOR 2013 WE DONT HAVE A VALIDATION SET TO TUNE HYPERPARAMETERS, SO USE NORMAL TRAIN, TEST SPLIT
        if yy == 2013:
            print('Train Year %i' % (yy))
            idx_train = dat_X.operyr.isin(tmp_years) & (dat_X.operyr < yy)
            idx_test = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy)
            Xtrain, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                            dat_X.loc[idx_test, cn_X].reset_index(drop=True)
            ytrain, ytest = dat_Y.loc[idx_train, [vv]].reset_index(drop=True), \
                            dat_Y.loc[idx_test, [vv]].reset_index(drop=True)
            # STORE CPT CODES AND DELETE FROM DATA
            tmp_cpt = Xtest.cpt
            del Xtrain['cpt']
            del Xtest['cpt']
            # TRAIN MODEL WITH EACH PARAMETER
            clf = RandomForestClassifier(bootstrap=True, max_depth=m_depth, n_estimators=n_est)
            rf_mod = clf.fit(Xtrain, ytrain.values.ravel())
            rf_preds = rf_mod.predict_proba(Xtest)[:, 1]
            auc_score=np.nan
        else:
            # FOR YEARS 2014-2018 WE HAVE A TRAIN, VALIDATION, AND TEST SET
            print('Train Year %i' % (yy))
            # get validation year
            yy_valid = yy-1
            idx_train = dat_X.operyr.isin(tmp_years) & (dat_X.operyr < yy_valid)
            idx_valid = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy_valid)
            idx_test = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy)
            Xtrain, Xvalid, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                                    dat_X.loc[idx_valid, cn_X].reset_index(drop=True), \
                                    dat_X.loc[idx_test, cn_X].reset_index(drop=True)
            ytrain, yvalid, ytest = dat_Y.loc[idx_train, [vv]].reset_index(drop=True), \
                                    dat_Y.loc[idx_valid, [vv]].reset_index(drop=True), \
                                    dat_Y.loc[idx_test, [vv]].reset_index(drop=True)
            # STORE CPT CODES AND DELETE FROM DATA
            tmp_cpt = Xtest.cpt
            del Xtrain['cpt']
            del Xtest['cpt']
            del Xvalid['cpt']
            # TRAIN A MODEL WITH EACH C VALUE AND TEST ON THE VALIDATION SET AND RETRIEVE BEST C VALUES
            clf = RandomForestClassifier(bootstrap=True, max_depth=m_depth, n_estimators=n_est)
            rf_mod = clf.fit(Xtrain, ytrain.values.ravel())
            rf_preds = rf_mod.predict_proba(Xvalid)[:, 1]
            auc_score = metrics.roc_auc_score(yvalid, rf_preds)
            # USE BEST C VALUE FROM LOOP
            clf = RandomForestClassifier(bootstrap=True, max_depth=m_depth, n_estimators=n_est)
            #COMBINE THE TRAIN AND VALIDATOIN SETS AND RETRAIN MODEL ON ALL DATA WITH THE BEST C VALUES
            Xtrain = pd.concat([Xtrain, Xvalid])
            ytrain = pd.concat([ytrain, yvalid])
            rf_mod = clf.fit(Xtrain, ytrain.values.ravel())
            rf_preds = rf_mod.predict_proba(Xtest)[:, 1]
        # STORE RESULTS FROM AGGREGATE MODEL
        within_holder = []
        valid_holder =[]
        tmp_holder_valid = pd.DataFrame({'m_depth': m_depth,'n_est': n_est ,'auc': auc_score}, index=[0])
        tmp_holder = pd.DataFrame({'y_preds': list(rf_preds), 'y_values': np.array(ytest).ravel(), 'cpt': list(tmp_cpt)})
        valid_holder.append(pd.DataFrame({'m_depth':tmp_holder_valid.m_depth.values,'n_est':tmp_holder_valid.n_est.values, 'auc_valid':tmp_holder_valid.auc.values}))
        within_holder.append(pd.DataFrame({'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,'cpt': tmp_holder.cpt}))        # LOOP THROUGH EACH CPT CODE
        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
        holder_y_valid.append(pd.concat(valid_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))
    holder_y_all_valid.append(pd.concat(holder_y_valid).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all_valid = pd.concat(holder_y_all_valid).reset_index(drop=True)

res_y_all.to_csv(os.path.join(dir_output_test, 'rf_agg_'+str(m_depth)+str(n_est)+'.csv'), index=False)
res_y_all_valid.to_csv(os.path.join(dir_output_validation, 'rf_agg_valid_'+str(m_depth)+str(n_est)+'.csv'), index=False)

####################################################
# ---- STEP 3: LEAVE-ONE-YEAR - ALL VARIABLES, FOR EACH CPT CODE, SUB MODELS---- #
holder_y_all = []
holder_y_all_valid = []
for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))
    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]
    holder_y = []
    holder_y_valid = []
    for yy in tmp_train_years:
        if yy ==2013:
            print('Train Year %i' % (yy))
            idx_train = dat_X.operyr.isin(tmp_years) & (dat_X.operyr < yy)
            idx_test = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy)
            Xtrain, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                            dat_X.loc[idx_test, cn_X].reset_index(drop=True)
            ytrain, ytest = dat_Y.loc[idx_train, [vv]].reset_index(drop=True), \
                            dat_Y.loc[idx_test, [vv]].reset_index(drop=True)
        else:
            print('Train Year %i' % (yy))
            # get validation year
            yy_valid = yy - 1
            idx_train = dat_X.operyr.isin(tmp_years) & (dat_X.operyr < yy_valid)
            idx_valid = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy_valid)
            idx_test = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy)
            Xtrain, Xvalid, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                                    dat_X.loc[idx_valid, cn_X].reset_index(drop=True), \
                                    dat_X.loc[idx_test, cn_X].reset_index(drop=True)
            ytrain, yvalid, ytest = dat_Y.loc[idx_train, [vv]].reset_index(drop=True), \
                                    dat_Y.loc[idx_valid, [vv]].reset_index(drop=True), \
                                    dat_Y.loc[idx_test, [vv]].reset_index(drop=True)
        within_holder = []
        valid_holder = []
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
            if yy==2013:
                # conditon by year here.
                # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
                if all(np.unique(sub_ytrain.values) == 0) or all(np.unique(sub_ytest.values) == 0):
                    within_holder.append(pd.DataFrame({'y': np.nan,
                                                       'preds': np.nan,
                                                       'cpt': np.nan}, index=[0]))
                else:
                    # grid search
                    clf = RandomForestClassifier(bootstrap=True, max_depth=m_depth, n_estimators=n_est)
                    rf_mod = clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                    rf_preds = rf_mod.predict_proba(sub_xtest)[:, 1]
                    cc_name = np.repeat(cc, rf_preds.shape[0])
                    tmp_holder = pd.DataFrame({'y_preds': list(rf_preds), 'y_values': np.array(sub_ytest).ravel(), 'cpt': list(cc_name)})
                    within_holder.append(pd.DataFrame({'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,
                                                       'cpt': tmp_holder.cpt}))  # LOOP THROUGH EACH CPT CODE

                    tmp_holder_valid = pd.DataFrame({'m_depth': m_depth, 'n_est': n_est, 'auc': np.nan,'cpt':cc }, index=[0])
                    valid_holder.append( pd.DataFrame({'m_depth': tmp_holder_valid.m_depth.values,'n_est':tmp_holder_valid.n_est.values ,
                                                       'auc_valid': tmp_holder_valid.auc.values, 'cpt':tmp_holder_valid.cpt}))
            else:
                sub_xvalid = Xvalid[Xvalid['cpt'] == cc]
                sub_yvalid = yvalid[yvalid.index.isin(sub_xvalid.index)]
                del sub_xvalid['cpt']
                # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
                if all(np.unique(sub_ytrain.values) == 0) or all(np.unique(sub_ytest.values) == 0) or all(np.unique(sub_yvalid.values) == 0):
                    within_holder.append(pd.DataFrame({'y': np.nan,
                                                       'preds': np.nan,
                                                       'cpt': np.nan}, index=[0]))
                else:
                    # TRAIN A MODEL WITH EACH C VALUE AND TEST ON THE VALIDATION SET AND RETRIEVE BEST C VALUES
                    clf = RandomForestClassifier(bootstrap=True, max_depth=m_depth, n_estimators=n_est)
                    rf_mod = clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                    rf_preds = rf_mod.predict_proba(sub_xvalid)[:, 1]
                    auc_score = metrics.roc_auc_score(sub_yvalid, rf_preds)

                    # USE BEST C VALUE FROM LOOP
                    clf = RandomForestClassifier(bootstrap=True, max_depth=m_depth, n_estimators=n_est)
                    # COMBINE THE TRAIN AND VALIDATOIN SETS AND RETRAIN MODEL ON ALL DATA WITH THE BEST C VALUES
                    sub_xtrain = pd.concat([sub_xtrain, sub_xvalid])
                    sub_ytrain = pd.concat([sub_ytrain, sub_yvalid])
                    rf_mod = clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                    rf_preds = rf_mod.predict_proba(sub_xtest)[:, 1]

                    # create a vector of cc, that repeats so its the same length as the other columns in the data frame
                    cc_name = np.repeat(cc, rf_preds.shape[0])
                    tmp_holder_valid = pd.DataFrame({'m_depth': m_depth,'n_est':n_est, 'auc': auc_score, 'cpt':cc}, index=[0])
                    valid_holder.append(pd.DataFrame({'m_depth': tmp_holder_valid.m_depth.values,'n_est':tmp_holder_valid.n_est.values ,'auc_valid': tmp_holder_valid.auc.values, 'cpt':tmp_holder_valid.cpt}))
                    tmp_holder = pd.DataFrame( {'y_preds': list(rf_preds), 'y_values': np.array(sub_ytest).ravel(), 'cpt': list(cc_name)})
                    within_holder.append(pd.DataFrame({'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,
                                                       'cpt': tmp_holder.cpt}))  # LOOP THROUGH EACH CPT CODE
        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
        holder_y_valid.append(pd.concat(valid_holder).assign(test_year=yy))

    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))
    holder_y_all_valid.append(pd.concat(holder_y_valid).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all_valid = pd.concat(holder_y_all_valid).reset_index(drop=True)

res_y_all.to_csv(os.path.join(dir_output_test, 'rf_sub_'+str(m_depth)+str(n_est)+'.csv'), index=False)
res_y_all_valid.to_csv(os.path.join(dir_output_validation, 'rf_sub_valid_'+str(m_depth)+str(n_est)+'.csv'), index=False)
