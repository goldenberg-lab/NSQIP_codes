#!/hpf/tools/centos7/python/3.7.6/bin/python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-cvalue', '--c_value', type=float, help='logit lambda', default=0)
args = parser.parse_args()
c_value= args.c_value

# random seed
random.seed(1234)

import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# DESCRIPTION: THIS SCRIPT SUBSETS BY THE MOST PREVALENT CPTS AND RUNS
# A LOGISTIC REGRESSION FOR THE AGGREGATE AND SUBMODELS AND DECOMPOSES THE AUC
# IT DOES THIS FOR BOTH THE CPT CODES AND THE CPT VALUES FROM NATIVE BAYES
# SAVES TO OUTPUT:
# --- logit_agg.csv
# --- logit_sub.csv
# --- logit_agg_phat.csv
# --- logit_sub_phat.csv
# --- logit_agg_model_auc_decomposed.csv
# --- logit_sub_model_auc_decomposed.csv
###############################
# ---- STEP 1: LOAD DATA ---- #
dir_base = dir_base = '/hpf/largeprojects/agoldenb/ben/Projects/nsqip/NSQIP_codes'
dir_output_test = os.path.join(dir_base, '..', 'logit_results/test_auc')
dir_output_validation = os.path.join(dir_base, '..', 'logit_results/validation_auc')
dir_output_sub_models = os.path.join(dir_base, '..', 'logit_results/sub_models') # here
dir_output_agg_models = os.path.join(dir_base, '..', 'logit_results/agg_models') # here
dir_data =os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_base, '..', 'figures')
fn_X = 'X_imputed.csv'
fn_Y = 'y_agg.csv'
dat_X = pd.read_csv(os.path.join(dir_data, fn_X))
dat_Y = pd.read_csv(os.path.join(dir_data, fn_Y))
# CREATE DUMMY VARIABLES FOR NON NUMERIC
#dat_X = pd.get_dummies(dat_X)
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
cn_X.append('caseid') # here
cn_Y = list(dat_Y.columns[25:37])
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
            tmp_id = Xtest.caseid

            del Xtrain['cpt']
            del Xtest['cpt']
            del Xtrain['caseid']  # here
            del Xtest['caseid']  # here

            # define the numeric variables and standard scaler
            scaler = StandardScaler()
            num_vars = list(['age_days', 'height', 'weight', 'workrvu'])
            # get cateogrical variable names and onehotencoder
            ohe = OneHotEncoder(handle_unknown='ignore')
            cat_vars = [i for i in Xtrain.columns if i not in num_vars]
            # define the preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', scaler, num_vars),
                    ('cat', ohe, cat_vars)])

            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(penalty='l2', C=c_value,solver='liblinear', max_iter=200))])
            # TRAIN MODEL WITH EACH PARAMETER
            logit_mod = clf.fit(Xtrain, ytrain.values.ravel())
            logit_preds = logit_mod.predict_proba(Xtest)[:, 1]
            auc_score=np.nan
            model_file_name = os.path.join(dir_output_agg_models, 'logit_agg_' + str(c_value) + '_' + str(vv) + '_' + str(yy) + '.sav')
            pickle.dump(logit_mod, open(model_file_name, 'wb'))
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
            tmp_id = Xtest.caseid
            del Xtrain['cpt']
            del Xtest['cpt']
            del Xvalid['cpt']
            del Xtrain['caseid']
            del Xtest['caseid']
            del Xvalid['caseid']
            # define the numeric variables and standard scaler
            scaler = StandardScaler()
            num_vars = list(['age_days', 'height', 'weight', 'workrvu'])
            # get cateogrical variable names and onehotencoder
            ohe = OneHotEncoder(handle_unknown='ignore')
            cat_vars = [i for i in Xtrain.columns if i not in num_vars]
            # define the preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', scaler, num_vars),
                    ('cat', ohe, cat_vars)])

            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier',
                                   LogisticRegression(penalty='l2', C=c_value, solver='liblinear', max_iter=200))])
            logit_mod = clf.fit(Xtrain, ytrain.values.ravel())
            logit_preds = logit_mod.predict_proba(Xvalid)[:, 1]
            auc_score = metrics.roc_auc_score(yvalid, logit_preds)
            #COMBINE THE TRAIN AND VALIDATOIN SETS AND RETRAIN MODEL ON ALL DATA WITH THE BEST C VALUES
            Xtrain = pd.concat([Xtrain, Xvalid])
            ytrain = pd.concat([ytrain, yvalid])
            logit_mod = clf.fit(Xtrain, ytrain.values.ravel())
            logit_preds = logit_mod.predict_proba(Xtest)[:, 1]
            model_file_name = os.path.join(dir_output_agg_models, 'logit_agg_' + str(c_value) + '_' + str(vv) + '_' + str(yy) + '.sav')
            pickle.dump(logit_mod, open(model_file_name, 'wb'))
            if yy == 2018:
                # combine all years in to one dataset
                Xtrain = pd.concat([Xtrain, Xtest])
                ytrain = pd.concat([ytrain, ytest])
                logit_mod = clf.fit(Xtrain, ytrain.values.ravel())
                model_file_name = os.path.join(dir_output_agg_models, 'logit_agg_final_' + str(c_value) + '_' + str(vv) + '.sav')
                pickle.dump(logit_mod, open(model_file_name, 'wb'))
        # STORE RESULTS FROM AGGREGATE MODEL
        within_holder = []
        valid_holder =[]
        tmp_holder_valid = pd.DataFrame({'c': c_value, 'auc': auc_score}, index=[0])
        valid_holder.append(pd.DataFrame({'c':tmp_holder_valid.c.values, 'auc_valid':tmp_holder_valid.auc.values}))
        tmp_holder = pd.DataFrame(
            {'caseid': list(tmp_id), 'y_preds': list(logit_preds), 'y_values': np.array(ytest).ravel(),
             'cpt': list(tmp_cpt)})
        within_holder.append(pd.DataFrame(
            {'caseid': tmp_holder.caseid, 'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,
             'cpt': tmp_holder.cpt}))
        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
        holder_y_valid.append(pd.concat(valid_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))
    holder_y_all_valid.append(pd.concat(holder_y_valid).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all_valid = pd.concat(holder_y_all_valid).reset_index(drop=True)

res_y_all.to_csv(os.path.join(dir_output_test, 'logit_agg_'+str(c_value)+'.csv'), index=False)
res_y_all_valid.to_csv(os.path.join(dir_output_validation, 'logit_agg_valid_'+str(c_value)+'.csv'), index=False)

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
        tmp_id = Xtest.caseid.to_frame().join(Xtest.cpt)
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
            tmp_id_sub = tmp_id[tmp_id['cpt'] == cc]
            caseids = tmp_id_sub.caseid
            if yy==2013:
                # conditon by year here.
                # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
                if all(np.unique(sub_ytrain.values) == 0) or all(np.unique(sub_ytest.values) == 0):
                    within_holder.append(pd.DataFrame({'caseid':np.nan,
                                                       'y': np.nan,
                                                       'preds': np.nan,
                                                       'cpt': np.nan}, index=[0]))
                else:
                    # define the numeric variables and standard scaler
                    scaler = StandardScaler()
                    num_vars = list(['age_days', 'height', 'weight', 'workrvu'])
                    # get cateogrical variable names and onehotencoder
                    ohe = OneHotEncoder(handle_unknown='ignore')
                    cat_vars = [i for i in sub_xtrain.columns if i not in num_vars]
                    # define the preprocessor
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', scaler, num_vars),
                            ('cat', ohe, cat_vars)])

                    clf = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier',
                                           LogisticRegression(penalty='l2', C=c_value, solver='liblinear',
                                                              max_iter=200))])
                    logit_mod = clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                    logit_preds = logit_mod.predict_proba(sub_xtest)[:, 1]
                    cc_name = np.repeat(cc, logit_preds.shape[0])
                    model_file_name = os.path.join(dir_output_sub_models,
                                                   'logit_sub_' + str(c_value) + '_' + str(vv) + '_' + str(yy) + '_' + str(cc) + '.sav')
                    pickle.dump(logit_mod, open(model_file_name, 'wb'))
                    tmp_holder = pd.DataFrame(
                        {'caseid': list(caseids), 'y_preds': list(logit_preds), 'y_values': np.array(sub_ytest).ravel(),
                         'cpt': list(cc_name)})
                    within_holder.append(pd.DataFrame(
                        {'caseid': tmp_holder.caseid, 'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,
                         'cpt': tmp_holder.cpt}))  # LOOP THROUGH EACH CPT CODE
                    tmp_holder_valid = pd.DataFrame({'c': c_value, 'auc': np.nan, 'cpt':cc}, index=[0])
                    valid_holder.append( pd.DataFrame({'c': tmp_holder_valid.c.values, 'auc_valid': tmp_holder_valid.auc.values, 'cpt':tmp_holder_valid.cpt}))
            else:
                sub_xvalid = Xvalid[Xvalid['cpt'] == cc]
                sub_yvalid = yvalid[yvalid.index.isin(sub_xvalid.index)]
                del sub_xvalid['cpt']
                # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
                if all(np.unique(sub_ytrain.values) == 0) or all(np.unique(sub_ytest.values) == 0) or all(np.unique(sub_yvalid.values) == 0):
                    within_holder.append(pd.DataFrame({'caseid':np.nan,'y': np.nan,
                                                       'preds': np.nan,
                                                       'cpt': np.nan}, index=[0]))
                else:
                    # define the numeric variables and standard scaler
                    scaler = StandardScaler()
                    num_vars = list(['age_days', 'height', 'weight', 'workrvu'])
                    # get cateogrical variable names and onehotencoder
                    ohe = OneHotEncoder(handle_unknown='ignore')
                    cat_vars = [i for i in sub_xtrain.columns if i not in num_vars]
                    # define the preprocessor
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', scaler, num_vars),
                            ('cat', ohe, cat_vars)])

                    clf = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier',
                                           LogisticRegression(penalty='l2', C=c_value, solver='liblinear',
                                                              max_iter=200))])
                    logit_mod= clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                    logit_preds = logit_mod.predict_proba(sub_xvalid)[:, 1]
                    auc_score = metrics.roc_auc_score(sub_yvalid, logit_preds)

                    # COMBINE THE TRAIN AND VALIDATOIN SETS AND RETRAIN MODEL ON ALL DATA WITH THE BEST C VALUES
                    sub_xtrain = pd.concat([sub_xtrain, sub_xvalid])
                    sub_ytrain = pd.concat([sub_ytrain, sub_yvalid])
                    logit_mod = clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                    logit_preds = logit_mod.predict_proba(sub_xtest)[:, 1]

                    # create a vector of cc, that repeats so its the same length as the other columns in the data frame
                    cc_name = np.repeat(cc, logit_preds.shape[0])
                    model_file_name = os.path.join(dir_output_sub_models,
                                                   'logit_sub_' + str(c_value) + '_' + str(vv) + '_' + str(yy) + '_' + str(cc) + '.sav')
                    pickle.dump(logit_mod, open(model_file_name, 'wb'))
                    tmp_holder_valid = pd.DataFrame({'c': c_value, 'auc': auc_score, 'cpt':cc}, index=[0])
                    valid_holder.append(pd.DataFrame({'c': tmp_holder_valid.c.values, 'auc_valid': tmp_holder_valid.auc.values, 'cpt':tmp_holder_valid.cpt}))
                    tmp_holder = pd.DataFrame(
                        {'caseid': list(caseids), 'y_preds': list(logit_preds), 'y_values': np.array(sub_ytest).ravel(),
                         'cpt': list(cc_name)})
                    within_holder.append(pd.DataFrame(
                        {'caseid': tmp_holder.caseid, 'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,
                         'cpt': tmp_holder.cpt}))  # LOOP THROUGH EACH CPT CODE
                    # get full model
                    if yy == 2018:
                        # combine all years in to one dataset
                        sub_xtrain = pd.concat([sub_xtrain, sub_xtest])
                        sub_ytrain = pd.concat([sub_ytrain, sub_ytest])
                        xgb_mod_full = clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                        model_file_name = os.path.join(dir_output_sub_models,
                                                       'logit_sub_final_' + str(c_value) + '_' + str(vv) + '_' + str(cc) + '.sav')
                        pickle.dump(xgb_mod_full, open(model_file_name, 'wb'))
        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
        holder_y_valid.append(pd.concat(valid_holder).assign(test_year=yy))

    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))
    holder_y_all_valid.append(pd.concat(holder_y_valid).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all_valid = pd.concat(holder_y_all_valid).reset_index(drop=True)

res_y_all.to_csv(os.path.join(dir_output_test, 'logit_sub_'+str(c_value)+'.csv'), index=False)
res_y_all_valid.to_csv(os.path.join(dir_output_validation, 'logit_sub_valid_'+str(c_value)+'.csv'), index=False)

# ###############################################
# # ---- STEP 4: LEAVE-ONE-YEAR - ALL VARIABLES (RISK SCORE INSTEAD OF CPT SCORE) ---- #
# #READ IN RISK SCORES
# file_name = 'nbayes_phat.csv'
# nb_phat = pd.read_csv(os.path.join(dir_output, file_name))
# # REMOVE Y COLUMN
# del nb_phat['y']
# # ADD VARIABLE NAME "PHAT"
# cn_X.append('phat')
# holder_y_all = []
# for ii, vv in enumerate(cn_Y):
#     print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))
#     # SUBSET NB_PHAT BY OUTCOME
#     tmp_phat = nb_phat[nb_phat['outcome']==vv].reset_index(drop=False)
#     tmp_phat_years = tmp_phat.operyr.unique()
#     # REMOVE OPERYR AND
#     del tmp_phat['operyr']
#     del tmp_phat['outcome']
#     tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv] == -1], axis=1)
#     tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
#     tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
#     tmp_years = tmp_years.astype(int)
#     tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]
#     # GET TRAINING YEARS - 2012 DOESNT HAVE PHAT VALUES
#     tmp_train_years = np.intersect1d(tmp_train_years, tmp_phat_years)
#     tmp_train_years = tmp_train_years[tmp_train_years > tmp_phat_years.min()]
#     # JOIN DATA AND PHAT DATA
#     sub_x = pd.merge(dat_X, tmp_phat, on = 'caseid')
#     # SUBSET DAT_Y BY THE SAME INDEX
#     sub_y= dat_Y[dat_Y.index.isin(sub_x.index)]
#     holder_y = []
#     for yy in tmp_train_years:
#         print('Train Year %i' % (yy))
#         idx_train = sub_x.operyr.isin(tmp_years) & (sub_x.operyr < yy)
#         idx_test = sub_x.operyr.isin(tmp_years) & (sub_x.operyr == yy)
#         Xtrain, Xtest = sub_x.loc[idx_train, cn_X].reset_index(drop=True), \
#                         sub_x.loc[idx_test, cn_X].reset_index(drop=True)
#         ytrain, ytest = sub_y.loc[idx_train, [vv]].reset_index(drop=True), \
#                         sub_y.loc[idx_test, [vv]].reset_index(drop=True)
#         # STORE CPT CODE
#         tmp_cpt = Xtest.cpt
#         del Xtrain['cpt']
#         del Xtest['cpt']
#         # TRAIN MODEL
#         logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
#         logit_fit = logisticreg.fit(Xtrain, ytrain.values.ravel())
#         # GET PREDICTIONS
#         logit_preds = logit_fit.predict_proba(Xtest)[:, 1]
#         tmp_holder = pd.DataFrame(
#             {'y_preds': list(logit_preds), 'y_values': list(ytest.values), 'cpt': list(tmp_cpt)})
#         within_holder = []
#         for cc in top_cpts:
#             #print('cpt %s' % (cc))
#             sub_tmp_holder = tmp_holder[tmp_holder['cpt'] == cc].reset_index(drop=True)
#             # FILL RESULTS LIST WITH NA IF ONLY TOW LEVELS OR NEGATIVE 1 IN OUTCOME
#             if all(sub_tmp_holder.y_values.values == 0) or any(sub_tmp_holder.y_values.values < 0):
#                 within_holder.append(pd.DataFrame({'auc': 'NA',
#                                                    'cpt': cc}, index=[0]))
#             else:
#                 within_holder.append(pd.DataFrame({'auc': metrics.roc_auc_score(list(sub_tmp_holder.y_values.values),
#                                                                                 list(sub_tmp_holder.y_preds.values)),
#                                                    'cpt': cc}, index=[0]))
#         holder_y.append(pd.concat(within_holder).assign(test_year=yy))
#     holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))
# res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
# res_y_all.to_csv(os.path.join(dir_output, 'logit_agg_phat.csv'), index=False)
# ###############################################
# # ---- STEP 5: LEAVE-ONE-YEAR - ALL VARIABLES (RISK SCORE INSTEAD OF CPT SCORE, SUB MODELS) ---- #
# holder_y_all = []
# for ii, vv in enumerate(cn_Y):
#     print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))
#     tmp_phat = nb_phat[nb_phat['outcome']==vv].reset_index(drop=False)
#     tmp_phat_years = tmp_phat.operyr.unique()
#     del tmp_phat['operyr']
#     del tmp_phat['outcome']
#     tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv] == -1], axis=1)
#     tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
#     tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
#     tmp_years = tmp_years.astype(int)
#     tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]
#     tmp_train_years = np.intersect1d(tmp_train_years, tmp_phat_years)
#     tmp_train_years = tmp_train_years[tmp_train_years > tmp_phat_years.min()]
#     # JOIN DATA TO GET PHAT VALUES
#     sub_x = pd.merge(dat_X, tmp_phat, on = 'caseid')
#     # SUBSET Y DATA BY SAME INDEX
#     sub_y= dat_Y[dat_Y.index.isin(sub_x.index)]
#     holder_y = []
#     for yy in tmp_train_years:
#         print('Train Year %i' % (yy))
#         idx_train = sub_x.operyr.isin(tmp_years) & (sub_x.operyr < yy)
#         idx_test = sub_x.operyr.isin(tmp_years) & (sub_x.operyr == yy)
#         Xtrain, Xtest = sub_x.loc[idx_train, cn_X].reset_index(drop=True), \
#                         sub_x.loc[idx_test, cn_X].reset_index(drop=True)
#         ytrain, ytest = sub_y.loc[idx_train, [vv]].reset_index(drop=True), \
#                         sub_y.loc[idx_test, [vv]].reset_index(drop=True)
#         within_holder = []
#         for cc in top_cpts:
#             # SUBSET XTRAIN AND XTEST BY CPT CODE
#             sub_xtrain = Xtrain[Xtrain['cpt'] == cc]
#             sub_xtest = Xtest[Xtest['cpt'] == cc]
#             # SUBSET YTRAIN AND YTEST BY THE CORRESPONDING INDICES IN SUBSETTED XDATA
#             sub_ytrain = ytrain[ytrain.index.isin(sub_xtrain.index)]
#             sub_ytest = ytest[ytest.index.isin(sub_xtest.index)]
#             # REMOVE CPT COLUMN
#             del sub_xtrain['cpt']
#             del sub_xtest['cpt']
#             # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE OR CONTAINS NEGATIVE NUMBER
#             if any(np.unique(sub_ytrain.values) < 0) or all(np.unique(sub_ytrain.values) == 0) or any(np.unique(sub_ytest.values) < 0) or all(np.unique(sub_ytest.values) == 0) or len(sub_ytrain.values) == 0 or len(sub_ytest.values) == 0:
#                 within_holder.append(pd.DataFrame({'auc': 'NA',
#                                                    'cpt': cc}, index=[0]))
#             else:
#                 # TRAIN MODEL
#                 logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
#                 logit_fit = logisticreg.fit(sub_xtrain, sub_ytrain.values.ravel())
#                 # GET PREDICTIONS
#                 logit_preds = logit_fit.predict_proba(sub_xtest)[:, 1]
#                 within_holder.append(
#                     pd.DataFrame({'auc': metrics.roc_auc_score(sub_ytest.values, logit_preds), 'cpt': cc}, index=[0]))
#         holder_y.append(pd.concat(within_holder).assign(test_year=yy))
#     holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))
# res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
# res_y_all.to_csv(os.path.join(dir_output, 'logit_sub_phat.csv'), index=False)
