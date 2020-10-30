import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from support.acc_funs import auc_decomp
from sklearn.model_selection import GridSearchCV

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
            param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            clf = GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear', max_iter=200), param_grid, n_jobs=6,
                               cv=2)

            logisiticreg = clf.fit(Xtrain, ytrain.values.ravel())
            logit_preds = logisiticreg.predict_proba(Xtest)[:, 1]
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
            c_value = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            best_c = []
            for c in c_value:
                clf = LogisticRegression(penalty='l2', C=c,solver='liblinear', max_iter=200)
                logisiticreg = clf.fit(Xtrain, ytrain.values.ravel())
                logit_preds = logisiticreg.predict_proba(Xvalid)[:, 1]
                auc_score = metrics.roc_auc_score(yvalid, logit_preds)
                best_c.append(pd.DataFrame({'c':c, 'auc':auc_score}, index=[0]))
            best_c = pd.concat(best_c)
            best_c = best_c[best_c['auc'] == max(best_c.auc)].c.values

            # USE BEST C VALUE FROM LOOP
            clf = LogisticRegression(penalty='l2', C=float(best_c), solver='liblinear', max_iter=200)
            #COMBINE THE TRAIN AND VALIDATOIN SETS AND RETRAIN MODEL ON ALL DATA WITH THE BEST C VALUES
            Xtrain = pd.concat([Xtrain, Xvalid])
            ytrain = pd.concat([ytrain, yvalid])
            logisiticreg = clf.fit(Xtrain, ytrain.values.ravel())
            logit_preds = logisiticreg.predict_proba(Xtest)[:, 1]

        # STORE RESULTS FROM AGGREGATE MODEL
        within_holder = []
        tmp_holder = pd.DataFrame({'y_preds': list(logit_preds), 'y_values': np.array(ytest).ravel(), 'cpt': list(tmp_cpt)})
        within_holder.append(pd.DataFrame({'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,'cpt': tmp_holder.cpt}))        # LOOP THROUGH EACH CPT CODE

        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'logit_agg.csv'), index=False)

###############################################
# decompose auc and save
#read_file_1 = 'logit_agg.csv'
#res_y_all = pd.read_csv(os.path.join(dir_output, read_file_1))
res_y_all = res_y_all.dropna().reset_index(drop=True)

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
agg_model_auc.to_csv(os.path.join(dir_output, 'logit_agg_model_auc_decomposed.csv'), index=False)

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
                    clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=200)
                    logisiticreg = clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                    logit_preds = logisiticreg.predict_proba(sub_xtest)[:, 1]
                    cc_name = np.repeat(cc, logit_preds.shape[0])
                    tmp_holder = pd.DataFrame(
                        {'y_preds': list(logit_preds), 'y_values': np.array(sub_ytest).ravel(), 'cpt': list(cc_name)})
                    within_holder.append(pd.DataFrame({'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,
                                                       'cpt': tmp_holder.cpt}))  # LOOP THROUGH EACH CPT CODE

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
                    c_value = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                    best_c = []
                    for c in c_value:
                        clf = LogisticRegression(penalty='l2', C=c, solver='liblinear', max_iter=200)
                        logisiticreg = clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                        logit_preds = logisiticreg.predict_proba(sub_xvalid)[:, 1]
                        auc_score = metrics.roc_auc_score(sub_yvalid, logit_preds)
                        best_c.append(pd.DataFrame({'c': c, 'auc': auc_score}, index=[0]))
                    best_c = pd.concat(best_c)
                    best_c = best_c[best_c['auc'] == max(best_c.auc)].c.values[0]


                    # USE BEST C VALUE FROM LOOP
                    clf = LogisticRegression(penalty='l2', C=float(best_c), solver='liblinear', max_iter=500)
                    # COMBINE THE TRAIN AND VALIDATOIN SETS AND RETRAIN MODEL ON ALL DATA WITH THE BEST C VALUES
                    sub_xtrain = pd.concat([sub_xtrain, sub_xvalid])
                    sub_ytrain = pd.concat([sub_ytrain, sub_yvalid])
                    logisiticreg = clf.fit(sub_xtrain, sub_ytrain.values.ravel())
                    logit_preds = logisiticreg.predict_proba(sub_xtest)[:, 1]
                    # create a vector of cc, that repeats so its the same length as the other columns in the data frame
                    cc_name = np.repeat(cc, logit_preds.shape[0])
                    tmp_holder = pd.DataFrame(
                        {'y_preds': list(logit_preds), 'y_values': np.array(sub_ytest).ravel(), 'cpt': list(cc_name)})
                    within_holder.append(pd.DataFrame({'y': tmp_holder.y_values, 'preds': tmp_holder.y_preds,
                                                       'cpt': tmp_holder.cpt}))  # LOOP THROUGH EACH CPT CODE
        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'logit_sub.csv'), index=False)


###############################################
# decompose auc and save

#read_file_1 = 'logit_sub.csv'
#res_y_all = pd.read_csv(os.path.join(dir_output, read_file_1))
res_y_all = res_y_all.dropna().reset_index(drop=True)

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
sub_model_auc.to_csv(os.path.join(dir_output, 'logit_sub_model_auc_decomposed.csv'), index=False)


###############################################
# ---- STEP 4: LEAVE-ONE-YEAR - ALL VARIABLES (RISK SCORE INSTEAD OF CPT SCORE) ---- #

#READ IN RISK SCORES
file_name = 'nbayes_phat.csv'
nb_phat = pd.read_csv(os.path.join(dir_output, file_name))

# REMOVE Y COLUMN
del nb_phat['y']

# ADD VARIABLE NAME "PHAT"
cn_X.append('phat')

holder_y_all = []
for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))
    # SUBSET NB_PHAT BY OUTCOME
    tmp_phat = nb_phat[nb_phat['outcome']==vv].reset_index(drop=False)
    tmp_phat_years = tmp_phat.operyr.unique()
    # REMOVE OPERYR AND
    del tmp_phat['operyr']
    del tmp_phat['outcome']

    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]

    # GET TRAINING YEARS - 2012 DOESNT HAVE PHAT VALUES
    tmp_train_years = np.intersect1d(tmp_train_years, tmp_phat_years)
    tmp_train_years = tmp_train_years[tmp_train_years > tmp_phat_years.min()]

    # JOIN DATA AND PHAT DATA
    sub_x = pd.merge(dat_X, tmp_phat, on = 'caseid')

    # SUBSET DAT_Y BY THE SAME INDEX
    sub_y= dat_Y[dat_Y.index.isin(sub_x.index)]

    holder_y = []
    for yy in tmp_train_years:
        print('Train Year %i' % (yy))
        idx_train = sub_x.operyr.isin(tmp_years) & (sub_x.operyr < yy)
        idx_test = sub_x.operyr.isin(tmp_years) & (sub_x.operyr == yy)
        Xtrain, Xtest = sub_x.loc[idx_train, cn_X].reset_index(drop=True), \
                        sub_x.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = sub_y.loc[idx_train, [vv]].reset_index(drop=True), \
                        sub_y.loc[idx_test, [vv]].reset_index(drop=True)

        # STORE CPT CODE
        tmp_cpt = Xtest.cpt
        del Xtrain['cpt']
        del Xtest['cpt']

        # TRAIN MODEL
        logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
        logit_fit = logisticreg.fit(Xtrain, ytrain.values.ravel())

        # GET PREDICTIONS
        logit_preds = logit_fit.predict_proba(Xtest)[:, 1]

        tmp_holder = pd.DataFrame(
            {'y_preds': list(logit_preds), 'y_values': list(ytest.values), 'cpt': list(tmp_cpt)})

        within_holder = []
        for cc in top_cpts:
            #print('cpt %s' % (cc))
            sub_tmp_holder = tmp_holder[tmp_holder['cpt'] == cc].reset_index(drop=True)
            # FILL RESULTS LIST WITH NA IF ONLY TOW LEVELS OR NEGATIVE 1 IN OUTCOME
            if all(sub_tmp_holder.y_values.values == 0) or any(sub_tmp_holder.y_values.values < 0):
                within_holder.append(pd.DataFrame({'auc': 'NA',
                                                   'cpt': cc}, index=[0]))
            else:
                within_holder.append(pd.DataFrame({'auc': metrics.roc_auc_score(list(sub_tmp_holder.y_values.values),
                                                                                list(sub_tmp_holder.y_preds.values)),
                                                   'cpt': cc}, index=[0]))

        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'logit_agg_phat.csv'), index=False)

###############################################
# ---- STEP 5: LEAVE-ONE-YEAR - ALL VARIABLES (RISK SCORE INSTEAD OF CPT SCORE, SUB MODELS) ---- #

holder_y_all = []

for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))
    tmp_phat = nb_phat[nb_phat['outcome']==vv].reset_index(drop=False)
    tmp_phat_years = tmp_phat.operyr.unique()
    del tmp_phat['operyr']
    del tmp_phat['outcome']

    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]
    tmp_train_years = np.intersect1d(tmp_train_years, tmp_phat_years)
    tmp_train_years = tmp_train_years[tmp_train_years > tmp_phat_years.min()]

    # JOIN DATA TO GET PHAT VALUES
    sub_x = pd.merge(dat_X, tmp_phat, on = 'caseid')

    # SUBSET Y DATA BY SAME INDEX
    sub_y= dat_Y[dat_Y.index.isin(sub_x.index)]

    holder_y = []
    for yy in tmp_train_years:
        print('Train Year %i' % (yy))
        idx_train = sub_x.operyr.isin(tmp_years) & (sub_x.operyr < yy)
        idx_test = sub_x.operyr.isin(tmp_years) & (sub_x.operyr == yy)
        Xtrain, Xtest = sub_x.loc[idx_train, cn_X].reset_index(drop=True), \
                        sub_x.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = sub_y.loc[idx_train, [vv]].reset_index(drop=True), \
                        sub_y.loc[idx_test, [vv]].reset_index(drop=True)
        within_holder = []
        for cc in top_cpts:
            # SUBSET XTRAIN AND XTEST BY CPT CODE
            sub_xtrain = Xtrain[Xtrain['cpt'] == cc]
            sub_xtest = Xtest[Xtest['cpt'] == cc]

            # SUBSET YTRAIN AND YTEST BY THE CORRESPONDING INDICES IN SUBSETTED XDATA
            sub_ytrain = ytrain[ytrain.index.isin(sub_xtrain.index)]
            sub_ytest = ytest[ytest.index.isin(sub_xtest.index)]

            # REMOVE CPT COLUMN
            del sub_xtrain['cpt']
            del sub_xtest['cpt']

            # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE OR CONTAINS NEGATIVE NUMBER
            if any(np.unique(sub_ytrain.values) < 0) or all(np.unique(sub_ytrain.values) == 0) or any(np.unique(sub_ytest.values) < 0) or all(np.unique(sub_ytest.values) == 0) or len(sub_ytrain.values) == 0 or len(sub_ytest.values) == 0:
                within_holder.append(pd.DataFrame({'auc': 'NA',
                                                   'cpt': cc}, index=[0]))
            else:
                # TRAIN MODEL
                logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
                logit_fit = logisticreg.fit(sub_xtrain, sub_ytrain.values.ravel())
                # GET PREDICTIONS
                logit_preds = logit_fit.predict_proba(sub_xtest)[:, 1]

                within_holder.append(
                    pd.DataFrame({'auc': metrics.roc_auc_score(sub_ytest.values, logit_preds), 'cpt': cc}, index=[0]))
        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'logit_sub_phat.csv'), index=False)
