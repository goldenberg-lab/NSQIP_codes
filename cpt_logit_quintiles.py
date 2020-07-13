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

# DELETE NON AGG LABELS
dat_Y.drop(dat_Y.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
           axis=1, inplace=True)

# JOIN DAT_X AND DAT_Y
dat = pd.merge(dat_Y, dat_X, on='caseid')

####################################################
# ---- STEP 2: LEAVE-ONE-YEAR - AGGREGATE MODEL AUC FOR QUINTILE BINS AND CPT---- #

# LIST FOR BIN AUC AND CPT (WITHIN BIN) AUC FOR AGGREGATE MODEL
outcome_coef = []
outcome_bin = []
outcome_cpt = []

for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))

    # GROUP BY CPT GET MEAN OF OUTCOME (RISK)
    cpt_groups = pd.DataFrame(dat.groupby('cpt')[vv].apply(np.mean).reset_index().rename(columns={vv: 'outcome_mean'}))

    # REMOVE CPTS THAT HAVE NO RISK (ALL ZERO) FOR OUTCOME VV
    cpt_groups = cpt_groups[cpt_groups['outcome_mean'] > 0].reset_index(drop=False)

    # GET QUINTILES
    cpt_groups['bin'] = pd.qcut(cpt_groups['outcome_mean'], 5, labels=False)

    # subet data by cpt groups
    sub_cpts = cpt_groups.cpt.unique()

    # subset data by cpts
    sub_dat = dat[dat['cpt'].isin(sub_cpts)].reset_index(drop=False)

    # GET TRAIN YEARS
    tmp_ii = pd.concat([sub_dat.operyr, dat[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]

    year_coef = []
    year_bin = []
    year_cpt = []
    for yy in tmp_train_years:
        print('Train Year %i' % (yy))
        idx_train = sub_dat.operyr.isin(tmp_years) & (sub_dat.operyr < yy)
        idx_test = sub_dat.operyr.isin(tmp_years) & (sub_dat.operyr == yy)

        # GET TRAIN AND TEST DATA
        Xtrain, Xtest = sub_dat.loc[idx_train, cn_X].reset_index(drop=True), \
                        sub_dat.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = sub_dat.loc[idx_train, [vv]].reset_index(drop=True), \
                        sub_dat.loc[idx_test, [vv]].reset_index(drop=True)

        # STORE CPT CODES
        tmp_cpt = Xtest.cpt

        # REMOVE CPT FROM DATA
        del Xtrain['cpt']
        del Xtest['cpt']

        # TRAIN MODEL
        logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
        logit_fit = logisticreg.fit(Xtrain, ytrain.values.ravel())

        # TEST MODEL
        logit_preds = logit_fit.predict_proba(Xtest)[:, 1]

        # get coefficients from model
        coef = logit_fit.coef_

        tmp_coef = []
        tmp_coef.append(pd.DataFrame({'coef': list(coef)}, index=[0]))


        # GET BIN NAMES FOR LOOP
        cpt_bin = cpt_groups.bin.sort_values().unique()

        # STORE RESULTS FROM AGGREGATE MODEL
        tmp_holder = pd.DataFrame({'y_preds': list(logit_preds), 'y_values': list(ytest.values), 'cpt': list(tmp_cpt)})

        # LOOP THROUGH BINS
        bin_holder = []
        cpt_bin_holder = []
        for bb in cpt_bin:

            # GET CPTS FROM BIN
            tmp_cpts = np.array(cpt_groups.cpt[cpt_groups['bin'] == bb])

            # SUBSET MODEL RESULT BY CPTS
            bin_tmp_holder = tmp_holder[tmp_holder['cpt'].isin(tmp_cpts)].reset_index(drop=False)

            # GET UNIQUE CPTS FOR FOR LOOP
            cpt_codes = bin_tmp_holder.cpt.sort_values().unique()

            cpt_holder = []
            for cc in cpt_codes:

                # SUBSET BY CPT
                cpt_tmp_holder =bin_tmp_holder[bin_tmp_holder['cpt']==cc]

                # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
                if all(cpt_tmp_holder.y_values.values == 0) or  all(cpt_tmp_holder.y_values.values == 1) or len(cpt_tmp_holder.y_values) <= 1:
                    cpt_holder.append(pd.DataFrame({'auc': 'NA',
                                                    'cpt': cc,
                                                    'num_obs': cpt_tmp_holder.y_values.size}, index=[0]))

                else:
                    cpt_holder.append(pd.DataFrame({'auc': metrics.roc_auc_score(list(cpt_tmp_holder.y_values.values),
                                                                   list(cpt_tmp_holder.y_preds.values)),
                                                    'cpt': cc,
                                                    'num_obs': cpt_tmp_holder.y_values.size}, index=[0]))
            cpt_bin_holder.append(pd.concat(cpt_holder).assign(bin=bb))

            # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
            if all(bin_tmp_holder.y_values.values == 0):
                bin_holder.append(pd.DataFrame({'auc': 'NA',
                                                'bin': bb,
                                                'num_obs': bin_tmp_holder.y_values.size}, index=[0]))
            else:
                bin_holder.append(pd.DataFrame({'auc': metrics.roc_auc_score(list(bin_tmp_holder.y_values.values),
                                                                                list(bin_tmp_holder.y_preds.values)),
                                                'bin': bb,
                                                'num_obs': bin_tmp_holder.y_values.size}, index=[0]))

        year_coef.append(pd.concat(tmp_coef).assign(test_year=yy))
        year_cpt.append(pd.concat(cpt_bin_holder).assign(test_year=yy))
        year_bin.append(pd.concat(bin_holder).assign(test_year=yy))
    outcome_coef.append(pd.concat(year_coef).assign(outcome=vv))
    outcome_cpt.append(pd.concat(year_cpt).assign(outcome=vv))
    outcome_bin.append(pd.concat(year_bin).assign(outcome=vv))


# SAVE CPT AUC FOR AGGREGATE MODEL
agg_auc_cpt = pd.concat(outcome_cpt).reset_index(drop=True)
agg_auc_cpt.to_csv(os.path.join(dir_output, 'logit_auc_agg_quin_cpt.csv'), index=False)

# SAVE QUINTILE AUC FOR AGGREGATE MODEL
agg_auc_bin = pd.concat(outcome_bin).reset_index(drop=True)
agg_auc_bin.to_csv(os.path.join(dir_output, 'logit_auc_agg_quin_bin.csv'), index=False)

# SAVE coefficients for agg model
agg_coef = pd.concat(outcome_coef).reset_index(drop=True)
agg_coef.to_csv(os.path.join(dir_output, 'logit_agg_coef.csv'), index=False)

####################################################
# ---- STEP 3: LEAVE-ONE-YEAR - SUB MODEL AUC FOR QUINTILE BINS AND CPT---- #

# LIST FOR BIN AUC AND CPT (WITHIN BIN) AUC FOR SUB MODELS

outcome_bin_coef = []
outcome_cpt_coef = []
outcome_bin = []
outcome_cpt = []

for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))

    # GROUP BY CPT GET MEAN OF OUTCOME (RISK)
    cpt_groups = pd.DataFrame(dat.groupby('cpt')[vv].apply(np.mean).reset_index().rename(columns={vv: 'outcome_mean'}))

    # REMOVE CPTS THAT HAVE NO RISK (ALL ZERO) FOR OUTCOME VV
    cpt_groups = cpt_groups[cpt_groups['outcome_mean'] > 0].reset_index(drop=False)

    # GET BINS
    cpt_groups['bin'] = pd.qcut(cpt_groups['outcome_mean'], 5, labels=False)

    # subet data by cpt groups
    sub_cpts = cpt_groups.cpt.unique()

    # subset data by cpts
    sub_dat = dat[dat['cpt'].isin(sub_cpts)].reset_index(drop=False)

    # GET TRAIN YEARS
    tmp_ii = pd.concat([dat.operyr, dat[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]

    year_cpt_coef = []
    year_bin_coef = []
    year_bin = []
    year_cpt = []
    for yy in tmp_train_years:
        print('Train Year %i' % (yy))
        idx_train = sub_dat.operyr.isin(tmp_years) & (sub_dat.operyr < yy)
        idx_test = sub_dat.operyr.isin(tmp_years) & (sub_dat.operyr == yy)

        # TRAIN AND TEST DATA
        Xtrain, Xtest = sub_dat.loc[idx_train, cn_X].reset_index(drop=True), \
                        sub_dat.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = sub_dat.loc[idx_train, [vv]].reset_index(drop=True), \
                        sub_dat.loc[idx_test, [vv]].reset_index(drop=True)

        # GET UNIQUE BINS FOR LOOP
        cpt_bin = cpt_groups.bin.sort_values().unique()
        bin_holder = []
        bin_coef_holder =[]
        cpt_bin_holder = []
        cpt_bin_coef_holder = []
        for bb in cpt_bin:
            # SUBSET BY BIN
            tmp_bins = cpt_groups[cpt_groups['bin']==bb]

            # SUBSET XTRAIN AND XTEST BY CPTS
            bin_xtrain = Xtrain[Xtrain['cpt'].isin(tmp_bins.cpt)]
            bin_xtest = Xtest[Xtest['cpt'].isin(tmp_bins.cpt)]

            # SUBSET YTRAIN AND YTEST BY THE CORRESPONDING INDICES IN SUBSETTED XDATA
            bin_ytrain = ytrain[ytrain.index.isin(bin_xtrain.index)]
            bin_ytest = ytest[ytest.index.isin(bin_xtest.index)]

            # GET CPT CODES
            cpt_codes = tmp_bins.cpt.sort_values().unique()
            cpt_holder = []
            cpt_coef_holder = []
            for cc in cpt_codes:
                # GET CPTS CODES FROM CURRENT BIN
                tmp_cpts = tmp_bins[tmp_bins['cpt'] == cc]

                # SUBSET XTRAIN AND XTEST BY CPTS
                cpt_xtrain = bin_xtrain[bin_xtrain['cpt'].isin(tmp_cpts.cpt)]
                cpt_xtest = bin_xtest[bin_xtest['cpt'].isin(tmp_cpts.cpt)]

                # SUBSET YTRAIN AND YTEST BY THE CORRESPONDING INDICES IN SUBSETTED XDATA
                cpt_ytrain = bin_ytrain[bin_ytrain.index.isin(cpt_xtrain.index)]
                cpt_ytest = bin_ytest[bin_ytest.index.isin(cpt_xtest.index)]

                # REMOVE CPT
                del cpt_xtrain['cpt']
                del cpt_xtest['cpt']

                # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
                if np.unique(cpt_ytrain.values).size <= 1 or np.unique(cpt_ytest.values).size <= 1:
                    cpt_holder.append(pd.DataFrame({'auc': 'NA',
                                                    'cpt': cc,
                                                    'num_obs': cpt_ytest.values.size}, index=[0]))
                    cpt_coef_holder.append(pd.DataFrame({'coef':'NA',
                                                         'cpt': cc,
                                                         'num_obs': cpt_ytest.values.size}, index=[0]))
                else:
                    # TRAIN MODEL
                    logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
                    logit_fit = logisticreg.fit(cpt_xtrain, cpt_ytrain.values.ravel())

                    # TEST MODEL
                    logit_preds = logit_fit.predict_proba(cpt_xtest)[:, 1]

                    # get coefficients from model
                    coef_cpt = logit_fit.coef_

                    cpt_holder.append(
                        pd.DataFrame({'auc': metrics.roc_auc_score(cpt_ytest.values, logit_preds),
                                      'cpt': cc,
                                      'num_obs': cpt_ytest.values.size}, index=[0]))
                    cpt_coef_holder.append(pd.DataFrame({'coef':list(coef_cpt),
                                                         'cpt': cc,
                                                         'num_obs':cpt_ytest.values.size}, index=[0]))


            cpt_bin_holder.append(pd.concat(cpt_holder).assign(bin=bb))
            cpt_bin_coef_holder.append(pd.concat(cpt_coef_holder).assign(bin=bb))


            if all(np.unique(bin_ytrain.values) == 0) or all(np.unique(bin_ytest.values) == 0):
                bin_holder.append(pd.DataFrame({'auc': 'NA',
                                                'bin': bb,
                                                'num_obs': bin_ytest.values.size}, index=[0]))
                bin_coef_holder.append(pd.DataFrame({'coef': 'NA',
                                                    'bin': bb,
                                                     'num_obs': bin_ytest.values.size}, index=[0]))
            else:
                # REMOVE CPT COLUMN
                del bin_xtrain['cpt']
                del bin_xtest['cpt']

                # TRAIN MODEL
                logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
                logit_fit = logisticreg.fit(bin_xtrain, bin_ytrain.values.ravel())

                # TEST MODEL
                logit_preds = logit_fit.predict_proba(bin_xtest)[:, 1]

                # get coefficients from model
                coef_bin = logit_fit.coef_


                bin_holder.append(pd.DataFrame({'auc': metrics.roc_auc_score(bin_ytest.values, logit_preds),
                                  'bin': bb,
                                  'num_obs': bin_ytest.values.size}, index=[0]))
                bin_coef_holder.append(pd.DataFrame({'coef': list(coef_bin),
                                                'bin': bb,
                                                'num_obs': bin_ytest.values.size}, index=[0]))
        year_cpt_coef.append(pd.concat(cpt_bin_coef_holder).assign(test_year=yy))
        year_bin_coef.append(pd.concat(bin_coef_holder).assign(test_year=yy))
        year_cpt.append(pd.concat(cpt_bin_holder).assign(test_year=yy))
        year_bin.append(pd.concat(bin_holder).assign(test_year=yy))
    outcome_cpt_coef.append(pd.concat(year_cpt_coef).assign(outcome=vv))
    outcome_bin_coef.append(pd.concat(year_bin_coef).assign(outcome=vv))
    outcome_cpt.append(pd.concat(year_cpt).assign(outcome=vv))
    outcome_bin.append(pd.concat(year_bin).assign(outcome=vv))


# SAVE AUC FOR CPTS ON SUB MODELS
agg_auc_cpt = pd.concat(outcome_cpt).reset_index(drop=True)
agg_auc_cpt.to_csv(os.path.join(dir_output, 'logit_auc_sub_quin_cpt.csv'), index=False)

agg_coef_cpt = pd.concat(outcome_cpt_coef).reset_index(drop=True)
agg_coef_cpt.to_csv(os.path.join(dir_output, 'logit_coef_sub_cpt.csv'), index=False)

# SAVE AUC FOR QUNITILE BINS FOR SUB MODELS
agg_auc_bin = pd.concat(outcome_bin).reset_index(drop=True)
agg_auc_bin.to_csv(os.path.join(dir_output, 'logit_auc_sub_quin_bin.csv'), index=False)

agg_coef_bin = pd.concat(outcome_bin_coef).reset_index(drop=True)
agg_coef_bin.to_csv(os.path.join(dir_output, 'logit_coef_sub_bin.csv'), index=False)