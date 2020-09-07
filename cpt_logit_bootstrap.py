import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import statsmodels.stats.multitest as smm

# DESCRIPTION: THIS SCRIPT GENERATES AUC SCORES FOR THE AGGREGATE AND SUB MODELS FROM BOOTSTRAPPED Y VALUES AND PREDICTIONS
# SAVES TO OUTPUT:
# --- logit_boot_agg.csv
# --- logit_boot_sub.csv
# --- logit_sig_cpts.csv'

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
        Xtrain, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                        dat_X.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = dat_Y.loc[idx_train, [vv]].reset_index(drop=True), \
                        dat_Y.loc[idx_test, [vv]].reset_index(drop=True)

        # STORE CPT CODES AND DELETE FROM DATA
        tmp_cpt = Xtest.cpt
        del Xtrain['cpt']
        del Xtest['cpt']

        # TRAIN MODEL
        logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
        logit_fit = logisticreg.fit(Xtrain, ytrain.values.ravel())

        # GET PREDICTIONS
        logit_preds = logit_fit.predict_proba(Xtest)[:, 1]

        # STORE RESULTS FROM AGGREGATE MODEL
        tmp_holder = pd.DataFrame({'y_preds': list(logit_preds), 'y_values': list(ytest.values.ravel()), 'cpt': list(tmp_cpt)})
        within_holder = []
        # LOOP THROUGH EACH CPT CODE
        for cc in top_cpts:
            sub_tmp_holder = tmp_holder[tmp_holder['cpt'] == cc].reset_index(drop=True)
            # GET Y VALUES AND PREDICTIONS FOR BOOTSTRAPPING
            y_pred = sub_tmp_holder.y_preds.values
            y_true = sub_tmp_holder.y_values.values.ravel()

            # 1000 BOOTSTRAPS
            n_bootstraps = 1000
            rng_seed = 42  # control reproducibility
            bootstrapped_scores = []
            rng = np.random.RandomState(rng_seed)
            for i in range(n_bootstraps):
                # bootstrap by sampling with replacement on the prediction indices
                indices = rng.randint(0, len(y_pred), len(y_pred))
                if len(np.unique(y_true[indices])) < 2:
                    # We need at least one positive and one negative sample for ROC AUC
                    # to be defined: reject the sample
                    continue

                score = roc_auc_score(y_true[indices], y_pred[indices])
                bootstrapped_scores.append(score)
                #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

            within_holder.append(pd.DataFrame({'boot_aucs': list(bootstrapped_scores), 'cpt': cc}))

        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

auc_agg = pd.concat(holder_y_all).reset_index(drop=True)
auc_agg.to_csv(os.path.join(dir_output, 'logit_boot_agg.csv'), index=False)

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

            # DELETE CPT CODE FROM DATA
            del sub_xtrain['cpt']
            del sub_xtest['cpt']

            # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
            if all(np.unique(sub_ytrain.values) == 0) or all(np.unique(sub_ytest.values) == 0):
                within_holder.append(pd.DataFrame({'boot_aucs': list('0'), 'cpt': cc}))

            else:
                # TRAIN MODEL
                logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
                logit_fit = logisticreg.fit(sub_xtrain, sub_ytrain.values.ravel())

                # GET PREDICTION
                logit_preds = logit_fit.predict_proba(sub_xtest)[:, 1]

                # 1000 BOOTSTRAPS
                n_bootstraps = 1000
                rng_seed = 42  # control reproducibility
                bootstrapped_scores = []
                y_true = sub_ytest.values.ravel()
                y_pred = logit_preds
                rng = np.random.RandomState(rng_seed)
                for i in range(n_bootstraps):
                    # bootstrap by sampling with replacement on the prediction indices
                    indices = rng.randint(0, len(y_pred), len(y_pred))
                    if len(np.unique(y_true[indices])) < 2:
                        # We need at least one positive and one negative sample for ROC AUC
                        # to be defined: reject the sample
                        continue

                    score = roc_auc_score(y_true[indices], y_pred[indices])
                    bootstrapped_scores.append(score)
                    # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

                within_holder.append(pd.DataFrame({'boot_aucs': list(bootstrapped_scores), 'cpt': cc}))


        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

auc_sub = pd.concat(holder_y_all).reset_index(drop=True)
auc_sub.to_csv(os.path.join(dir_output, 'logit_boot_sub.csv'), index=False)

# ---------------------------- bootstrap analysis
# compare aggregate and sub model auc
#auc_agg = pd.read_csv(os.path.join(dir_output, 'logit_boot_agg.csv'))
#auc_sub = pd.read_csv(os.path.join(dir_output, 'logit_boot_sub.csv'))

# REMOVE ROWS WITH 0 AS AUC - THIS WAS A PLACEHOLDER FOR CPT CODES WITH NO POSITIVE VALUES
auc_agg = auc_agg[auc_agg['boot_aucs']!=0]
auc_sub = auc_sub[auc_sub['boot_aucs']!=0]

# CREATE COLUMN TO IDENTIFY DATA
auc_agg = auc_agg.rename(columns = {'boot_aucs': 'agg_boot_aucs'}, inplace = False)
auc_sub = auc_sub.rename(columns = {'boot_aucs': 'sub_boot_aucs'}, inplace = False)

# GET LIST OF OUTCOME NAMES TO LOOP THROUGH
outcome_list = list(auc_agg.outcome.unique())
num_boots = 1000
outcome_results = []
for i in outcome_list:
    print(i)
    temp_agg = auc_agg[auc_agg['outcome'] == i].reset_index(drop=True)
    temp_sub = auc_sub[auc_sub['outcome'] == i].reset_index(drop=True)
    year_list =np.intersect1d(temp_agg.test_year.unique(), temp_agg.test_year.unique())
    year_results = []
    for j in year_list:
        print(j)
        year_agg = temp_agg[temp_agg['test_year'] == j].reset_index(drop=True)
        year_sub = temp_sub[temp_sub['test_year'] == j].reset_index(drop=True)
        cpt_list = np.intersect1d(year_agg.cpt.unique(), year_sub.cpt.unique())
        cpt_results = []
        for k in cpt_list:
            cpt_agg = year_agg[year_agg['cpt'] == k].reset_index(drop=True)
            cpt_sub = year_sub[year_sub['cpt'] == k].reset_index(drop=True)
            if cpt_sub.shape[0] == 0 or cpt_agg.shape[0] == 0:
                cpt_results.append(pd.DataFrame(
                    {'sig_value': 'NA', 'agg_p_value': 'NA', 'diff_p_value': 'NA', 'cpt': 'NA'},index=[0]))
            else:
                temp = pd.merge(cpt_agg, cpt_sub, left_index=True, right_index=True)
                temp = temp[['sub_boot_aucs', 'agg_boot_aucs']]
                temp['auc_diff'] = temp.sub_boot_aucs.values - temp.agg_boot_aucs.values
                temp['agg_diff'] = temp.agg_boot_aucs.values - 0.5
                # GET 2.5% VALUE
                sig_value = temp.auc_diff.quantile(0.025)
                sig_value_agg = temp.agg_diff.quantile(0.025)
                # GENERATE PVALUES
                agg_p_value = 1 - ((temp[temp['agg_boot_aucs'] > 0.5].shape[0]))/(temp.shape[0] +1)
                diff_p_value = 1 - (temp[temp['auc_diff'] > 0].shape[0]) / (temp.shape[0] +1)
                cpt_results.append(pd.DataFrame(
                    {'sig_value_diff': sig_value, 'sig_value_agg':sig_value_agg, 'agg_p_value': agg_p_value, 'diff_p_value': diff_p_value, 'cpt': k},
                    index=[0]))
        year_results.append(pd.concat(cpt_results).assign(test_year=j))
    outcome_results.append(pd.concat(year_results).assign(outcome=i))

sig_cpts = pd.concat(outcome_results).reset_index(drop=True)


# LOOP THROUGH OUTCOME AND YEAR AND GET FDR CORRECT PVALUES
outcome_results = []
for i in outcome_list:
    print(i)
    temp_sig = sig_cpts[sig_cpts['outcome'] == i].reset_index(drop=True)
    year_list = temp_sig.test_year.unique()
    year_results = []
    for j in year_list:
        print(j)
        year_sig = temp_sig[temp_sig['test_year'] == j].reset_index(drop=True)
        year_sig['agg_p_value_adj'] = smm.fdrcorrection(year_sig.agg_p_value.values)[1]
        year_sig['diff_p_value_adj'] = smm.fdrcorrection(year_sig.diff_p_value.values)[1]
        year_results.append(year_sig)

    outcome_results.append(pd.concat(year_results))

sig_cpts = pd.concat(outcome_results).reset_index(drop=True)

#sig_cpts = sig_cpts[sig_cpts['sig_value_diff'] >0]
#sig_cpts = sig_cpts[sig_cpts['sig_value_agg']>0]
sig_cpts.to_csv(os.path.join(dir_output, 'logit_sig_cpts.csv'), index=False)
