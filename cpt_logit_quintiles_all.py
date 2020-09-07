import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import statsmodels.api as sm

# DESCRIPTION: THIS SCRIPT GENERATES COEFFICIENT VALUES FOR THE AGGREGATE AND SUB MODELS.
# THE SUBMODELS ARE DEFINED BY THEIR RISK QUINTILE, NOT INDIVIDUAL CPT CODE
# THIS SCRIPT DIFFERS FROM CPT_LOGIT_QUINTILES.PY BECUASE IT AGGREGATES OVER ALL YEARS (NO CROSS VALIDATION)
# AND ONLY SAVES COEFFICIENTS NOT AUC
# SAVES TO OUTPUT:
# --- logit_agg_coef_all_years.csv
# --- logit_sub_coef_all_years.csv

###############################
# ---- STEP 1: LOAD DATA ---- #
dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_base, '..', 'figures')

fn_X = 'X_imputed.csv'
fn_Y = 'y_agg.csv'

dat_X = pd.read_csv(os.path.join(dir_output, fn_X))
dat_Y = pd.read_csv(os.path.join(dir_output, fn_Y))

# CREATE DUMMY VARIABLES FOR NON NUMERIC # move this into the loop
dat_X = pd.get_dummies(dat_X, drop_first=True)

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

for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))

    # GROUP BY CPT GET MEAN OF OUTCOME (RISK)
    cpt_groups = pd.DataFrame(dat.groupby('cpt')[vv].apply(np.mean).reset_index().rename(columns={vv: 'outcome_mean'}))

    # REMOVE CPTS THAT HAVE NO RISK (ALL ZERO) FOR OUTCOME VV
    cpt_groups = cpt_groups[cpt_groups['outcome_mean'] > 0].reset_index(drop=False)

    # GET QUINTILES
    cpt_groups['bin'] = pd.qcut(cpt_groups['outcome_mean'], 5, labels=False)

    # GET LIST OF CPTS
    sub_cpts = cpt_groups.cpt.unique()

    # SUBSET DATA BY CPTS
    sub_dat = dat[dat['cpt'].isin(sub_cpts)].reset_index(drop=False)

    # GET TRAIN YEARS
    tmp_ii = pd.concat([sub_dat.operyr, dat[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)

    # GET INDEX FOR W/OUT -1 OUTCOME
    idx_years = dat.operyr.isin(tmp_years)

    # TRAIN AND TEST DATA
    dat_x = dat.loc[idx_years, cn_X].reset_index(drop=True)
    dat_y = dat.loc[idx_years, [vv]].reset_index(drop=True)

    # STORE CPTS
    tmp_cpt = dat_x.cpt.unique()

    # REMOVE CPTS
    del dat_x['cpt']

    # NORMALIZE DATA
    x = dat_x.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    dat_x = pd.DataFrame(x_scaled)

    # RUN MODEL AND STORE COEFFICIENTS AND P VALUES
    sm_model = sm.Logit(dat_y.values.ravel(), sm.add_constant(dat_x), max_iter=3000).fit(disp=0,method='bfgs')
    p_vals = sm_model.pvalues
    coef = sm_model.params
    outcome_coef.append(pd.DataFrame({'coef': list(coef), 'p_val': list(p_vals), 'outcome':vv}))

# SAVE CPT AUC FOR AGGREGATE MODEL
agg_coef = pd.concat(outcome_coef)
agg_coef.to_csv(os.path.join(dir_output, 'logit_agg_coef_all_years.csv'), index=False)

####################################################
# ---- STEP 3: LEAVE-ONE-YEAR - SUB MODEL AUC FOR QUINTILE BINS AND CPT---- #

outcome_bin_coef = []

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

    # GET INDEX FOR YEARS WITHOUT -1 OUTCOME
    idx_years = dat.operyr.isin(tmp_years)

    # TRAIN AND TEST DATA
    dat_x = dat.loc[idx_years, cn_X].reset_index(drop=True)
    dat_y = dat.loc[idx_years, [vv]].reset_index(drop=True)

    # GET UNIQUE BINS FOR LOOP
    cpt_bin = cpt_groups.bin.sort_values().unique()
    bin_coef_holder = []
    for bb in cpt_bin:
        # SUBSET BY BIN
        tmp_bins = cpt_groups[cpt_groups['bin'] == bb]

        # SUBSET XTRAIN AND XTEST BY CPTS
        bin_x = dat_x[dat_x['cpt'].isin(tmp_bins.cpt)]

        # SUBSET YTRAIN AND YTEST BY THE CORRESPONDING INDICES IN SUBSETTED XDATA
        bin_y = dat_y[dat_y.index.isin(bin_x.index)]


        if np.unique(bin_y.values).size <= 1 or np.count_nonzero(bin_y.values == 1) < 3:
            bin_coef_holder.append(pd.DataFrame({'coef': 'NA',
                                                     'p_val': 'NA',
                                                     'bin': bb,
                                                     'num_obs': cpt_ytest.values.size}))
        else:
            # REMOVE CPT COLUMN
            del bin_x['cpt']

            for col in bin_x.columns:
                if len(bin_x[col].unique()) == 1:
                    bin_x.drop(col, inplace=True, axis=1)
            # NORMALIZE DATA
            x = bin_x.values  # returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            bin_x = pd.DataFrame(x_scaled)

            # RUN MODEL AND STORE COEFFICIENTS AND P VALUES
            sm_model_bin = sm.Logit(bin_y.values.ravel(), sm.add_constant(bin_x), max_iter=2000).fit(disp=0,method ='bfgs')
            p_vals_bin = sm_model_bin.pvalues
            coef_bin = sm_model_bin.params
            bin_coef_holder.append(pd.DataFrame({'coef': list(coef_bin), 'p_val': list(p_vals_bin), 'bin':bb, 'outcome': vv}))

    outcome_bin_coef.append(pd.concat(bin_coef_holder).assign(outcome=vv))

# save data
agg_coef_bin = pd.concat(outcome_bin_coef).reset_index(drop=True)
agg_coef_bin.to_csv(os.path.join(dir_output, 'logit_sub_coef_all_years.csv'), index=False)
