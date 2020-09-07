import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# DESCRIPTION: THIS SCRIPT GENERATES AUC SCORES FOR THE AGGREGATE AND SUB MODELS.
# THE SUBMODELS ARE DEFINED BY THEIR ORGAN GROUP, NOT INDIVIDUAL CPT CODE
# SAVES TO OUTPUT:
# --- logit_agg_title.csv
# --- logit_sub_title.csv
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

# GET CPT ANNOTATIONS
file_name = 'cpt_anno.csv'
cpt_anno = pd.read_csv(os.path.join(dir_output, file_name))
cpt_anno['cpt'] = 'c' + cpt_anno.cpt.astype(str)

# GROUP BY TITLE AND GET COUNTS - REMOVE GROUPS WITH ONLY 1 COUNT
cpt_groups = cpt_anno.groupby('title').size().sort_values(ascending=False)
cpt_groups = pd.DataFrame({'cpt_title': cpt_groups.index, 'count': cpt_groups.values})

# KEEP CPT TITLES THAT HAVE MORE THAN ONE CPT CODE ASSOCIATED WITH THEM
top_groups = cpt_groups[cpt_groups['count'] > 1]
top_groups = top_groups.cpt_title.unique()

# SUBSET CPT GROUPS TO KEEP ONLY CPTS THAT HAVE A CORRESPONDING TITLE
cpt_anno = cpt_anno[cpt_anno.title.isin(top_groups)].reset_index(drop=True)
top_cpts = cpt_anno.cpt.unique()

# SUBSET CPT_GROUPS TO GET
# HERE NEED TO FIND A WAY TO GET THE CPT NAMES ASSOCIATED WITH EACH TITLE (MAYBE REFORMAT DATA IN R)

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

        # STORE CPT CODES
        tmp_cpt = Xtest.cpt

        # REMOVE CPT CODES
        del Xtrain['cpt']
        del Xtest['cpt']

        # TRAIN MODEL
        logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
        logit_fit = logisticreg.fit(Xtrain, ytrain.values.ravel())

        # PREDICT
        logit_preds = logit_fit.predict_proba(Xtest)[:, 1]

        # STORE RESULTS FROM AGGREGATE MODEL
        tmp_holder = pd.DataFrame({'y_preds': list(logit_preds), 'y_values': list(ytest.values), 'cpt': list(tmp_cpt)})
        within_holder = []

        # GET TOP TITLES
        top_titles = cpt_anno.title.unique()

        # LOOP THROUGH EACH CPT TITLE
        for cc in top_titles:
            title_cpts = cpt_anno[cpt_anno['title']==cc].reset_index(drop=True)
            title_cpts = title_cpts.cpt.unique()
            sub_tmp_holder = tmp_holder[tmp_holder['cpt'].isin(title_cpts)].reset_index(drop=True)
            if all(sub_tmp_holder.y_values.values == 0) or len(sub_tmp_holder.y_values.values) <= 1:
                within_holder.append(pd.DataFrame({'auc': 'NA',
                                                   'cpt': cc}, index=[0]))
            else:
                within_holder.append(pd.DataFrame({'auc': metrics.roc_auc_score(list(sub_tmp_holder.y_values.values),
                                                                                list(sub_tmp_holder.y_preds.values)),
                                                   'cpt': cc}, index=[0]))

        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'logit_agg_title.csv'), index=False)

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

        # GET TOP TITLES TO LOOP THROUGH
        top_titles = cpt_anno.title.unique()
        within_holder = []
        for cc in top_titles:

            # GET LIST OF TITLES
            title_cpts = cpt_anno[cpt_anno['title'] == cc].reset_index(drop=True)
            title_cpts = list(title_cpts.cpt.unique())

            # SUBSET XTRAIN AND XTEST BY CPT CODE
            sub_xtrain = Xtrain[Xtrain['cpt'].isin(title_cpts)]
            sub_xtest = Xtest[Xtest['cpt'].isin(title_cpts)]

            # SUBSET YTRAIN AND YTEST BY THE CORRESPONDING INDICES IN SUBSETTED XDATA
            sub_ytrain = ytrain[ytrain.index.isin(sub_xtrain.index)]
            sub_ytest = ytest[ytest.index.isin(sub_xtest.index)]

            # REVMOVE CPT COLUMN
            del sub_xtrain['cpt']
            del sub_xtest['cpt']

            # FILL RESULTS WITH NA IF TRAIN OR TEST OUTCOMES ARE ALL ONE VALUE
            if all(np.unique(sub_ytrain.values) == 0) or all(np.unique(sub_ytest.values) == 0) or all(sub_ytest.values == 1) or len(sub_ytest.values) <= 1:
                within_holder.append(pd.DataFrame({'auc': 'NA',
                                                   'cpt': cc}, index=[0]))
            else:
                # TRAIN MODEL
                logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
                logit_fit = logisticreg.fit(sub_xtrain, sub_ytrain.values.ravel())

                # TEST MODEL
                logit_preds = logit_fit.predict_proba(sub_xtest)[:, 1]

                within_holder.append(
                    pd.DataFrame({'auc': metrics.roc_auc_score(sub_ytest.values, logit_preds), 'cpt': cc}, index=[0]))

        holder_y.append(pd.concat(within_holder).assign(test_year=yy))
    holder_y_all.append(pd.concat(holder_y).assign(outcome=vv))

res_y_all = pd.concat(holder_y_all).reset_index(drop=True)
res_y_all.to_csv(os.path.join(dir_output, 'logit_sub_title.csv'), index=False)

