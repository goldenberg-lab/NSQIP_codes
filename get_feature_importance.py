import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import scipy.stats as stat
from plotnine import *
import random
from plydata.cat_tools import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class LogisticReg:
    """
    Wrapper Class for Logistic Regression which has the usual sklearn instance
    in an attribute self.model, and pvalues, z scores and estimated
    errors for each coefficient in

    self.z_scores
    self.p_values
    self.sigma_estimates

    as well as the negative hessian of the log Likelihood (Fisher information)

    self.F_ij
    """

    def __init__(self, *args, **kwargs):  # ,**kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)  # ,**args)

    def fit(self, X, y):
        self.model.fit(X, y)
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X)  ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)  ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates  # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]  ### two tailed test for p-values

        return p_values

###############################
# ---- STEP 1: LOAD DATA ---- #
dir_base = os.getcwd()
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
#cn_X.append('caseid') # here
# just keep SSI
cn_Y = list(dat_Y.columns[29:31])
# DELETE NON AGG LABELS
dat_Y.drop(dat_Y.columns[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26,27,28,32,33,34,35]],
           axis=1, inplace=True)

# recode htooday
dat_X['htooday'] = np.where(dat_X['htooday'] == 1, 'yes', 'no')

# LIST FOR BIN AUC AND CPT (WITHIN BIN) AUC FOR AGGREGATE MODEL
logit_coef = []
xgb_imp = []
logit_new =[]
for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))

    # GET TRAIN YEARS
    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)

    # GET INDEX FOR W/OUT -1 OUTCOME
    idx_years = dat_Y.operyr.isin(tmp_years)
    dat_x = dat_X.loc[idx_years, cn_X].reset_index(drop=True)
    dat_y = dat_Y.loc[idx_years, [vv]].reset_index(drop=True)

    # STORE CPTS
    tmp_cpt = dat_x.cpt.unique()

    # REMOVE CPTS
    del dat_x['cpt']

    # USE COLUMNTRASNFORM HERE
    scaler = StandardScaler()
    num_vars = list(['age_days', 'height', 'weight', 'workrvu'])    # NORMALIZE DATA
    cat_vars = [i for i in dat_x.columns if i not in num_vars]

    dat_x[num_vars] = scaler.fit_transform(dat_x[num_vars])
    dat_x[cat_vars]= dat_x[cat_vars].apply(lambda x: cat_infreq(x), 0)
    dat_x = pd.get_dummies(dat_x, drop_first=True)

    # RUN MODELs AND STORE COEFFICIENTS AND P VALUES - here use the class above to get p values (right now still not converging, try changing max iter above)
    clf = LogisticReg(penalty='l2', solver='liblinear',max_iter=2000)
    p_vals = clf.fit(dat_x, dat_y.values.ravel())
    feat_names = dat_x.columns
    logit_new.append(pd.DataFrame({'feature_names':list(feat_names), 'p_val': list(p_vals), 'outcome':vv, 'seed': s}))

    # LOGIT
    sm_model = sm.Logit(dat_y.values.ravel(), sm.add_constant(dat_x), max_iter=1000000).fit(disp=0,method='ncg')
    p_vals = sm_model.pvalues
    feat_names = p_vals.index
    coef = sm_model.params
    logit_coef.append(pd.DataFrame({'feature_names':list(feat_names), 'coef': list(coef), 'p_val': list(p_vals), 'outcome':vv, 'seed': s}))

    # XGBoost
    clf = xgb.XGBClassifier()
    xgb_mod = clf.fit(dat_x, dat_y.values.ravel())
    feat_imp = xgb_mod.feature_importances_
    feat_names = dat_x.columns
    xgb_imp.append(pd.DataFrame({'feature_names':list(feat_names), 'feature_importance': list(feat_imp),'outcome':vv, 'seed': s}))
    print('finished ' + str(vv) + ' ' + str(s))

# SAVE CPT AUC FOR AGGREGATE MODEL
xgb_imp = pd.concat(xgb_imp)
logit_coef = pd.concat(logit_coef)
logit_new = pd.concat(logit_new)

# make logit p values significant if large (to compare with feature importance)
logit_coef.to_csv(os.path.join(dir_data, 'logit_features.csv'), index=False)
xgb_imp.to_csv(os.path.join(dir_data, 'xgb_features.csv'), index=False)
logit_new.to_csv(os.path.join(dir_data, 'logit_new.csv'), index=False)

# shuffle the x matrix rows and run xgboost
xgb_imp = []
#logit_new =[]
for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii + 1, len(cn_Y)))

    # GET TRAIN YEARS
    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)

    # GET INDEX FOR W/OUT -1 OUTCOME
    idx_years = dat_Y.operyr.isin(tmp_years)
    dat_y = dat_Y.loc[idx_years, [vv]].reset_index(drop=True)

    # randomly shuffle rows
    for s in range(100):
        random.seed(s)
        if s == 0:
            # TRAIN AND TEST DATA
            dat_x = dat_X.loc[idx_years, cn_X].reset_index(drop=True)
        else:
            dat_X = dat_X.loc[idx_years, cn_X].reset_index(drop=True)
            idx_random = random.sample(list(dat_X.index), dat_X.shape[0])
            dat_x = dat_X.iloc[idx_random, ].reset_index(drop=True)

        # STORE CPTS
        tmp_cpt = dat_x.cpt.unique()

        # REMOVE CPTS
        del dat_x['cpt']

        # USE COLUMNTRASNFORM HERE
        scaler = StandardScaler()
        num_vars = list(['age_days', 'height', 'weight', 'workrvu'])    # NORMALIZE DATA
        cat_vars = [i for i in dat_x.columns if i not in num_vars]

        dat_x[num_vars] = scaler.fit_transform(dat_x[num_vars])
        dat_x[cat_vars]= dat_x[cat_vars].apply(lambda x: cat_infreq(x), 0)

        dat_x = pd.get_dummies(dat_x, drop_first=True)

        # XGBoost
        clf = xgb.XGBClassifier()
        xgb_mod = clf.fit(dat_x, dat_y.values.ravel())
        feat_imp = xgb_mod.feature_importances_
        feat_names = dat_x.columns
        xgb_imp.append(pd.DataFrame({'feature_names':list(feat_names), 'feature_importance': list(feat_imp),'outcome':vv, 'seed': s}))
        print('finished ' + str(vv) + ' ' + str(s))

# SAVE CPT AUC FOR AGGREGATE MODEL
xgb_imp = pd.concat(xgb_imp)

# save data
xgb_imp.to_csv(os.path.join(dir_data, 'xgb_shuffled_features.csv'), index=False)

# logit_coef = pd.read_csv(os.path.join(dir_data, 'logit_features.csv'))
# xgb_imp = pd.read_csv(os.path.join(dir_data, 'xgb_features.csv'))
#
# ##################
# # plot logit ssi1 p values
# ##################
# logit_ssi1 = logit_coef[logit_coef['outcome']=='agg_ssi1'].reset_index(drop=True)
# feat_sorted = logit_ssi1.sort_values('p_val').feature_names
# img= (ggplot(logit_ssi1, aes(x= 'feature_names', y = 'p_val')) + geom_bar(stat='identity')) + scale_x_discrete(limits=feat_sorted) + labs(x='Feature names', y='p Values') + coord_flip() +theme_bw() + theme(axis_text=element_text(size = 5))
# img.save(os.path.join(dir_data, 'logit_ssi1_pvalue.png'))
#
# ##################
# # plot logit ssi2 p values
# ##################
# logit_ssi2 = logit_coef[logit_coef['outcome']=='agg_ssi2'].reset_index(drop=True)
# feat_sorted = logit_ssi2.sort_values('p_val').feature_names
# img= (ggplot(logit_ssi2, aes(x= 'feature_names', y = 'p_val')) + geom_bar(stat='identity')) + scale_x_discrete(limits=feat_sorted) + labs(x='Feature names', y='p Values') + coord_flip() +theme_bw() + theme(axis_text=element_text(size = 5))
# img.save(os.path.join(dir_data, 'logit_ssi2_pvalue.png'))
#
# ##############
# # plot xgb ssi1 feature importance
# ##################
# xgb_ssi1 = xgb_imp[xgb_imp['outcome']=='agg_ssi1'].reset_index(drop=True)
# feat_sorted = xgb_ssi1.sort_values('feature_importance').feature_names
# img= (ggplot(xgb_ssi1, aes(x= 'feature_names', y = 'feature_importance')) + geom_bar(stat='identity')) + scale_x_discrete(limits=feat_sorted) + labs(x='Feature names', y='Feature importance') + coord_flip() +theme_bw() + theme(axis_text=element_text(size = 5))
# img.save(os.path.join(dir_data, 'xgb_ssi1_importance.png'))
#
# ##################
# # plot xgb ssi2 feature importance
# ##################
# xgb_ssi2 = xgb_imp[xgb_imp['outcome']=='agg_ssi2'].reset_index(drop=True)
# feat_sorted = xgb_ssi2.sort_values('feature_importance').feature_names
# img= (ggplot(xgb_ssi2, aes(x= 'feature_names', y = 'feature_importance')) + geom_bar(stat='identity')) + scale_x_discrete(limits=feat_sorted) + labs(x='Feature names', y='Feature importance') + coord_flip() +theme_bw() + theme(axis_text=element_text(size = 5))
# img.save(os.path.join(dir_data, 'xgb_ssi2_importance.png'))
