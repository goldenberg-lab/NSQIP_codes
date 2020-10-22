import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot
from support.naive_bayes import mbatch_NB
from sklearn import metrics
import xgboost as xgb

import seaborn as sns

###############################
# ---- STEP 1: LOAD DATA ---- #

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_figures = os.path.join(dir_base,'..','figures')

fn_X = 'X_imputed.csv'
fn_Y = 'y_agg.csv'
dat_X = pd.read_csv(os.path.join(dir_output,fn_X))
dat_Y = pd.read_csv(os.path.join(dir_output,fn_Y))
print(dat_X.shape); print(dat_Y.shape)
stopifnot(all(dat_X.caseid == dat_Y.caseid))
u_years = dat_X.operyr.unique()
# !! ENCODE CPT AS CATEGORICAL !! #
dat_X['cpt'] = 'c'+dat_X.cpt.astype(str)
cn_X = list(dat_X.columns[2:])
cn_Y = list(dat_Y.columns[2:])

###############################################
# ---- STEP 2: LEAVE-ONE-YEAR - CPT ONLY ---- #

holder_vv = []
holder_phat = []
for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii+1, len(cn_Y)))
    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv]==-1],axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv:'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_train_years = tmp_years[tmp_years > tmp_years.min()]
    holder_metrics = []
    holder_score = []
    for yy in tmp_train_years:
        print('Year %i' % (yy))
        idx_train = dat_X.operyr.isin(tmp_years) & (dat_X.operyr < yy)
        idx_test = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy)
        Xtrain, Xtest = dat_X.loc[idx_train, ['cpt']].reset_index(drop=True), \
                        dat_X.loc[idx_test, ['cpt']].reset_index(drop=True)
        ytrain, ytest = dat_Y.loc[idx_train,vv].reset_index(drop=True), \
                        dat_Y.loc[idx_test,vv].reset_index(drop=True)
        # --- train model --- #
        mdl_bernoulli = mbatch_NB(method='bernoulli')
        mdl_bernoulli.fit(Xtrain,ytrain.values,mbatch=100000)
        phat_bernoulli = mdl_bernoulli.predict(Xtest)[:,1]
        case_id = Xtext.caseid
        holder_score.append(pd.DataFrame({'y':ytest.values,'phat':phat_bernoulli,'operyr':yy, 'caseid':case_id}))
        holder_metrics.append(pd.DataFrame({'auc':metrics.roc_auc_score(ytest.values, phat_bernoulli),
            'pr':metrics.average_precision_score(ytest.values, phat_bernoulli)},index=[0]))
    holder_vv.append(pd.concat(holder_metrics).assign(outcome=vv,operyr=tmp_train_years))
    holder_phat.append(pd.concat(holder_score).assign(outcome=vv))

res_cpt = pd.concat(holder_vv).reset_index(drop=True)
res_cpt.insert(0,'tt','cpt')

res_phat = pd.concat(holder_phat).reset_index(drop=True)
res_phat.to_csv(os.path.join(dir_output,'nbayes_phat.csv'),index=False)


####################################################
# ---- STEP 3: LEAVE-ONE-YEAR - ALL VARIABLES ---- #

holder_vv = []
for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii+1, len(cn_Y)))
    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv]==-1],axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv:'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_train_years = tmp_years[tmp_years > tmp_years.min()]
    holder_auc = []
    for yy in tmp_train_years:
        print('Year %i' % (yy))
        idx_train = dat_X.operyr.isin(tmp_years) & (dat_X.operyr < yy)
        idx_test = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy)
        Xtrain, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                        dat_X.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = dat_Y.loc[idx_train,vv].reset_index(drop=True), \
                        dat_Y.loc[idx_test,vv].reset_index(drop=True)
        # --- train model --- #
        mdl_bernoulli = mbatch_NB(method='bernoulli')
        mdl_bernoulli.fit(Xtrain,ytrain.values,mbatch=100000)
        phat_bernoulli = mdl_bernoulli.predict(Xtest)[:,1]
        mdl_gaussian = mbatch_NB(method='gaussian')
        holder_auc.append(metrics.roc_auc_score(ytest.values, phat_bernoulli))
    df_ii = pd.DataFrame({'outcome':vv,'operyr':tmp_train_years, 'auc':holder_auc})
    holder_vv.append(df_ii)

res_all = pd.concat(holder_vv).reset_index(drop=True)
res_all.insert(0,'tt','all')

##############################################
# ---- STEP 4: COMBINE RESULTS AND SAVE ---- #

res_both = pd.concat([res_cpt, res_all],axis=0).reset_index(drop=True)
res_both.to_csv(os.path.join(dir_output,'naivebayes_results.csv'),index=False)

# Calculate percentage
res_pct = res_both.pivot_table('auc',['outcome','operyr'],'tt').reset_index()
res_pct['pct'] = 1 - (res_pct.cpt-0.5) / (res_pct['all']-0.5)
res_pct.pct = np.where(res_pct.pct < 0 , 0 ,res_pct.pct)

g = sns.FacetGrid(data=res_both,col='outcome',col_wrap=5,hue='tt')
g.map(sns.scatterplot,'operyr','auc')
g.add_legend()
g.savefig(os.path.join(dir_figures,'auc_naivebayes1.png'))

g = sns.FacetGrid(data=res_pct,col='outcome',col_wrap=5)
g.map(sns.scatterplot,'operyr','pct')
g.savefig(os.path.join(dir_figures,'auc_naivebayes2.png'))