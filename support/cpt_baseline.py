import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot
from support.naive_bayes import mbatch_NB
from sklearn import metrics

import seaborn as sns

###############################
# ---- STEP 1: LOAD DATA ---- #

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_figures = os.path.join(dir_base,'..','figures')

fn_X = 'X_imputed.csv'
fn_Y = 'y_bin.csv'
dat_X = pd.read_csv(os.path.join(dir_output,fn_X))
dat_Y = pd.read_csv(os.path.join(dir_output,fn_Y))
print(dat_X.shape); print(dat_Y.shape)
stopifnot(all(dat_X.caseid == dat_Y.caseid))
u_years = dat_X.operyr.unique()
# !! ENCODE CPT AS CATEGORICAL !! #
dat_X['cpt'] = 'c'+dat_X.cpt.astype(str)
cn_X = list(dat_X.columns[2:])

cn_Y = list(dat_Y.columns[2:])
missing_Y = dat_Y.melt('operyr',cn_Y)
missing_Y['value'] = (missing_Y.value==-1)
missing_Y = missing_Y.groupby(list(missing_Y.columns)).size().reset_index()
missing_Y = missing_Y.pivot_table(values=0,index=['operyr','variable'],columns='value').reset_index().fillna(0)
missing_Y.columns = ['operyr','cn','complete','missing']
missing_Y[['complete','missing']] = missing_Y[['complete','missing']].astype(int)
missing_Y['prop'] = missing_Y.missing / missing_Y[['complete','missing']].sum(axis=1)
print(missing_Y[missing_Y.prop > 0].sort_values(['cn','operyr']).reset_index(drop=True))
tmp = missing_Y[missing_Y.prop > 0].cn.value_counts().reset_index()
tmp_drop = tmp[tmp.cn > 2]['index'].to_list()
# Remove outcomes missing is two or more years
dat_Y.drop(columns=tmp_drop,inplace=True)
# Remove any Y's that have less than 100 events in 6 yeras
tmp = dat_Y.iloc[:,2:].apply(lambda x: x[~(x==-1)].sum() ,axis=0).reset_index().rename(columns={0:'n'})
tmp_drop = tmp[tmp.n < 100]['index'].to_list()
dat_Y.drop(columns=tmp_drop,inplace=True)
cn_Y = list(dat_Y.columns[2:])

###############################################
# ---- STEP 2: LEAVE-ONE-YEAR - CPT ONLY ---- #

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
        Xtrain, Xtest = dat_X.loc[idx_train, ['cpt']].reset_index(drop=True), \
                        dat_X.loc[idx_test, ['cpt']].reset_index(drop=True)
        ytrain, ytest = dat_Y.loc[idx_train,vv].reset_index(drop=True), \
                        dat_Y.loc[idx_test,vv].reset_index(drop=True)
        # --- train model --- #
        mdl_bernoulli = mbatch_NB(method='bernoulli')
        mdl_bernoulli.fit(Xtrain,ytrain.values,mbatch=100000)
        phat_bernoulli = mdl_bernoulli.predict(Xtest)[:,1]
        holder_auc.append(metrics.roc_auc_score(ytest.values, phat_bernoulli))
    df_ii = pd.DataFrame({'outcome':vv,'operyr':tmp_train_years, 'auc':holder_auc})
    holder_vv.append(df_ii)

res_cpt = pd.concat(holder_vv).reset_index(drop=True)
res_cpt.insert(0,'tt','cpt')

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
# ---- STEP 4: COMBINE RESULTS AND SAVEFFdsdsdfsdfmnjm jmn  mnj mnjd ---- #

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

