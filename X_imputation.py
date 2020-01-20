# load necessary modules
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split as splitter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

# set up directories
dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_data = os.path.join(dir_base,'..','data')
dir_figures = os.path.join(dir_base,'..','figures')

from support import mdl_funs as mf
from support import support_funs as sf
import acc_funs as af
sf.stopifnot(all([os.path.exists(x) for x in [dir_output,dir_figures]]))

##############################################
### ---- (1) LOAD IN AND PROCESS DATA ---- ###

fn_y = 'y_bin.csv'
fn_X = 'X_preop.csv'
#if fn_X not in os.listdir(dir_output):
#    fn_X = 'X_preop.csv'

y_df = pd.read_csv(os.path.join(dir_output,fn_y))
X_df = pd.read_csv(os.path.join(dir_output,fn_X))
sf.stopifnot( (y_df.shape[0] == X_df.shape[0]) & all(y_df.caseid == X_df.caseid) )
# --- (!) Encode CPT as string --- #
X_df['cpt'] = X_df.cpt.astype(str)

# Get the missingness by column
holder = []
for jj in range(X_df.shape[1]):
    holder.append(pd.Series({'cc':X_df.columns[jj],'nu':X_df.iloc[:,jj].unique().shape[0],
                             'nmiss':X_df.iloc[:,jj].isnull().sum()}))
dat_X_missing = pd.concat(holder,axis=1).T
dat_X_missing[['nu','nmiss']] = dat_X_missing[['nu','nmiss']].astype(int)
dat_X_missing.nu = dat_X_missing.nu - np.where(dat_X_missing.nmiss>0,1,0)
print(dat_X_missing[dat_X_missing.nmiss > 0])

# Define the index columns
cn_idx = ['caseid','operyr']
# Remove from X
dat_X_missing = dat_X_missing[~dat_X_missing.cc.isin(cn_idx)].reset_index(drop=True)
tmp = X_df.dtypes[X_df.dtypes.index.isin(dat_X_missing.cc)].reset_index().rename(columns={'index':'cc',0:'tt'})
dat_X_missing = dat_X_missing.merge(tmp,on=['cc'])
print(dat_X_missing.sort_values('nu'))

########################################
### ---- (2) SIMPLE IMPUTE N<10 ---- ###

cn_simple_impute = dat_X_missing[(dat_X_missing.nmiss >0) & (dat_X_missing.nmiss < 10)].cc

for ii, cc in enumerate(cn_simple_impute):
    print('Simple imputation for column %s (%i of %i)' % (cc, ii+1, len(cn_simple_impute)))
    imp_fac = X_df[cc].value_counts(dropna=True).reset_index().rename(columns={'index': 'vv', cc: 'n'}).loc[0,'vv']
    X_df.loc[:,cc] = np.where(X_df[cc].isnull(),imp_fac,X_df[cc])

cn_complete = dat_X_missing[dat_X_missing.nmiss==0].cc.to_list() + cn_simple_impute.to_list()

cn_mdl_impute = dat_X_missing[(dat_X_missing.nmiss >= 10)].cc
# Find missingness by year
cn_year = [] #
cn_partial = []
dat_mdl_missing = []
for ii, cc in enumerate(cn_mdl_impute):
    tmp = pd.crosstab(X_df.operyr, X_df[cc].isnull())
    tmp = pd.concat([pd.DataFrame({'operyr':tmp.index.values,'cc':cc}),
    pd.DataFrame(tmp.values / tmp.sum(axis=1).values.reshape([tmp.shape[0],1]),columns=['complete','missing'])],axis=1)
    dat_mdl_missing.append(tmp)
    if all(tmp.missing < 1):
        cn_partial.append(cc)
    else:
        cn_year.append(cc)
    # yy_full = ', '.join(tmp[tmp.complete == 1].index.values.astype(str))
    # yy_empty = ', '.join(tmp[tmp.missing == 1].index.values.astype(str))
    # print('Column %s\nYears full: %s\nYears empty: %s\n' % (cc, yy_full, yy_empty))

dat_mdl_missing = pd.concat(dat_mdl_missing).reset_index()

##########################################
### ---- (3) MISSING ACROSS YEARS ---- ###

u_years = X_df['operyr'].unique()

# Data types for the complete columns
cn_complete_cat = dat_X_missing[(dat_X_missing.cc.isin(cn_complete)) & (dat_X_missing.tt == 'object')].cc
cn_complete_num = dat_X_missing[(dat_X_missing.cc.isin(cn_complete)) & ~(dat_X_missing.tt == 'object')].cc

score_partial = []
#ii=0;cc=cn_partial[0]
for ii, cc in enumerate(cn_partial):
    print('Imputation for column %s (%i of %i)' % (cc, ii + 1, len(cn_partial)))
    tmp_y = X_df[cc].copy()
    tmp_idx = np.where(tmp_y.notnull())[0]
    tmp_idx_null = np.where(tmp_y.isnull())[0]
    tmp_tt = tmp_y.dtype
    if tmp_tt == 'object':
        li = LabelEncoder().fit(tmp_y[tmp_y.notnull()])
        tmp_y[tmp_y.notnull()] = li.transform(tmp_y[tmp_y.notnull()])
        tmp_mf = af.auc
        tmp_method = 'bernoulli'
    else:
        tmp_mf = metrics.r2_score
        tmp_method= 'gaussian'
    # Split data
    train_idx, test_idx = splitter(tmp_idx, test_size=0.1, random_state=1234)
    X_train_ii, X_test_ii = X_df.loc[train_idx, cn_complete].reset_index(drop=True), \
                            X_df.loc[test_idx, cn_complete].reset_index(drop=True)
    y_train_ii, y_test_ii = tmp_y[train_idx].values, tmp_y[test_idx].values

    # Fit model
    mdl_ii = mf.mbatch_NB(method=tmp_method)
    mdl_ii.fit(data=X_train_ii,lbls=y_train_ii.astype(int),mbatch=100000)
    score_train_ii = mdl_ii.predict(X_train_ii)
    score_test_ii = mdl_ii.predict(X_test_ii)
    score_impute_ii = mdl_ii.predict(X_df.loc[tmp_idx_null, cn_complete])

    if tmp_tt == 'object':
        score_test_ii = score_test_ii[:, 1] # drop the first class
        score_impute_ii = score_impute_ii[:,1]
        cal_ii = af.plot_auc(y_test_ii,score_test_ii,num=250,figure=False)
        ybar = (y_train_ii.sum()+y_test_ii.sum())/len(tmp_idx)
        cal_ii['prop'] = cal_ii.tpr * ybar + cal_ii.fpr * (1 - ybar)
        # balance yhat==1 to y==1
        thresh_ii = cal_ii.loc[((cal_ii.prop - ybar)**2).idxmin()].thresh
        y_null_ii = li.inverse_transform(np.where(score_impute_ii >= thresh_ii, 1, 0))
        X_df.loc[X_df[cc].isnull(), cc] = y_null_ii
    else:
        X_df.loc[X_df[cc].isnull(), cc] = score_impute_ii

    # --- store accuracy --- #
    store_ii = pd.DataFrame([sf.bs_wrapper(tmp_mf, [y_test_ii[X_df['operyr'][test_idx]==yy],
                            score_test_ii[X_df['operyr'][test_idx]==yy]]) for yy in u_years])
    store_ii.insert(0,'operyr',u_years)
    store_ii.rename(columns={0:'score',1:'lb',2:'ub'},inplace=True)
    all_ii = pd.Series({'operyr':'all'}).append(pd.Series(sf.bs_wrapper(tmp_mf,
                [y_test_ii, score_test_ii]),index=['score','lb','ub']))
    store_ii = store_ii.append(all_ii,ignore_index=True)
    store_ii.insert(0,'metric',pd.Series(str(tmp_mf)).astype(str).str.split(' ')[0][1])
    store_ii.insert(0,'cn',cc)
    print(store_ii)
    score_partial.append(store_ii)

dat_partial = pd.concat(score_partial).reset_index(drop=True)
dat_partial.insert(0,'tt','all_years')
# Save data for later
dat_partial.to_csv(os.path.join(dir_output,'score_partial.csv'),index=False)
X_df.to_csv(os.path.join(dir_output,'X_partial.csv'),index=False)

############################################
### ---- (4) MISSING SPECIFIC YEARS ---- ###

cn_imputed = list(np.union1d(cn_complete, cn_partial))
#print(X_df[cn_imputed].isnull().any().any())

score_year = []
ii, cc = 0, cn_year[0]
for ii, cc in enumerate(cn_year):
    print('---------- Imputation for column %s (%i of %i) -------------' % (cc, ii + 1, len(cn_year)))
    tmp_y = X_df[cc].copy()
    li = LabelEncoder().fit(tmp_y[tmp_y.notnull()])
    tmp_y[tmp_y.notnull()] = li.transform(tmp_y[tmp_y.notnull()])
    # Loop over the years
    tmp_uyears = np.unique(X_df.operyr[tmp_y.notnull()])
    yy_holder = []
    for yy in tmp_uyears[1:]:
        print('Test year: %i' % yy)
        # Split data
        tmp_idx_train = np.where((tmp_y.notnull()) & (X_df.operyr < yy))[0]
        tmp_idx_test = np.where((tmp_y.notnull()) & (X_df.operyr == yy))[0]
        X_train_ii = X_df.loc[tmp_idx_train,cn_imputed].reset_index(drop=True)
        X_test_ii = X_df.loc[tmp_idx_test,cn_imputed].reset_index(drop=True)
        y_train_ii = tmp_y[tmp_idx_train].astype(int).values
        y_test_ii = tmp_y[tmp_idx_test].astype(int).values
        print('Training samples: %i, test samples: %i' % (len(tmp_idx_train), len(tmp_idx_test)))
        mdl_ii = mf.mbatch_NB(method='bernoulli')
        mdl_ii.fit(data=X_train_ii, lbls=y_train_ii, mbatch=100000)
        score_test_ii = mdl_ii.predict(X_test_ii)
        Y_test_ii = OneHotEncoder(sparse=False,dtype='int').fit_transform(y_test_ii.reshape([y_test_ii.shape[0],1]))
        auc_ii = metrics.roc_auc_score(Y_test_ii, score_test_ii)
        precision_ii = af.plot_ppv(Y_test_ii[:, 1], score_test_ii[:, 1], figure=False)
        yy_holder.append(pd.Series({'auc':auc_ii,'precision':precision_ii.precision.mean(),
                            'recall':precision_ii.tpr.mean()}))
        # # Store AUC and inference
        # yy_holder.append(pd.Series(sf.bs_wrapper(af.pairwise_auc,[y_test_ii, score_test_ii],nbs=99),
        #           index=['score', 'lb', 'ub']))
    df_yy = pd.concat(yy_holder,axis=1).T
    df_yy.insert(0,'test_yr',tmp_uyears[1:])
    df_yy.insert(0, 'cn', cc)
    score_year.append(df_yy)

df_year = pd.concat(score_year)
df_year.to_csv(os.path.join(dir_output,'score_year.csv'))
df_year_agg = df_year.drop(columns='test_yr').groupby('cn').mean().reset_index()

# print()
# df_year_long = df_year.melt(['cn','test_yr'],var_name='metric')
# import seaborn as sns
# g = sns.FacetGrid(data=df_year_long,col='cn',row='metric',margin_titles=False)
# g.map(sns.scatterplot,'test_yr','value')

cn_toimpute = df_year_agg[(df_year_agg.precision > 0.1) & (df_year_agg.recall > 0.2)].cn.to_list()

for ii, cc in enumerate(cn_toimpute):
    print('---------- Imputation for column %s (%i of %i) -------------' % (cc, ii + 1, len(cn_toimpute)))
    tmp_y = X_df[cc].copy()
    li = LabelEncoder().fit(tmp_y[tmp_y.notnull()])
    tmp_y[tmp_y.notnull()] = li.transform(tmp_y[tmp_y.notnull()])
    tmp_idx = np.where(tmp_y.notnull())[0]
    tmp_idx_impute = np.where(tmp_y.isnull())[0]
    y_train = tmp_y[tmp_idx].astype(int).values
    X_train = X_df.loc[tmp_idx, cn_imputed].reset_index(drop=True)
    X_impute = X_df.loc[tmp_idx_impute, cn_imputed].reset_index(drop=True)
    mdl_train = mf.mbatch_NB(method='bernoulli')
    mdl_train.fit(data=X_train, lbls=y_train, mbatch=100000)
    yhat_train = mdl_train.predict(X_train)
    yhat_impute = mdl_train.predict(X_impute)

    # Loop over the columns to find the threshold
    ohe_ii = OneHotEncoder(sparse=False, dtype='int').fit(y_train.reshape([y_train.shape[0], 1]))
    Y_train = ohe_ii.transform(y_train.reshape([y_train.shape[0], 1]))
    Y_impute = np.zeros(yhat_impute.shape).astype(int)
    for jj in np.arange(1, Y_train.shape[1]):
        y_jj = Y_train[:,jj]
        ybar_jj = y_jj.mean()
        df_jj = af.plot_auc(lbl=y_jj,score=yhat_train[:,jj],num=250,figure=False)
        df_jj['prop'] = df_jj.tpr*ybar_jj + df_jj.fpr*(1-ybar_jj)
        thresh_jj = df_jj.loc[((df_jj.prop - ybar_jj)**2).idxmin()].thresh
        Y_impute[:,jj] = np.where(yhat_impute[:, jj] > thresh_jj, 1, 0)
    Y_impute[Y_impute.sum(axis=1)==0,0] = 1
    sf.stopifnot(all(Y_impute.sum(axis=1)==1))
    y_impute = li.inverse_transform(ohe_ii.inverse_transform(Y_impute).flatten())
    print(pd.Series(y_impute).value_counts(normalize=True))
    print(pd.Series(tmp_y).value_counts(normalize=True))
    X_df.loc[X_df[cc].isnull(),cc] = y_impute

X_df.to_csv(os.path.join(dir_output,'X_imputed.csv'),index=False)


