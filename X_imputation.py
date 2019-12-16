# load necessary modules
import numpy as np
import pandas as pd
import os
import gc
from modelling_funs import stopifnot

# set up directories
dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_figures = os.path.join(dir_base,'..','figures')
stopifnot(all([os.path.exists(x) for x in [dir_output,dir_figures]]))

##############################################
### ---- (1) LOAD IN AND PROCESS DATA ---- ###

fn_y = 'y_bin.csv'
fn_X = 'X_preop.csv'
y_df = pd.read_csv(os.path.join(dir_output,fn_y))
X_df = pd.read_csv(os.path.join(dir_output,fn_X))
stopifnot( (y_df.shape[0] == X_df.shape[0]) & all(y_df.caseid == X_df.caseid) )

# Get the missingness by column
holder = []
for jj in range(X_df.shape[1]):
    holder.append(pd.Series({'cc':X_df.columns[jj],'nu':X_df.iloc[:,jj].unique().shape[0],
                             'nmiss':X_df.iloc[:,jj].isnull().sum()}))
df_X_missing = pd.concat(holder,axis=1).T
df_X_missing[['nu','nmiss']] = df_X_missing[['nu','nmiss']].astype(int)
df_X_missing.nu = df_X_missing.nu - np.where(df_X_missing.nmiss>0,1,0)
print(df_X_missing)

# Define the index columns
cn_idx = ['caseid','operyr']
# Remove from X
df_X_missing = df_X_missing[~df_X_missing.cc.isin(cn_idx)].reset_index(drop=True)
tmp = X_df.dtypes[X_df.dtypes.index.isin(df_X_missing.cc)].reset_index().rename(columns={'index':'cc',0:'tt'})
df_X_missing = df_X_missing.merge(tmp,on=['cc'])
print(df_X_missing.sort_values('nu'))

# # If total CPT is 20 or less, convert to other
# df_cpt = X_df.cpt.value_counts().reset_index().rename(columns={'cpt':'n','index':'cpt'})
# # df_cpt.groupby('n').size().reset_index().rename(columns={0:'count'}).head(40).tail(30)
# cpt_less20 = df_cpt[df_cpt.n < 20].cpt.to_list()
# X_df.cpt = np.where(X_df.cpt.isin(cpt_less20),'other',X_df.cpt)
# df_X_missing.loc[df_X_missing.cc == 'cpt','tt'] = 'object'
# print(df_X_missing)

# Impute with the mode for any factors that are less than 10, including NAs
holder = []
for cc in np.setdiff1d(df_X_missing[(df_X_missing.tt == 'object')].cc,'cpt'):
    print('Column: %s' % cc)
    tmp = X_df[cc].value_counts(dropna=False).reset_index().rename(columns={'index':'vv',cc:'n'})
    tmp.insert(0,'cc',cc)
    holder.append(tmp)
tmp = pd.concat(holder).sort_values('n').reset_index(drop=True)
cn_X_mode = tmp[tmp.n < 10].cc.to_list()

for cc in cn_X_mode:
    print('Column: %s' % cc)
    tmp = X_df[cc]
    X_df[cc] = np.where(tmp.isnull(),tmp.value_counts().idxmax(),tmp)
    df_X_missing.loc[df_X_missing.cc == cc,'nmiss'] = X_df[cc].isnull().sum()

# Remaining columns with Missing values
cn_X_missing = df_X_missing[df_X_missing.nmiss > 0].cc.to_list()

def streaming_OLS(X):
    print(X_df.dtypes)
    return(1)

streaming_OLS(X=X_df)

######################################$########
### ---- (2) PREPARE DATA FOR TRAINING ---- ###

dat_n_operyr = X_df.operyr.value_counts().reset_index().rename(columns={'operyr':'tot','index':'operyr'})

# Get the three column indices: categorical, cpt, numerical
# COME UP WITH SOME GENREAL FUNCTION THAT WILL TAKE THESE THREE INDICES AND RETURN A
#   (1) COUNTVECTORIZER, (2) STANDARDIZER, AND (3) HASHER

# Loop over each of the columns
for ii, cy in enumerate(cn_X_missing):
    tt_ii = np.where(X_df[cy].dtypes == 'object', 'cat', 'num')
    print('Target column: %s (type=%s), %i of %i' % (cy, tt_ii, ii+1, len(cn_X_missing)))
    # Find any years of high missingness
    miss_ii = pd.concat([X_df.operyr,X_df[cy].isnull()],axis=1).groupby(['operyr',cy]).size().reset_index()
    miss_ii = miss_ii.pivot('operyr',cy,0).reset_index().melt('operyr').rename(columns={cy:'missing','value':'n'}).fillna(0)
    miss_ii.n = miss_ii.n.astype(int)
    miss_ii = miss_ii.merge(dat_n_operyr)
    miss_ii['share'] = miss_ii.n / miss_ii.tot
    yy_miss_ii = miss_ii[(miss_ii.missing == True) & (miss_ii.share > 0.2)].operyr
    # Get train/test years
    yy_available = np.sort(np.setdiff1d(dat_n_operyr.operyr,yy_miss_ii))
    yy_test = yy_available[-1:]
    yy_train = yy_available[:-1]
    # Get the column names to train with
    cX = np.setdiff1d(df_X_missing[df_X_missing.nmiss == 0].cc,cy)
    tt_X = X_df.loc[0:1,cX].dtypes # Data-types




    cX_obj = cX[np.where(tt_X == 'object')[0]]
    cX_num = np.setdiff1d(cX, list(cX_obj) + ['cpt'] )






# We will hash the CPT codes
from sklearn.feature_extraction import FeatureHasher
cpt_hasher = FeatureHasher(n_features=25,input_type='string')


np.random.binomial(1,0.5,100).reshape([20,5])







################################
### ---- (5) IMPUTATION ---- ###

# --- (i) Calculating missingness by column --- #

# --- (ii) If <10 missing, impute with median/mode --- #

# --- (iii) Get accuracy for the LOYO --- #