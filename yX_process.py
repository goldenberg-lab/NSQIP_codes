# load necessary modules
import numpy as np
import pandas as pd
import os
import gc


from support import support_funs as sf

# set up directories
dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_data = os.path.join(dir_base,'..','data')
dir_figures = os.path.join(dir_base,'..','figures')
sf.stopifnot(all([os.path.exists(x) for x in [dir_output,dir_figures]]))

# manual list of columns to drop

fn ='combined_raw.csv'
# load in the combined data file
dat = pd.read_csv(os.path.join(dir_output, fn),encoding='iso-8859-1')
dat.sort_values(by='operyr',inplace=True)
dat.reset_index(drop=True,inplace=True)
#dat.drop(columns=vv_drop,inplace=True)
gc.collect() # needed!

###################################################
### ---- (1) convert missing values to nas ---- ###

cn = dat.columns # extract columns for later

print('encoding -99 as missing values')
cidx_num = np.where((dat.dtypes=='float64') | (dat.dtypes=='int64'))[0]
for ii, cidx in enumerate(cidx_num):
    if ii % 20 == 0:
        print(ii)
    cc = cn[cidx]
    tmp = np.where(dat.loc[:, cc] == -99, np.nan, dat.loc[:, cc])
    if np.mean(np.isnan(tmp))==0:
        tmp = tmp.astype(int)
    dat.loc[:,cc] = tmp


print('encoding nulls as missing values')
cidx_str = np.where(dat.dtypes=='object')[0]
for ii, cidx in enumerate(cidx_str):
    if ii % 2 == 0:
        print(ii)
    cc = cn[cidx]
    tmp = dat.loc[:, cc].str.lower()
    tmp = np.where(tmp.str.contains('null'),np.nan,tmp)
    dat.loc[:, cc] = tmp

########################################
### ---- (2) summary statistics ---- ###

import seaborn as sns
import matplotlib.pyplot as plt

# missingness by column
holder = []
for jj in range(dat.shape[1]):
    if (jj+1) % 25 == 0:
        print('column %i of %i' % (jj+1, dat.shape[1]) )
    cn_jj = cn[jj]
    if cn_jj == 'operyr':
        continue
    tmp = pd.concat([dat.operyr,dat[cn_jj].isnull()],1).groupby('operyr').mean().reset_index().rename(columns={cn_jj:'mu'})
    tmp.insert(0,'vv',cn_jj)
    holder.append(tmp)
df_missing = pd.concat(holder).reset_index(drop=True)
df_n_operyr = dat.operyr.value_counts().reset_index().rename(columns={'operyr':'n','index':'operyr'})
df_n_operyr['share'] = df_n_operyr.n / df_n_operyr.n.sum()
df_agg_missing = df_missing.merge(df_n_operyr).groupby('vv').apply(lambda x:pd.Series({'x':sum(x['mu']*x['share'])})).reset_index().rename(columns={'x':'mu'})
# remove any columns with 100% missingness
vv_missing_100pct = df_agg_missing[df_agg_missing.mu == 1].vv.to_list()
dat.drop(columns=vv_missing_100pct,inplace=True)
gc.collect()
# columns with all the features
vv_complete = df_agg_missing[df_agg_missing.mu == 0].vv.to_list()
print('a total of %i columns have no missingness' % len(vv_complete))

# find columns with high missiness in one year, and low in another
df_missing_delta = df_missing.groupby('vv').apply(lambda x: pd.Series({'dd':x['mu'].max()-x['mu'].min()})).sort_values('dd',ascending=False).reset_index()
df_missing_delta = df_missing_delta[df_missing_delta.dd > 0].reset_index(drop=True)
vv_dd = df_missing_delta[df_missing_delta.dd > 0.2].vv.to_list()
print('a total of %i columns with high missingness in one year, and low in another' % (len(vv_dd)))

# plot features missingness by year
df_missing_some = df_missing[df_missing.vv.isin(df_agg_missing[df_agg_missing.mu > 0].vv)]
df_missing_some  = df_missing_some[~df_missing_some.vv.isin(vv_missing_100pct)].reset_index(drop=True)
print(df_missing_some)
fig, ax = plt.subplots(figsize=(14,8))
fig = sns.heatmap(df_missing_some[df_missing_some.vv.isin(vv_dd)].pivot('vv','operyr','mu'),yticklabels=vv_dd,ax=ax)
fig.figure.savefig(os.path.join(dir_figures,'missing_delta_large.png'))
plt.close()

# features with consistently low missingness
tmp = df_agg_missing[~df_agg_missing.vv.isin(vv_dd + vv_complete + vv_missing_100pct)].sort_values('mu').reset_index(drop=True)
vv_low = tmp[tmp.mu < 0.4].vv.to_list()
fig = sns.distplot(tmp.mu)
fig.set_xlabel('missing pct')
fig.set_title('distribution of missingness by low delta',size=12)
fig.figure.savefig(os.path.join(dir_figures,'missing_low_delta.png'))
plt.close()

##########################################
### ---- (3) LABEL CATEGORIZATION ---- ###

for ii, rr in df_desc[df_desc.vars.isin(vv_low)].iterrows():
    print('---------- Variable: %s: ----------\n %s' % (rr['vars'], rr['desc']))

# --------------- CONSISTENTLY LOW MISSINGNESS (<20% for max-year) ------------------ #

cn_X_low_impute = ['sex','race','ethnicity_hispanic','height','weight',
                   'crf','prsepis','malignancy']

cn_X_low_drop = ['ivh_grade', 'wndclas', 'anesurg', 'surgane', 'anetime','hdisdt','tothlos','dischdest']

cn_y_low_num = ['dpatrm','optime','doptodis']

cn_y_low_bin = ['death30yn']

# --------------- LOW + HIGH SOME YEARS ------------------ #
# --- LOW + HIGH MISSINGNESS --- #
cn_X_dd_impute = ['diabetes','cpneumon','cystic_fib','lbp_disease',
               'renafail','dialysis','cva','immune_dis','wtloss',
                'bleeddis','lapthor']

cn_X_dd_drop = ['coma','tumorcns','bone_marrow_trans','organ_trans',
                'chemo','radio','proper30', 'cm_icd9_1','gestationalage_birth',
                'admqtr','podiag','podiagtx','days_ventilation',
                'reoperation2', 'reoperation3','readmission2','readmission3','readmission4','readmission5',
                'lapthor_mis','podiag10','podiagtx10','lap_disease','ostomy','pufyear']

cn_y_impute_bin = ['pulembol', 'othgrafl','sdehis','othseshock','othcdiff',
                   'reoperation','readmission1','nutrition_at_discharge','oxygen_at_discharge']

cn_y_impute_num = ['npulembol', 'nothgrafl', 'nsdehis','nothseshock','nothcdiff']

# --------------- 100% NON-MISSING ------------------ #

# Outcome labels (number)
cn_y_num = ['nsupinfec', 'nwndinfd', 'norgspcssi', 'ndehis', 'noupneumo', 'nreintub', 'nrenainsf', 'noprenafl',
            'nurninfec', 'ncnscoma', 'ncnscva', 'nszre', 'nneurodef', 'nivhg1', 'nivhg2', 'nivhg3', 'nivhg4',
            'nivhunk', 'ncdarrest', 'nothbleed', 'nothvt', 'nothsysep', 'nothclab']
cn_y_bin = ['supinfec', 'wndinfd', 'orgspcssi', 'dehis', 'oupneumo', 'reintub', 'renainsf', 'oprenafl',
            'urninfec', 'cnscoma', 'cnscva', 'cszre', 'neurodef', 'civhg1', 'civhg2', 'civhg3', 'civhg4',
            'civhunk', 'cdarrest', 'othbleed', 'othvt', 'othsysep', 'othclab']

# columns for indexing
cn_idx = ['caseid','operyr']

# columns that should not be in x
cn_X_drop = ['admyr','prncptx']

# columns to keep in design matrix
cn_X_keep = ['acq_abnormality','cpt','workrvu','inout', 'transt','age_days',
             'anestech','dnr','ventilat','asthma','hxcld','struct_pulm_ab',
             'esovar', 'prvpcs','impcogstat','seizure','cerebral_palsy',
             'neuromuscdis','steroid','wndinf','hemodisorder',
             'cpr_prior_surg','transfus','casetype','asaclas','surgspec',
             'oxygen_sup', 'tracheostomy','nutr_support','inotr_support',
             'cong_malform','htooday' # intraoperative
            ]


# Make sure the complete variables were properly distributed
tmp1 = pd.Series(cn_y_num + cn_y_bin + cn_idx + cn_X_keep + cn_X_drop)
print(np.setdiff1d(tmp1, vv_complete)); print(np.setdiff1d(vv_complete,tmp1))
sf.stopifnot(len(vv_complete)==len(tmp1)-1)
# Make sure the large delta variables were distributed
tmp2 = pd.Series(cn_X_dd_impute + cn_X_dd_drop + cn_y_impute_bin + cn_y_impute_num)
sf.stopifnot(np.setdiff1d(tmp2,vv_dd).shape[0]==0)
# Make sure low_missingness lines up
tmp3 = pd.Series(cn_X_low_impute + cn_X_low_drop + cn_y_low_num + cn_y_low_bin)
sf.stopifnot(np.setdiff1d(tmp3,vv_low).shape[0]==0)

#########################################
### ---- (4) FEATURE ENGINEERING ---- ###

# (1) Convert dtypes --- #
dat.cpt = dat.cpt.astype(str)

# (2) Force to NAs
dat.sex = np.where(~dat.sex.isin(['male','female']),np.NaN,dat.sex)

# (3) Variables where NaN should be a category
dat.race = np.where(dat.race.isnull(),'unknown/not reported',dat.race)
dat.death30yn = np.where(dat.death30yn.isnull(),'no',dat.death30yn)
dat.prem_birth = np.where(dat.prem_birth.isnull(),'unknown', dat.prem_birth)

# (4) Manual imputation #
dat.asaclas = np.where(dat.asaclas.isin(['none assigned','asa not assigned']),'none',dat.asaclas)
dat.prsepis = dat.prsepis.str.replace('sirs|sepsis|septic\\sshock','sepsis')
dat.malignancy = np.where(dat.malignancy == 'no','no current or prior history of cancer',dat.malignancy)
dat.diabetes = np.where(dat.diabetes.isin(['insulin','non-insulin']),'yes',dat.diabetes)
dat.surgspec = dat.surgspec.str.replace('pediatric\\s|\\s(ent)','')
dat.surgspec = dat.surgspec.str.replace('general\\ssurgery','surgery')
dat.anestech = np.where(dat.anestech=='general','general','non-general')
dat.cong_malform = np.where(dat.cong_malform == 'no','no','yes')

# Column types
df_dtypes = dat.dtypes.reset_index().rename(columns={0:'tt','index':'cc'}).sort_values('tt').reset_index(drop=True)

# X variables we want to keep
cn_X = cn_X_keep + cn_X_low_impute + cn_X_dd_impute
# y variables we want to keep
cn_y = cn_y_bin + cn_y_impute_bin + cn_y_low_bin

# Concert y to binary: 0, 1, and -1 (missing)
for cc in df_dtypes[(df_dtypes.cc.isin(cn_y)) & (df_dtypes.tt=='object')].cc:
    print('-----------------Column X: %s----------------' % cc)
    tmp = dat[cc]
    tmp2 = tmp.value_counts()
    #print('mi: %s, mx: %s' % (tmp2.idxmin(), tmp2.idxmax()) )
    tmp3 = np.where(tmp.isnull(),-1,np.where(tmp == tmp2.idxmin(), 1, 0))
    dat[cc] = tmp3

# Aggregate the intricular hemorrhage
cn_ivh = cn[cn.str.contains('civh')]
dat['civhg'] = np.where(dat[cn_ivh].sum(axis=1)>0,1,0)
cn_y = list(np.append(np.setdiff1d(cn_y,cn_ivh),'civhg'))

##########################################
### ---- (5) SEMANTIC CPT HASHING ---- ###


# ENCODE CPT ANNOTATIONS #
# cpt_df = pd.read_csv(os.path.join(dir_data,'cpt_anno.csv'))
# sf.stopifnot(len(np.setdiff1d(dat.cpt.unique().astype(int),cpt_df.cpt))==0)
# cpt_txt = list(cpt_df.title.str.strip().str.replace('Under\\s|on\\s|the\\s','').unique())


###############################
### ---- (6) SVAE DATA ---- ###

# Save the X and y matrices for later
dat[cn_idx + cn_y].to_csv(os.path.join(dir_output,'y_bin.csv'),index=False)
dat[cn_idx + cn_X].to_csv(os.path.join(dir_output,'X_preop.csv'),index=False)










