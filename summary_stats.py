"""
SCRIPT TO EXPLORE PROPERTIES OF NSQIP DATA
"""

import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

dir_base = os.getcwd()
dir_data = os.path.join(dir_base, '..', 'data')
dir_raw = os.path.join(dir_data, 'raw')
dir_output = os.path.join(dir_base, '..', 'output')
fn_years = np.sort(os.listdir(dir_raw))

pd.options.display.max_rows = 100

######################################################
# ------------ (1) CONVERT .dta to .csv ------------ #

# tmp = []
# for ff in fn_years:
#     print('Folder: %s' % ff)
#     fold = os.path.join(dir_raw, ff)
#     fns = pd.Series(os.listdir(fold))
#     regex = fns.str.contains('.dta$') & ~fns.str.contains('clean')
#     if any(regex):
#         print('converting')
#         fn = fns[regex].to_list()[0]
#         df = pd.read_stata(os.path.join(fold, fn))
#         df.to_csv(os.path.join(fold,fn.replace('dta','csv')))

###############################################
# ------------ (2) LOAD IN DATA  ------------ #

# ---- Load csv or text ---- #
holder = []
for ff in fn_years:
    print('Folder: %s' % ff)
    fold = os.path.join(dir_raw, ff)
    fns = pd.Series(os.listdir(fold))
    reg1 = fns.str.contains('\\.txt$') & ~fns.str.contains('clean')
    reg2 = fns.str.contains('\\.csv$') & ~fns.str.contains('clean')
    if sum(reg1)==1:
        print('extracting txt')
        fn = fns[reg1].to_list()[0]
        df = pd.read_csv(os.path.join(fold, fn), sep='\t')
    elif sum(reg2)==1:
        print('extracting csv')
        fn = fns[reg2].to_list()[0]
        df = pd.read_csv(os.path.join(fold, fn),encoding='ISO-8859-1')
    else:
        print('error!')
    df.columns = df.columns.str.lower()
    holder.append(df)



###############################################################
# ------------ (3) COLUMNS INTERSECTION PATTERN  ------------ #

# --- How many intersecting columns are there? --- #
isec = holder[0].columns
for x in [x.columns for x in holder[1:]]:
    isec = np.intersect1d(isec, x)
print('Total number of intersecting columns: %i' % len(isec))

# --- Tabular count --- #
df_cc = pd.Series(np.concatenate([x.columns for x in holder])).value_counts().reset_index().rename(columns={'index':'cc',0:'n'})

years = pd.Series(fn_years).str.replace('ACS\\sNSQIP\\s','').astype(int)

cc_holder = [list(x.columns) for x in holder]
# Create the count of which datasets are missing the different n values
store = []
for nn in list(np.sort(df_cc.n.unique()[df_cc.n.unique() < 7])):
    cc_nn = df_cc[df_cc.n == nn].cc.to_list()
    tmp = []
    for cc in cc_nn:
        tmp.append(list(years[[cc in x for x in cc_holder]]))
    tmp = pd.Series(np.concatenate(tmp)).value_counts().reset_index().rename(columns={'index':'year',0:'share'})
    tmp.share = tmp.share / len(cc_nn)
    tmp.insert(0,'n',nn)
    store.append(tmp)
dat_isec = pd.concat(store).reset_index(drop=True).pivot(index='year',columns='n',values='share').reset_index().fillna(0)

print('---- MISSINGNESS BY TERMS IS HIGH FOR 2012 ----')
print(np.round(dat_isec,2))

###############################################################
# ----------- (4) Distribution of CPT frequency ------------- #

test = pd.concat([x[['operyr','prncptx']] for x in holder]).reset_index(drop=True)
test2 = test.groupby(['operyr','prncptx']).size().reset_index().rename(columns={0:'n'}).pivot('prncptx','operyr','n').reset_index().melt('prncptx').fillna(0)




# Collect CPT
df_cpt = pd.concat([x[['cpt','operyr']] for x in holder]).reset_index(drop=True)
# Load annotation and hash
df_anno = pd.read_csv(os.path.join(dir_data,'cpt_anno.csv'))
df_anno.title = df_anno.title.str.strip()
u_title = df_anno.title.unique()
di_anno = dict(zip(u_title,np.arange(len(u_title))))
di_anno_rev = dict(zip(list(di_anno.values()), di_anno.keys()))
df_anno.title = df_anno.title.map(di_anno)
di_cpt = dict(zip(df_anno.cpt, df_anno.title))
# Map onto cpt
df_cpt['title'] = [di_cpt[x] for x in df_cpt.cpt]
# Get aggregate counts
df_tot_yr = df_cpt.groupby('operyr').size().reset_index().rename(columns={0:'tot'})
df_tot_cpt = df_cpt.groupby('cpt').size().reset_index().rename(columns={0:'n'})
df_tot_title = df_cpt.groupby('title').size().reset_index().rename(columns={0:'n'})

# --- PLOT 1: CONSISTENCY OF TITLES BY YEAR --- #
dat_title_yr = df_cpt.groupby(['operyr','title']).size().reset_index().rename(columns={0:'n'})
dat_title_yr[dat_title_yr.title == 0 ]

    # .merge(df_tot_yr)


dat_title_yr['share'] = dat_title_yr.n / dat_title_yr.tot
tmp = dat_title_yr.groupby('title').share.var().fillna(0).reset_index().rename(columns={'share':'vv'}).sort_values('vv')
dat_title_yr = dat_title_yr.merge(tmp)
dat_title_yr
pd.Categorical(np.sort(np.array(tmp.title)),tmp.title)

tmp

dat_title_yr.sort_values('vv',ascending=False)
dat_title_yr.

sns.scatterplot(x='title',y='share',hue='operyr',data=dat_title_yr)


# Get CPT code percentage by year
df_cpt_pct = df_cpt.groupby(['operyr','title','cpt']).size().reset_index().rename(columns={0:'n'})
df_cpt_pct = df_cpt_pct.merge(df_tot_yr)
df_cpt_pct['share'] = df_cpt_pct.n / df_cpt_pct.tot
df_cpt_pct = df_cpt_pct.pivot(index='title',columns='operyr',
  values='share').reset_index().fillna(0).melt('title',value_name='share').sort_values('title').reset_index(drop=True)
# Statistics about CPT
title_total = df_cpt_pct.title.unique()
title_miss = df_cpt_pct[df_cpt_pct.share == 0].title.unique()
title_complete = np.setdiff1d(title_total, title_miss)
print('There are total of %i unique title codes, with %i across all years' %
      (title_total.shape[0], title_complete.shape[0]))
# df_cpt_pct = df_cpt_pct.merge(df_tot_yr).rename(columns={'operyr':'year'})
df_cpt_pct = df_cpt_pct.merge(df_tot_title,on='title')
df_cpt_pct['nlog'] = np.log10(df_cpt_pct.n)
df_cpt_pct['complete'] = df_cpt_pct.title.isin(title_complete)
# Save for later
df_cpt_pct[['cpt','year','n']].to_csv(os.path.join(dir_data,'df_cpt_year.csv'),index=False)

holder[6].cpt.isin([11400, 11401, 11402, 11403, 11404, 11406, 11420, 11421, 11422,
       11423, 11424, 11426, 11440, 11441, 11442, 11443, 11444, 11446,
       11450])


df_cpt_pct[df_cpt_pct.complete == False].sort_values('n')

df_cpt_pct[df_cpt_pct.cpt == 43324]

g = sns.FacetGrid(df_cpt_pct, col="complete", sharex=True,sharey=False,margin_titles=True,height=5)
g.map(sns.distplot,'nlog',rug=False)
g.set_xticklabels(10**np.arange(0,6))
g.savefig(os.path.join(dir_output,'CPT_freq_complete.png'))

###########################################################
# ------------ (5) SUBSET IMPORTANT COLUMNS  ------------ #

cc_keep = ['caseid', # unique patient identifier
            'age_days', 'sex', 'race']


print(np.setdiff1d(cc_keep, df_cc[df_cc.n == 7].cc.to_list()))


# read_csv(os.path.join(fold,fn),sep='\t',usecols=['CaseID','AGE_DAYS'])
# print((dat_2018.AGE_DAYS / 365.25).describe())
















