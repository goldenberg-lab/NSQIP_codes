
"""
01. extract_combine_data.py
Reads in NSQIP-P PUFs statas (2012 - 2017) and text file (2018), concatenating the individual years together
into a single dataframe and csv, 'combined_raw.csv'.
For each year, also extracts the variable names (columns) and saves into 'yr_vars.csv'
TODO: implement command line version given a root directory...
TODO: also don't really like the use of placeholders because of memory
"""

import os
import re
import pandas as pd
import numpy as np
import sys
import gc

def stopifnot(cond,stmt=None):
    if stmt is None:
        stmt = 'Condition not met!'
    if not cond:
        sys.exit(stmt)

# the root of the nsqip directory
main_dir = os.path.dirname(os.getcwd()) # '/Users/delvin/Documents/OneDrive - SickKids/nsqip/'
# assumes data sits in  'nsqip/data/raw/*'
data_dir = os.path.join(main_dir, 'data')
raw_data_dir = os.path.join(data_dir, 'raw')
stopifnot(os.path.exists(data_dir),'data_dir')
stopifnot(os.path.exists(raw_data_dir),'raw_data_dir')

di_fix = {'anesthes':'anestech', 'deaddate_unk':'death30dtunk',
 'dothsepshock':'dothseshock','nothsepshock':'nothseshock',
 'retopor2icd101': 'reopor2icd101',
 'retopor2icd91': 'reopor2icd91',
 'readsuspreason1': 'readmsuspreason1',
 'readsuspreason2': 'readmsuspreason2',
 'readsuspreason3': 'readmsuspreason3',
 'readsuspreason4': 'readmsuspreason4',
 'readsuspreason5': 'readmsuspreason5'}

#############################################################
# ---- FIND THE COLUMN OVERLAP TO INITIALIZE DATAFRAME ---- #

# Get the filenames
lst_fn = []
for dirname, dirnames, filenames in os.walk(raw_data_dir, topdown = True):
    for filename in filenames:
        if re.search('[0-9]{2}.dta|18.txt$|15_v2.dta', filename): # raw statas and 2018 file
            lst_fn.append(os.path.join(dirname, filename))

#lst_fn = lst_fn[-1:]

vars_ph = []
n_ph = []
# Get the header and data.types
nr = 10
for path in lst_fn:
    fn = path.split('\\')[-1]
    print('File: %s' % fn)
    if fn.endswith('dta'):
        df1 = pd.read_stata(path,chunksize=nr)
        for ii, chunk in enumerate(df1):
            if ii > 0:
                break
            else:
                df1 = chunk
        df2 = pd.read_stata(path,columns=['operyr'])
    else:
        if fn.split('.')[-1]=='txt':
            df1 = pd.read_csv(path,sep='\t',nrows=nr)
            df2 = pd.read_csv(path, sep='\t', usecols=['OPERYR'])
        else:
            df1 = pd.read_csv(path,nrows = nr)
            df2 = pd.read_csv(path, usecols=['OPERYR'])
    df1.columns = df1.columns.str.lower()
    df1.rename(columns=di_fix, inplace=True)
    tmp1 = pd.DataFrame({'cn':df1.columns,'dt':df1.dtypes}).reset_index(drop=True)
    tmp2 = pd.Series({'fn':fn,'n':df2.shape[0]})
    vars_ph.append(tmp1)
    n_ph.append(tmp2)
# Column names and datatypes
df_cn = pd.concat(vars_ph)
df_cn = df_cn.groupby(['cn','dt']).size().reset_index().rename(columns={0:'n'})
df_cn.sort_values(by=['cn','n'],ascending=False,inplace=True)
df_cn['cidx'] = df_cn.groupby('cn').cumcount()
df_cn = df_cn[df_cn.cidx == 0].reset_index(drop=True)
# Number of rows
df_n = pd.concat(n_ph,axis=1).T
df_n = df_n.assign(end = lambda x: x.n.cumsum()).assign(start = lambda x:
    x.end.shift(+1),path=lst_fn).fillna(0)


def df_empty(columns, dtypes, n):
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=range(n))
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(np.repeat(np.NaN,n),dtype=d)
    return df

df = df_empty(df_cn.cn.to_list(),dtypes=df_cn.dt.to_list(),n=df_n.n.sum())
print(df.shape)


########################################
# ---- READ IN THE DATA COMPLETEY ---- #

for ii, rr in df_n.iterrows():
    start, end, path = rr['start'], rr['end'], rr['path']
    fn = path.split('\\')[-1]
    print('File: %s' % fn)
    if fn.endswith('dta'):
        tmp_df = pd.read_stata(path)
    else:
        if fn.split('.')[-1]=='txt':
            tmp_df = pd.read_csv(path,sep='\t')
        else:
            tmp_df = pd.read_csv(path)
    tmp_df.columns = tmp_df.columns.str.lower()
    tmp_df.rename(columns=di_fix, inplace=True)
    print('start: %i, end: %i, nrow1: %i, nrow2: %i' % 
             (start,end, tmp_df.shape[0],df.iloc[start:end].shape[0]))
    stopifnot(df.iloc[start:end].shape[0] == tmp_df.shape[0],'row align')
    stopifnot(tmp_df.columns.isin(df.columns).all(),'column align')
    for cc in tmp_df.columns:
        cidx = np.where(df.columns == cc)[0][0]
        df.iloc[start:end,cidx] = tmp_df[cc].values
    del tmp_df
    gc.collect()

print(df.shape) # (602584, 388)

# ---- save down the files -----
# outdir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'output')
outdir = os.path.join(main_dir, 'output')           # output directory
out_dat = os.path.join(outdir, 'combined_raw.csv')  # the combined, uncleaned data
out_vars = os.path.join(outdir, 'yr_vars.csv')      # year by variable vars
if not os.path.exists(outdir):
    print('Making output directory')
    os.makedirs(outdir)

if not os.path.exists(out_dat):
    print('{} does not exist yet... saving..'.format(out_dat))
    df.to_csv(out_dat, index = False)

# if not os.path.exists(out_vars):
#     print('{} does not exist yet... saving...'.format(out_vars))
#     yr_vars.to_csv(out_vars, index = False)

# NULL can mean Unknown, Did not Occur (eg. for
# -99 can mean either no response or patient did not experience this
# apparently pandas reads these in as null so we gud? (actually not since i'm going to use r for the eda)
