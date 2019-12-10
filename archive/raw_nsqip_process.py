# load the necessary modules
import os
import pandas as pd
import numpy as np

dir_base = os.getcwd()
dir_data = os.path.join(dir_base, '..', 'data')
dir_raw = os.path.join(dir_data, 'raw')
dir_output = os.path.join(dir_base, '..', 'output')

######################################################
# ------------ (1) CONVERT .dta to .csv ------------ #

fn_years = np.sort(os.listdir(dir_raw))
yy_years = pd.Series([x.replace('ACS NSQIP ','') for x in fn_years]).astype(int)

tmp = []
for ff in fn_years:
    print('folder: %s' % ff)
    fold = os.path.join(dir_raw, ff)
    fns = pd.Series(os.listdir(fold))
    regex = fns.str.contains('.dta$|.txt$') & ~fns.str.contains('clean')
    if any(regex):
        fn = fns[regex].to_list()[0]
        if any(fns.str.contains('.csv$')):
            continue
        if fn.__contains__('.dta'):
            print('Convert DTA file')
            df = pd.read_stata(os.path.join(fold, fn))
        elif fn.__contains__('.txt'):
            print('Concerting .txt file')
            df = pd.read_csv(os.path.join(fold, fn),sep='\t')
        else:
            print('Error! How did we get here?')
        # Save file
        print('Saving file')
        df.to_csv(os.path.join(fold,fn.split('.txt')[0]+'.csv'))


############################################
# ------------ (2) LOAD FILES ------------ #

# Apply manual conversion
di_fix = {'anesthes':'anestech', 'deaddate_unk':'death30dtunk',
 'dothsepshock':'dothseshock','nothsepshock':'nothseshock',
 'retopor2icd101': 'reopor2icd101',
 'retopor2icd91': 'reopor2icd91',
 'readsuspreason1': 'readmsuspreason1',
 'readsuspreason2': 'readmsuspreason2',
 'readsuspreason3': 'readmsuspreason3',
 'readsuspreason4': 'readmsuspreason4',
 'readsuspreason5': 'readmsuspreason5'}

holder = []
for ff in fn_years:
    print('Folder: %s' % ff)
    fold = os.path.join(dir_raw, ff)
    fns = pd.Series(os.listdir(fold))
    reg = fns.str.contains('\\.csv$') & ~fns.str.contains('clean')
    if sum(reg)==1:
        print('extracting csv')
        fn = fns[reg].to_list()[0]
        df = pd.read_csv(os.path.join(fold, fn),encoding='ISO-8859-1')
    else:
        print('error!')
    df.columns = df.columns.str.lower()
    df.rename(columns=di_fix,inplace=True)
    holder.append(df)

# # Find column intersection and differences
# df_cc = pd.concat([pd.DataFrame({'cc':dd.columns,'tt':yy}) for (dd, yy) in zip(holder, yy_years)])
# df_cc = df_cc[~df_cc.cc.str.contains('unnamed')].reset_index(drop=True)
# df_cc_n = df_cc.groupby('cc').size().reset_index().rename(columns={0:'n'}).sort_values('n',ascending=False)
# print(df_cc_n.n.value_counts())
# # Look at columns with less than perfect equality
# for nn in -np.sort(-np.setdiff1d(df_cc_n.n.unique(),df_cc_n.n.max())):
#     print('----------- %i of %i -----------' % (nn, df_cc_n.n.max()))
#     tmp1 = df_cc_n[df_cc_n.n == nn]
#     tmp2 = df_cc[df_cc.cc.isin(tmp1.cc)].reset_index().pivot('cc','tt','index').reset_index().melt('cc')
#     tmp3 = tmp2[tmp2.value.notnull()].groupby('cc').tt.apply(lambda x: np.setdiff1d(yy_years,x))
#     print(tmp3)
#
# # import spacy
# # nlp = spacy.load("en_core_web_md")
# # doc = nlp(' '.join(all_cc))
#
# # Function to combine to words
# def wjoin(vec):
#     return(vec.sort_values().str.cat(sep='-'))
# import Levenshtein as lev
# # Some column names are very similar: anestech vs anesthes
# all_cc = df_cc.cc.unique()
# di_cc = dict(zip(range(len(all_cc)),all_cc))
# cc_missing = list(df_cc_n[df_cc_n.n < df_cc_n.n.max()].cc)
# # Create similarity matrix
# sim_cc = np.ones([len(all_cc),len(all_cc)]) * np.NAN
# for ii in range(len(all_cc)-1):
#     for jj in range(ii+1,len(all_cc)): #print('i: %i, j: %i' % (ii, jj))
#         s_ii, s_jj = all_cc[ii], all_cc[jj]
#         sim_cc[jj, ii] = lev.distance(s_ii,s_jj)/max(len(s_ii),len(s_jj))
# # loop through and calculate the year overlap
# df_tt = df_cc[df_cc.cc.isin(cc_missing)].reset_index().pivot('cc','tt','index').reset_index()
# df_tt.iloc[:,1:] = np.where(df_tt.iloc[:,1:].isnull(),0,1)
# df_tt2 = pd.DataFrame(df_tt.iloc[:,1:].values.dot(df_tt.iloc[:,1:].T.values),index=df_tt.cc,columns=df_tt.cc).reset_index().rename(columns={'cc':'var1'}).melt('var1').rename(columns={'cc':'var2'})
# df_tt2 = df_tt2[df_tt2.value == 0].reset_index(drop=True)
# df_tt2['v1v2'] = df_tt2.drop(columns='value').apply(wjoin,1)
#
# # long format
# df_sim = pd.DataFrame(sim_cc).reset_index().melt('index')
# df_sim = df_sim[df_sim.value.notnull()].reset_index(drop=True).rename(columns={'index':'var1','variable':'var2','value':'dist'})
# df_sim.var1 = df_sim.var1.map(di_cc)
# df_sim.var2 = df_sim.var2.map(di_cc)
# df_sim = df_sim[(df_sim.var1.isin(cc_missing)) | (df_sim.var2.isin(cc_missing))].reset_index(drop=True)
# df_sim6 = df_sim[df_sim.dist < 0.6].reset_index(drop=True)
# df_sim6['v1v2'] = df_sim6.drop(columns='dist').apply(wjoin,1)
# # Candidate column
# df_cc_cand = df_sim6[['v1v2','dist']].merge(df_tt2[['v1v2']],on='v1v2',how='inner')
# df_cc_cand = df_cc_cand[~df_cc_cand.duplicated()].sort_values('v1v2').reset_index(drop=True)
# df_cc_cand = pd.concat([df_cc_cand.v1v2.str.split('-',expand=True).rename(columns={0:'v1',1:'v2'}),
#            df_cc_cand.drop(columns='v1v2')],axis=1)
# df_tt_lst = df_cc[df_cc.cc.isin(cc_missing)].groupby('cc').tt.apply(list).reset_index()
# di_tt_lst = dict(zip(df_tt_lst.cc, df_tt_lst.tt))
# df_cc_cand.insert(1,'v1_tt',df_cc_cand.v1.map(di_tt_lst))
# df_cc_cand.insert(3,'v2_tt',df_cc_cand.v2.map(di_tt_lst))
# # Loop through and print
# for ii, rr in df_cc_cand.iterrows():
#     print(rr)

############################################################
# ------------ (3) COLUMN CONSISTENCY BY YEAR ------------ #




