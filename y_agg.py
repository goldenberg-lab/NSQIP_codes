import numpy as np
import pandas as pd
import os

from support import support_funs as sf

# set up directories
dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
sf.stopifnot(all([os.path.exists(x) for x in [dir_output]]))


di_lbls = {'cdarrest':'cardiac arrest',
            'cnscva':'CVA, stroke or hemorrhage',
            'cszre':'seizure',
            'civhg':'intraventricular hemorrhage',
            'death30yn':'death in 30 days',
            'dehis':'deep wound disruption',
            'neurodef':'nerve injury',
            'oprenafl':'acute renal failure',
            'orgspcssi':'organ SSI',
            'othbleed':'bleeding or transfusion',
            'othclab':'central line infection',
            'othseshock':'septic shock',
            'othsysep':'sepsis',
            'othvt':'ventricular tachycardia',
            'oupneumo':'pneumonia',
            'readmission1':'unplanned readmission',
            'reintub':'unplanned reintubation',
            'renainsf':'renal insufficiency',
            'reoperation':'unplanned repoeration',
            'sdehis':'superficial wound disruption',
            'supinfec':'occurrences superficial incisional SSI',
            'urninfec':'urinary tract infection',
            'wndinfd':'deep incisional SSI'}

di_agg = {'nsi1':['othsysep','othseshock'], # non-site infection
          'nsi2':['othsysep','othseshock','urninfec'],
          'nsi3':['othsysep','othseshock','urninfec','othclab'],
          'nsi4':['oupneumo','urninfec','othclab','othsysep','othseshock'],
          'ssi1':['supinfec','wndinfd','orgspcssi'],
          'ssi2':['supinfec','wndinfd','orgspcssi','sdehis','dehis'],
          'aki':['renainsf', 'oprenafl'],
          'adv1':['death30yn','renainsf', 'oprenafl','cdarrest'],
          'adv2':['death30yn','reintub', 'reoperation','cdarrest','cnscva'],
          'unplan1':['readmission1','reoperation'],
          'unplan2':['readmission1','reoperation','reintub'],
          'cns':['cnscva','cszre','civhg']}

################################################################
# ------ (1) LOAD DATA AND REMOVE OVERLY MISSING LABELS ------ #

# load in the processed labels
dat = pd.read_csv(os.path.join(dir_output, 'y_bin.csv'))
cn_Y = list(dat.columns[2:])

missing_Y = dat.melt('operyr',cn_Y)
missing_Y['value'] = (missing_Y.value==-1)
missing_Y = missing_Y.groupby(list(missing_Y.columns)).size().reset_index()
missing_Y = missing_Y.pivot_table(values=0,index=['operyr','variable'],columns='value').reset_index().fillna(0)
missing_Y.columns = ['operyr','cn','complete','missing']
missing_Y[['complete','missing']] = missing_Y[['complete','missing']].astype(int)
missing_Y['prop'] = missing_Y.missing / missing_Y[['complete','missing']].sum(axis=1)
#print(missing_Y[missing_Y.prop > 0].sort_values(['cn','operyr']).reset_index(drop=True))
tmp = missing_Y[missing_Y.prop > 0].cn.value_counts().reset_index()
tmp_drop = tmp[tmp.cn > 2]['index'].to_list()
# Remove outcomes missing is two or more years
print('Dropping columns: %s (>2 years of >0%% missing)' % ', '.join(tmp_drop))
dat.drop(columns=tmp_drop,inplace=True)
# Remove any Y's that have less than 100 events in 6 years
tmp = dat.iloc[:,2:].apply(lambda x: x[~(x==-1)].sum() ,axis=0).reset_index().rename(columns={0:'n'})
tmp_drop = tmp[tmp.n < 100]['index'].to_list()
print('Dropping columns: %s (<100 events in 6 years)' % ', '.join(tmp_drop))
dat.drop(columns=tmp_drop,inplace=True)
cn_Y = list(dat.columns[2:])

sf.stopifnot(len(np.setdiff1d(cn_Y,pd.Series(list(di_lbls.keys()))))==0)

##################################################
# ------ (2) DEFINE THE LABEL AGGREGATORS ------ #



# Loop through
for tt in list(di_agg.keys()):
    sf.stopifnot(len(np.setdiff1d(di_agg[tt],dat.columns))==0)
    dat.insert(dat.shape[1],'agg_'+tt,
    np.where(np.where(dat.loc[:,dat.columns.isin(di_agg[tt])]==1,1,0).sum(axis=1)==0,0,1))
# Save
print('writing file to csv')
dat.to_csv(os.path.join(dir_output, 'y_agg.csv'),index=False)














