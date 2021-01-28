import numpy as np
import pandas as pd
import os

def get_cpt_description(cpt_name):

    # ---- STEP 1: LOAD DATA ---- #
    dir_base = os.getcwd()
    dir_output = os.path.join(dir_base, '..', 'output')

    # read in annotation csv
    cpt_anno = pd.read_csv(os.path.join(dir_output, 'cpt_anno.csv'))
    cpt_group = pd.read_csv(os.path.join(dir_output, 'cpt_anno_group.csv'))
    cpt_organ = pd.read_csv(os.path.join(dir_output, 'cpt_anno_organ.csv'))

    # merge cpt_anno and cpt_group by cpt
    cpt_anno_group = pd.merge(cpt_anno, cpt_group, on=['cpt', 'title'], how='right')

    # merge cpt_anno_group with cpt_organ
    cpt_anno_group.drop(['Unnamed: 0'], axis=1, inplace=True)
    cpt_organ.drop(['Unnamed: 0'], axis=1, inplace=True)
    cpt_all = pd.merge(cpt_anno_group, cpt_organ, on=['cpt', 'title'], how='left')

    # code cpt
    cpt_all['cpt'] = 'c' + cpt_all.cpt.astype(str)


    result = cpt_all[cpt_all['cpt']==cpt_name]

    return print('The annotation group is '+str(result.title.values), ' and main group is '+str(result.main_group.values))



get_cpt_description(cpt_name='c69643')
