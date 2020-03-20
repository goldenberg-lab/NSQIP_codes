import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot

import seaborn as sns

###############################
# ---- STEP 1: LOAD DATA ---- #

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_figures = os.path.join(dir_base,'..','figures')

# Load the aggregated data
df_y = pd.read_csv(os.path.join(dir_output,'y_agg.csv'))
# Get a slice of the data
Xslice = pd.read_csv(os.path.join(dir_output,'X_imputed.csv'),nrows=10)
# Get all the CPT codes
df_cpt = pd.read_csv(os.path.join(dir_output,'X_imputed.csv'),usecols=['caseid','operyr','cpt'])

#############################################
# ---- STEP 2: PRINT OFF SUMMARY STATS ---- #

u_years = df_cpt.operyr.unique()
di_cpt = dict(zip(u_years,[df_cpt[df_cpt.operyr == yy].cpt.unique() for yy in u_years]))

jac_mat = np.zeros([len(u_years),len(u_years)],dtype='int')
for i1,y1 in enumerate(u_years):
    for i2,y2 in enumerate(u_years):
        jac_mat[i1, i2] = len(np.intersect1d(di_cpt[y1],di_cpt[y2]))
jac_mat = pd.DataFrame(jac_mat,columns=u_years,index=u_years)

g = sns.heatmap(jac_mat,annot=True, fmt='d', cbar=False)
g.set_title('Union of CPT codes by year')
g.figure.savefig(os.path.join(dir_figures,'cpt_union.png'))

####################################

cn_X = np.setdiff1d(Xslice.columns,['caseid','operyr','cpt'])

print('A total of %i features without CPT' % (len(cn_X)))

n_tot = df_y.shape[0]
n_years = df_y.operyr.unique().shape[0]
tab_yy = df_y.operyr.value_counts()

print('A total of %i samples of %i years (average: %i, min: %i, max: %i)' %
      (n_tot,n_years,n_tot/n_years,tab_yy.min(), tab_yy.max()))

cn_Y = df_y.columns[~df_y.columns.str.contains('^agg')][2:]
cn_agg = df_y.columns[df_y.columns.str.contains('^agg')].str.replace('[0-9]','').unique()
print('A total of %i outcomes and %i aggregates' % (len(cn_Y), len(cn_agg)))

# Class balance for aggregate categories
dat_ybar = df_y.iloc[:,2:].apply(lambda x: x[~(x==-1)].mean()).reset_index().rename(columns={0:'n','index':'cn'})
dat_ybar = dat_ybar.assign(ratio = lambda x: np.log(1/x.n))
dat_ybar['cn2'] = pd.Categorical(values=dat_ybar.cn,categories=dat_ybar.sort_values('ratio',ascending=False).cn)
dat_ybar['tt'] = np.where(dat_ybar.cn.isin(cn_Y),'Outcome','Aggregate')


g = sns.catplot(x='cn2',y='ratio',hue='tt',data=dat_ybar,legend_out=False)
g.set_ylabels('log(Outcome per X patients)')
g.set_xlabels('')
g.set_xticklabels(rotation=90)
g.add_legend(title='')
g.fig.tight_layout(pad=2)
g.savefig(os.path.join(dir_figures,'outcome_ratio.png'))

