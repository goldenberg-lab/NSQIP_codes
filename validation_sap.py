"""
STATISTICAL ANALYSIS PLAN FOR SSI
i) Compare SSI @SK compared to @NSQIP
-> Raw vs Propensity score adjusted rates
-> CPT codes
ii) How many CPTs do we have "tailored" models for? How many do we have baseline risk for? (As in how many using SK data)
iii) What threshold do we need for a 5% PPV? Show the lower-bound (BCA). What sensitivity/PPV/FPR would we have obtained on SK?
iv) Power analysis
"""

import os
import dill
import pickle
from joblib import dump, load
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection as fdr_corr
from support.stats_funs import gen_CI, auc2se, bootstrap_x
from support.acc_funs import fast_auc, strat_bs_auc
from support.support_funs import find_dir_nsqip, gg_save, gg_color_hue, cvec
from support.dict import di_agg
from scipy import stats
import plotnine
from plotnine import *
from plydata.cat_tools import *
from statsmodels.stats.proportion import proportion_confint as propCI

# Set directories
dir_NSQIP = find_dir_nsqip()
dir_figures = os.path.join(dir_NSQIP, 'figures')
dir_output = os.path.join(dir_NSQIP, 'output')
dir_models = os.path.join(dir_output, 'models')
dir_ssi = os.path.join(dir_output, 'ssi')
lst_dir = [dir_figures, dir_output, dir_models, dir_ssi]
assert all([os.path.exists(fold) for fold in lst_dir])

di_outcome = {'adv':'ADV', 'aki':'AKI', 'cns':'CNS',
              'nsi':'nSSIs', 'ssi':'SSIs', 'unplan':'UPLN'}

#############################
# ----- (1) LOAD DATA ----- #

# (i) Load SK predicted risk scores
df_sk = pd.read_csv(os.path.join(dir_output,'dat_sk_score.csv'))
df_sk = df_sk.query('outcome == "ssi"').reset_index(None, True).drop(columns='outcome')
df_sk.insert(0,'ds','sk')

# (ii) Load NSQIP predicted risk scores
df_nsq = pd.read_csv(os.path.join(dir_output, 'best_eta.csv'))
df_nsq = df_nsq.query('outcome == "ssi"').reset_index(None, True).drop(columns='outcome')
# The method/version/model are irrelavant as these are the "winning"
df_nsq.drop(columns=['model','method','version'],inplace=True)
df_nsq.caseid = df_nsq.caseid.astype(int)
df_nsq.insert(0,'ds','nsq')

# (iii) Low caseid/cpt dictionary
df_idt = pd.read_csv(os.path.join(dir_output, 'X_imputed.csv'), usecols=['caseid','cpt'])
df_idt.cpt = 'c'+ df_idt.cpt.astype(str)

# (iv) Load the NaiveBayes baseline
df_nb = pd.read_csv(os.path.join(dir_output, 'nbayes_phat.csv'))
# Map to the aggregate (best SSI is version 1)
cn_ssi = di_agg['ssi1']
cn_idx = ['operyr','caseid']
df_nb = df_nb[df_nb.outcome.isin(cn_ssi)]
df_nb = df_nb.pivot_table(['y','phat'],cn_idx,'outcome')
df_nb = df_nb['phat'].sum(1).reset_index().merge(df_nb['y'].sum(1).clip(0,1).reset_index(),'left',cn_idx)
df_nb.rename(columns={'0_x':'phat','0_y':'y'}, inplace=True)
# Add on CPT
df_nb = df_nb.merge(df_idt,'left','caseid')
# Append NB phats to NSQIP
df_nsq = df_nsq.merge(df_nb[['caseid','cpt','phat']],'left').rename(columns={'phat':'nb'})

# (v) Load in the feature name dictionary
df_cn = pd.read_csv(os.path.join(dir_output,'bhat_desc.csv'),usecols=['cn','desc'])

# (vi) Load coefficient/VI results
di_cn = {'feature_names':'cn', 'feature_importance':'bhat', 
'coef':'bhat', 'p_val':'pval','outcome':'y',
'sig':'is_sig', 'feature_pval_rank':'rnk', 'feature_importance_rank':'rnk'}
tmp1 = pd.read_csv(os.path.join(dir_ssi, 'logit_ssi1.csv'))
tmp2 = pd.read_csv(os.path.join(dir_ssi, 'logit_ssi2.csv'))
logit_ssi = pd.concat([tmp1,tmp2]).drop(columns=['Unnamed: 0'],errors='ignore')
logit_ssi = logit_ssi.reset_index(None,True).rename(columns=di_cn).assign(mdl='logit')
logit_ssi.pval = fdr_corr(logit_ssi.pval)[1]
logit_ssi.is_sig = logit_ssi.pval < 0.05
tmp1 = pd.read_csv(os.path.join(dir_ssi, 'xgb_ssi1.csv'))
tmp2 = pd.read_csv(os.path.join(dir_ssi, 'xgb_ssi2.csv'))
xgb_ssi = pd.concat([tmp1,tmp2]).drop(columns=['Unnamed: 0'],errors='ignore')
xgb_ssi = xgb_ssi.reset_index(None,True).rename(columns=di_cn).assign(mdl='xgb')
bhat_ssi = pd.concat([logit_ssi, xgb_ssi]).reset_index(None,True)
bhat_ssi = bhat_ssi.assign(y=lambda x: x.y.str.replace('agg_',''),)
bhat_ssi.is_sig = bhat_ssi.is_sig.ifllna(False)
# Match up the columns
assert len(np.setdiff1d(bhat_ssi.cn.unique(),df_cn.cn)) == 0
bhat_ssi = bhat_ssi.merge(df_cn,'left','cn')

# (vii) Load in the NB model
path_mdl = os.path.join(dir_models,'NB_cpt_agg_ssi1.pickle')
assert os.path.exists(path_mdl)
with open(path_mdl, 'rb') as file:
   mdl_nb = dill.load(file)
df_sk = df_sk.assign(nb=mdl_nb.predict(df_sk[['cpt']])[:,1])

# (viii) Load in the CPT definitions
tmp0 = pd.read_csv(os.path.join(dir_output,'cpt_anno.csv'))
tmp0.title = tmp0.title.str.strip().str.replace('Under\\s','').str.split('\\(',1,True).iloc[:,0]
tmp1 = pd.read_csv(os.path.join(dir_output,'cpt_anno_group.csv'),usecols=['cpt','main_group']).rename(columns={'main_group':'grp'})
tmp1.grp = tmp1.grp.str.replace('Surgical Procedures on the ','').str.replace(' System','')
tmp2 = pd.read_csv(os.path.join(dir_output,'cpt_anno_organ.csv'),usecols=['cpt','organ'])
di_cpt = tmp0.merge(tmp1).merge(tmp2)
di_cpt.cpt = 'c' + di_cpt.cpt.astype(str)

# (ix) Load in the within-CPT AUROCs for best model
best_mdl = pd.read_csv(os.path.join(dir_output,'best_mdl.csv'))
auc_cpt_nsq = pd.read_csv(os.path.join(dir_output,'df_within_cpt.csv'))
auc_cpt_nsq = auc_cpt_nsq.merge(best_mdl.query('outcome=="ssi"')).assign(ds='nsq')
auc_cpt_nsq.drop(columns=['version','method','r_s','outcome','model'],inplace=True)
auc_cpt_nsq.rename(columns={'g':'cpt'},inplace=True)

# Check that CPTs overlap
cpt_sk = pd.Series(df_sk.cpt.unique())
cpt_nsq = pd.Series(df_nsq.cpt.unique())
cpt_nb = pd.Series(df_nb.cpt.unique())
assert cpt_sk.isin(cpt_nb).all()  # All CPT codes should be in here
print('SK data has %i unique CPT codes, NSQIP has %i' % (len(cpt_sk),len(cpt_nb)))

# Get number of observations for later
dat_n = pd.DataFrame({'ds':['SK','NSQIP'],'n':[len(df_sk),len(df_nsq)]})

########################################
# ----- (2) STRATIFIED BOOTSTRAP ----- #

# Q1: How many CPTs have an outcome?
cpt_sk_n = df_sk.groupby(['cpt','y']).size().reset_index().pivot('cpt','y',0).fillna(0).astype(int)
cpt_sk_n.columns = ['y0','y1']
cpt_sk_n = cpt_sk_n.query('y0>0 & y1>0').sort_values('y1',ascending=False).reset_index()
print('Only %i CPTs have at least one y==1 (out of %i)' % (cpt_sk_n.shape[0], len(cpt_sk)))
# Keep only those with 30 pairwise obs and y>=2
cpt_sk_n = cpt_sk_n.query('y0*y1>=30 & y1>=2').reset_index(None, True)
print('Only %i CPTs have at least 30 pairwise obs' % (cpt_sk_n.shape[0]))
print(cpt_sk_n)
cpt_ssi = cpt_sk_n.cpt

n_bs = 1000
alpha = 0.05
holder = []
for cpt in cpt_ssi:
    print('CPT: %s' % cpt)
    tmp_sk = df_sk.query('cpt==@cpt')
    tmp_nsq = df_nsq.query('cpt==@cpt')
    tmp_bounds = np.c_[strat_bs_auc(y=tmp_sk.y,score=tmp_sk.preds,n_bs=n_bs,alpha=alpha),
                       strat_bs_auc(y=tmp_nsq.y,score=tmp_nsq.preds,n_bs=n_bs,alpha=alpha)].T
    tmp_bounds = pd.DataFrame(tmp_bounds,columns=['lb','ub']).assign(ds=['sk','nsq'])
    tmp_auc = np.array([fast_auc(tmp_sk.y, tmp_sk.preds),fast_auc(tmp_nsq.y, tmp_nsq.preds)])
    tmp_bounds = tmp_bounds.assign(auc=tmp_auc, cpt=cpt, y=tmp_sk.y.sum())
    holder.append(tmp_bounds)
# Q2: How does the within AUROC compare?
auc_cpt = pd.concat(holder).merge(di_cpt)
auc_cpt = auc_cpt.assign(title=lambda x: cat_reorder(x.title+' ('+x.y.astype(str)+')',x.y))

# Plot it
posd = position_dodge(0.5)
gg_auc_cpt_sk = (ggplot(auc_cpt,aes(x='title',y='auc',color='ds')) + theme_bw() + 
    geom_point(position=posd) + 
    geom_linerange(aes(ymin='lb',ymax='ub'),position=posd) + 
    labs(x='Surgery procedure',y='AUROC') + 
    ggtitle('Linerange shows 95% CI') + 
    scale_color_discrete(name='Dataset',labels=['NSQIP','SK']) + 
    coord_flip() + 
    theme(axis_title_y=element_blank(),legend_position=(0.5,-0.01),
          legend_direction='horizontal') + 
    scale_y_continuous(limits=[0,1]) + 
    geom_hline(yintercept=0.5,linetype='--'))
gg_save('gg_auc_cpt_sk.png',dir_figures,gg_auc_cpt_sk,5,5)

########################################
# ----- (3) SSI RATE BY CATEGORY ----- #

cn = ['ds','cpt','y']
dat_n_organ = pd.concat([df_sk[cn],df_nsq[cn]]).merge(di_cpt).melt(cn,['grp','organ'],'tt').pivot_table('cpt',['ds','tt','value'],'y','size').fillna(0).astype(int).rename(columns={0:'y0',1:'y1'}).assign(tot=lambda x: x.y0+x.y1).reset_index()
tmp = pd.DataFrame(np.c_[propCI(count=dat_n_organ.y1,nobs=dat_n_organ.tot,alpha=alpha,method='beta')],columns=['lb','ub'])
dat_n_organ = pd.concat([dat_n_organ,tmp],1).assign(prop=lambda x: x.y1 / x.tot)
# Remove if missing in one dataset
tmp = dat_n_organ.groupby('value').size()
tmp = tmp[tmp==1].index.values
dat_n_organ = dat_n_organ.query('~value.isin(@tmp)')
# Remove cases where SK==0
tmp = dat_n_organ.query("y1==0").value.values
dat_n_organ = dat_n_organ.query('~value.isin(@tmp)').reset_index(None, True)
# Set order based on SK
dat_n_organ = dat_n_organ.assign(value=lambda x: cat_reorder(x.value,x.ub))

# Plot the proportion
posd = position_dodge(0.5)
di_tt = {'grp':'System','organ':'Organ'}
gg_event_prop_sk = (ggplot(dat_n_organ,aes(x='value',y='prop',color='ds')) + 
    theme_bw() + geom_point(position=posd) + 
    geom_linerange(aes(ymin='lb',ymax='ub'),position=posd) + 
    facet_wrap('~tt',labeller=labeller(tt=di_tt),scales='free_y') + 
    ggtitle('Linerange shows 95% CI') + 
    labs(y='SSI event %') + 
    theme(axis_title_y=element_blank(),legend_position=(0.5,0.01),
          legend_direction='horizontal',subplots_adjust={'wspace': 0.35}) + 
    coord_flip() + geom_hline(yintercept=0) + 
    scale_color_discrete(name='Dataset',labels=['NSQIP','SK']))
gg_save('gg_event_prop_sk.png',dir_figures,gg_event_prop_sk,8,8)

###################################################
# ----- (4) SSI RATE (PARAMETRIC VS ACTUAL) ----- #

# For y's, use the binomial CI
cn = ['ds','y']
tmp_y = pd.concat([df_nsq[cn], df_sk[cn]]).groupby('ds').apply(lambda x: pd.Series({'prop':x.y.mean(),'tot':len(x)})).reset_index().assign(mdl='y')
cn = ['ds','preds','nb']
tmp_p = pd.concat([df_nsq[cn], df_sk[cn]]).melt('ds',None,'mdl').groupby(['ds','mdl']).apply(lambda x: pd.Series({'prop':x.value.mean(),'tot':len(x)})).reset_index()
dat_calib = pd.concat([tmp_y, tmp_p]).reset_index(None, True)
dat_calib = dat_calib.assign(se=lambda x: np.sqrt(x.prop*(1-x.prop)/x.tot)).drop(columns='tot')
dat_calib = dat_calib.assign(lb=lambda x: x.prop-crit95*x.se, ub=lambda x: x.prop+crit95*x.se)
di_ds = {'nsq':'NSQIP','sk':'SK'}
di_mdl = {'y':'Actual', 'nb':'NaiveBayes', 'preds':'XGBoost'}
dat_calib = dat_calib.assign(ds = lambda x: x.ds.map(di_ds), mdl = lambda x: x.mdl.map(di_mdl))

posd = position_dodge(0.5)
gg_calib_ssi = (ggplot(dat_calib,aes(x='ds',y='prop',color='mdl')) + 
    theme_bw() + 
    geom_point(size=2,position=posd) + 
    geom_linerange(aes(ymin='lb',ymax='ub'),position=posd) + 
    labs(y='SSI event %') + 
    theme(axis_title_x=element_blank(),legend_position=(0.35,0.70)) + 
    ggtitle('Linerange shows 95% CI') + 
    scale_y_continuous(limits=[0,0.05]) + 
    scale_color_discrete(name='Outcome estimate'))
gg_save('gg_calib_ssi.png',dir_figures,gg_calib_ssi,6,5)

# Are the differences statistically significant?
tmp = df_sk.melt(['caseid','y'],['preds','nb'],'mdl','score')

cn = ['y','preds','nb']
np.random.state(1234)
holder = []
for i in range(2000):
    holder.append(df_sk.sample(n=len(df_sk),replace=True,random_state=i)[cn].sum(0).reset_index().assign(idx=i))
# Merge and calculate
dat_bs_yd = pd.concat(holder).rename(columns={0:'yhat','index':'mdl'})
tmp1 = dat_bs_yd.query('mdl=="y"').rename(columns={'yhat':'y'}).drop(columns='mdl')
dat_bs_yd = dat_bs_yd.query('mdl!="y"').merge(tmp1).assign(yd=lambda x: x.y - x.yhat, mdl=lambda x: x.mdl.map(di_mdl))
tmp2 = dat_bs_yd.groupby('mdl').yd.quantile(alpha/2).reset_index()
# Plot the difference
gg_ydiff_sk = (ggplot(dat_bs_yd,aes(x='yd',fill='mdl')) + theme_bw() + 
    labs(x='Difference in # of SSI events',y='Bootstrap frequency') + 
    ggtitle('Variation comes from bootstrap\nVertical lines shows 2.5% quantile') + 
    scale_fill_discrete(name='Model') + 
    geom_histogram(bins=30,position='identity',alpha=0.3,color='black') + 
    geom_vline(aes(xintercept='yd',color='mdl'),data=tmp2,inherit_aes=False,size=1.5) + 
    guides(color=False) + geom_vline(xintercept=0,size=1.5))
gg_save('gg_ydiff_sk.png',dir_figures,gg_ydiff_sk,6,5)

#########################################
# ----- (5) THRESHOLD CALIBRATION ----- #

# LOAD NSQIP RESULTS
# APPLY THE NP-UMBRELLA TO GET THE LOWER-BOUND...
# SHOW THE REAL-TIME PERFORMANCE AND THE REJECTION BOUNDS, WHEN THAT HAPPENS
# WHETHER THIS CORRESPONDS TO THE POWER ANALYSIS

###########################
# ----- (X) FIGURES ----- #

# (i) Significant coefficient by SSI
tmp = bhat_ssi.query('mdl=="logit"').assign(desc=lambda x: cat_reorder(x.desc, x.bhat))
posd = position_dodge(0.5)
xx = np.ceil(tmp.bhat.abs().max()*10)/10
gg_bhat_logit = (ggplot(tmp, aes(y='desc',x='bhat',color='y',alpha='is_sig.astype(str)')) + 
    theme_bw() + geom_point(position=posd) + 
    theme(axis_title_y=element_blank(), legend_position=(0.7,0.20)) + 
    labs(x='Coefficient') + 
    ggtitle('Highlighted points have FDR<5%') + 
    scale_x_continuous(limits=[-xx, xx]) + 
    geom_vline(xintercept=0,linetype='--') + 
    guides(alpha=False) +
    scale_alpha_manual(values=[0.25,1.0]) +
    scale_color_discrete(name='SSI',labels=['Version 1','Version 2']))
gg_save('gg_bhat_logit.png', dir_figures, gg_bhat_logit, 5, 11)

# (ii) XGBoost importance
tmp2 = bhat_ssi.query('mdl=="xgb"').assign(desc=lambda x: cat_reorder(x.desc, x.bhat))
# xx = np.ceil(tmp2.bhat.abs().max()*10)/10
gg_bhat_xgb = (ggplot(tmp2, aes(y='desc',x='np.log(bhat)',color='y')) + 
    theme_bw() + geom_point(position=posd) + 
    theme(axis_title_y=element_blank(), legend_position=(0.7,0.20)) + 
    labs(x='log(Feature importance)') + 
    scale_color_discrete(name='SSI',labels=['Version 1','Version 2']))
gg_save('gg_bhat_xgb.png', dir_figures, gg_bhat_xgb, 5, 11)


# (iii) Relationship between XGBoost and LR (Coefficient vs p-value)
tmp3 = bhat_ssi.pivot_table(['bhat','pval'],['desc','y'],'mdl').reset_index()
tmp3.columns.to_frame().assign(mdl=lambda x: np.where(x.mdl=='',x[0],x.mdl))
q1 = tmp3.columns.to_frame().reset_index(None,True)
q1 = q1.assign(mdl=lambda x: np.where(x.mdl=='',x[0],x.mdl)).assign(mdl=lambda x: np.where(x.mdl=="logit",x[0],x.mdl))
tmp3.columns = pd.MultiIndex.from_frame(q1)
tmp3 = tmp3.melt(['desc','y','xgb'],col_level=1).rename(columns={'mdl':'logit'})
tmp3 = tmp3.merge(tmp[['desc','is_sig']])
tmp3 = tmp3.assign(value=lambda x: np.where(x.logit == 'pval', np.sqrt(-np.log10(x.value)), np.abs(x.value)), xgb=lambda x: np.where(x.logit == 'bhat', np.log(x.xgb), x.xgb))
tmp4 = tmp3.groupby(['y','logit']).apply(lambda x: stats.pearsonr(x.xgb, x.value)[0]).reset_index().rename(columns={0:'rho'})
tmp4 = tmp4.assign(xx=np.tile([0.88,3],2),yy=[-4.7,0.14,-5.4,0.11], lbl=lambda x: 'rho='+np.round(x.rho*100,1).astype(str)+'%').rename(columns={'logit':'Logistic'})

gt = 'Text shows pearson correlation\nHighlighted points have FDR<5%%' % (rho*100)
tmp3.rename(columns={'logit':'Logistic'}, inplace=True)
tmp_di = {'bhat':'Coefficient (y=log, x=abs)', 'pval':'P-Value (y=None, x=-log10)'}
gg_logit_xgb = (ggplot(tmp3,aes(x='value',y='xgb',color='y',alpha='is_sig.astype(str)')) + 
    theme_bw() + geom_point() + ggtitle(gt) + 
    theme(legend_position='right', subplots_adjust={'wspace': 0.25}) + 
    labs(x='Logistic Regression',y='XGBoost Importance') + 
    facet_wrap('~Logistic',labeller=labeller(Logistic=tmp_di),scales='free') + 
    guides(alpha=False) + 
    scale_alpha_manual(values=[0.2,1.0]) +
    scale_color_discrete(name='SSI',labels=['Version 1','Version 2']) + 
    stat_smooth(aes(x='value',y='xgb',color='y'),data=tmp3,
        inherit_aes=False,method='lm',se=False) + 
    geom_text(aes(x='xx',y='yy',color='y',label='lbl'),
        data=tmp4,inherit_aes=False))
gg_save('gg_logit_xgb.png', dir_figures, gg_logit_xgb, 8, 4)

