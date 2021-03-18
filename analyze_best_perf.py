# Load modules
import os
import gc
import sys
from time import time
from scipy import stats
import pandas as pd
import numpy as np
from plotnine import *
# Load the help functions
from support.acc_funs import fast_auc, fast_decomp, write_fast_decomp, write_fast_inference, gen_CI, auc2se
from support.support_funs import makeifnot, decomp_var, find_dir_nsqip, gg_save
from support.fast_bootstrap import bs_student_spearman
from support.get_cpt_annotations import cpt_desciptions
from support.dict import di_outcome
from scipy.interpolate import UnivariateSpline
from scipy.stats import rankdata

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set directories
dir_NSQIP = find_dir_nsqip()
dir_output = os.path.join(dir_NSQIP, 'output')
dir_weights = os.path.join(dir_output, 'weights')
dir_figures = os.path.join(dir_NSQIP, 'figures')
lst_dir = [dir_output, dir_weights, dir_figures]
assert all([os.path.exists(fold) for fold in lst_dir])

di_model = {'logit':'Logistic-L2', 'rf':'RandomForest',
            'xgb':'XGBoost', 'nnet':'MultiNet'}
di_method = {'agg':'Aggregate', 'sub':'CPT-model'}

cpt_trans = cpt_desciptions()

#######################################
# ----- (1) LOAD SKLEARN MODELS ----- #

fn_output = pd.Series(os.listdir(dir_output))
fn_best = fn_output[fn_output.str.contains('.csv$') & fn_output.str.contains('^best_[agg|sub]')].reset_index(None,True)

fn_csv = 'df_best.csv'
check_csv = fn_csv in os.listdir(dir_output)
if not check_csv:
    print('Loading data via loop')
    holder = []
    cn_keep = ['caseid','model','outcome','test_year','cpt','y','preds']
    for fn in fn_best:
        print('Loading file: %s' % fn)
        path = os.path.join(dir_output, fn)
        tmp_df = pd.read_csv(path, usecols=cn_keep)  #, nrows=10
        tmp_df.rename(columns={'model':'method'},inplace=True)
        tmp_df.caseid = tmp_df.caseid.astype(int)
        mdl = fn.split('.')[0].split('_')[-1]
        tmp_df.insert(0,'model',mdl)
        holder.append(tmp_df)
        del tmp_df
        #break
    df_nsqip = pd.concat(holder).reset_index(None, True)
    df_nsqip.outcome = df_nsqip.outcome.str.replace('agg_','')
    df_nsqip['version'] = df_nsqip.outcome.str.replace('[^0-9]','')
    df_nsqip.version = np.where(df_nsqip.version == '', '1', df_nsqip.version).astype(int)
    df_nsqip.outcome = df_nsqip.outcome.str.replace('[^a-z]','')
    del holder
    print('Writing to file')
    df_nsqip.to_csv(os.path.join(dir_output, fn_csv), index=False)
else:
    print('Loading large CSV file')
    df_nsqip = pd.read_csv(os.path.join(dir_output, fn_csv))
dat_cpt_year = df_nsqip.groupby(['test_year','cpt']).size().reset_index().drop(columns=[0])

##################################
# ----- (2) LOAD MULTITASK ----- #

# Need to subset rows to equivalent
fn_weights = pd.Series(os.listdir(dir_weights))
fn_weights = fn_weights[fn_weights.str.contains('[0-9]{4}\\.csv$')].reset_index(None, True)
holder = []
for fn in fn_weights:
    print('fn: %s' % fn)
    tmp = pd.read_csv(os.path.join(dir_weights, fn)).drop(columns='Unnamed: 0',errors='ignore')
    tmp.rename(columns={'lbl':'outcome','operyr':'test_year','phat':'preds'},inplace=True)

    tmp = tmp[tmp.outcome.str.contains('^agg')].reset_index(None,True)
    tmp = tmp.merge(dat_cpt_year,'inner',['test_year','cpt'])
    tmp.outcome = tmp.outcome.str.replace('agg_','')
    tmp['version'] = tmp.outcome.str.replace('[^0-9]','')
    tmp.version = np.where(tmp.version == '', '1', tmp.version).astype(int)
    tmp.outcome = tmp.outcome.str.replace('[^a-z]','')
    holder.append(tmp)
dat_multi = pd.concat(holder)
dat_multi = dat_multi.sort_values(['test_year','outcome']).reset_index(None,True)
dat_multi = dat_multi.assign(method='agg',model='nnet')
df_nsqip = pd.concat([df_nsqip, dat_multi]).reset_index(None,True)
del dat_multi
gc.collect()

####################################
# ----- (3) Decompose AUROCs ----- #

# Decompose including the year
cn_gg1 = ['model','test_year','outcome','version','method']
fn_within_year = 'df_within_year.csv'
df_within_year = write_fast_decomp(df=df_nsqip, fn=fn_within_year, cn=cn_gg1, path=dir_output, ret_df=False)
# Repeat on the CPT level
fn_within_year_cpt = 'df_within_year_cpt.csv'
df_within_year_cpt = write_fast_decomp(df=df_nsqip, fn=fn_within_year_cpt, cn=cn_gg1, path=dir_output, ret_df=True)

# Decompose aggregating over years
cn_gg2 = ['model','outcome','version','method']
fn_within = 'df_within.csv'
df_within = write_fast_decomp(df=df_nsqip, fn=fn_within, cn=cn_gg2, path=dir_output, ret_df=False)
# Repeat on the CPT level
fn_within_cpt = 'df_within_cpt.csv'
df_within_cpt = write_fast_decomp(df=df_nsqip, fn=fn_within_cpt, cn=cn_gg2, path=dir_output, ret_df=True)


#########################################
# ----- (4) BEST label and models ----- #

# Subset to within
sub_within = df_within.query('tt == "within" & method=="agg"').reset_index(None, True)#.drop(columns=['tt'])
gg_best = ['outcome','version','method']
best_outcome = sub_within.groupby(gg_best).auc.mean().reset_index()
best_outcome = best_outcome.sort_values(['outcome','auc'],ascending=[True,False]).groupby('outcome').head(1)
best_outcome = best_outcome.drop(columns='auc').reset_index(None,True)
best_outcome.to_csv(os.path.join(dir_output,'best_outcome.csv'),index=False)
best_mdl = sub_within.merge(best_outcome, 'inner', gg_best)
best_mdl = best_mdl.groupby(gg_best+['model']).apply(lambda x: np.sum(x.auc*x.den)/x.den.sum()).reset_index()
best_mdl = best_mdl.rename(columns={0:'auc'}).sort_values(['outcome','auc'],ascending=[True,False])
best_mdl = best_mdl.groupby(['outcome']).head(1).reset_index(None, True).drop(columns='auc')
best_mdl.to_csv(os.path.join(dir_output,'best_mdl.csv'),index=False)
print(best_outcome)
print(best_mdl)

#########################################
# ----- (5) Label/model inference ----- #

# Run on data without years
tmp_base = df_within.merge(best_outcome,'inner',gg_best).copy()
tmp_cpt = df_within_cpt.merge(best_outcome,'inner',gg_best).copy()

fn_within_inf = 'df_within_inf.csv'
df_within_inf = write_fast_inference(dat_base=tmp_base.copy(), dat_cpt=tmp_cpt.copy(),fn_write=fn_within_inf, path=dir_output, n_bs=1000, n_max=int(1e6))

tmp_base = df_within_year.merge(best_outcome,'inner',gg_best).copy()
tmp_cpt = df_within_year_cpt.merge(best_outcome,'inner',gg_best).copy()
fn_within_year_inf = 'df_within_year_inf.csv'
df_within_year_inf = write_fast_inference(dat_base=tmp_base.copy(), dat_cpt=tmp_cpt.copy(),fn_write=fn_within_year_inf, path=dir_output, n_bs=1000, n_max=int(1e6))

q1 = df_within_year_inf.merge(best_mdl).groupby(['tt','outcome']).auc.mean().reset_index().assign(auc=lambda x: x.auc-0.5).pivot('outcome','tt','auc')
# Accounts for roughly half of the gain
print(q1.assign(dd=lambda x: x.total-x.within).assign(w_gain=lambda x: x.within/x.total))

#####################################
# ----- (6) CPT SIGNIFICANCE  ----- #

# Find relationships between the statistically significant CPTs
cn = ['model','outcome']
dat_cpt_sig = df_within_cpt.merge(best_outcome)
# Use normal approximation
dat_cpt_sig = dat_cpt_sig.assign(se=lambda x: np.sqrt((1+x.n0+x.n1)/(12*x.n0*x.n1)))
dat_cpt_sig = dat_cpt_sig.drop(columns=['r_s','n1','n0']).rename(columns={'g':'cpt'})
dat_cpt_sig = pd.concat([dat_cpt_sig,gen_CI(x=dat_cpt_sig.auc, se=dat_cpt_sig.se, alpha=0.05)],1)
dat_cpt_sig[['lb','ub']] = dat_cpt_sig[['lb','ub']].clip(lower=0,upper=1)
dat_cpt_sig = dat_cpt_sig.assign(is_sig=lambda x: x.lb > 0.5)
agg_cpt_sig = dat_cpt_sig.groupby(cn+['is_sig']).size().reset_index().rename(columns={0:'n'})
agg_cpt_sig = agg_cpt_sig.pivot_table('n',cn,'is_sig').fillna(0).astype(int).reset_index().melt(cn)
agg_cpt_sig = agg_cpt_sig.rename(columns={'value':'n'}).sort_values(cn).reset_index(None,True)
agg_cpt_sig = agg_cpt_sig.merge(agg_cpt_sig.groupby(cn).n.sum().reset_index().rename(columns={'n':'tot'}))
agg_cpt_sig = agg_cpt_sig.assign(pct=lambda x: x.n/x.tot)
dat_cpt_sig = pd.concat([dat_cpt_sig,cpt_trans.trans(dat_cpt_sig.cpt.values)],1)
dat_cpt_sig.group = dat_cpt_sig.group.str.replace('Surgical\\sProcedures\\son\\sthe\\s','').str.replace('\\sSystem','')

# Get the "best" model and check for organ enrichment
dat_cpt_sig_mdl = dat_cpt_sig.merge(best_mdl,'inner')
sig_organ = dat_cpt_sig_mdl.pivot_table(index='organ',columns='is_sig',values='cpt',aggfunc='size', fill_value=0)
sig_group = dat_cpt_sig_mdl.pivot_table(index='group',columns='is_sig',values='cpt',aggfunc='size', fill_value=0)
sig_both = pd.concat([sig_organ.reset_index().rename(columns={'organ':'term'}).assign(tt='organ'),
           sig_group.reset_index().rename(columns={'group':'term'}).assign(tt='group')]).reset_index(None,True)
sig_both.rename(columns={False:'false1',True:'true1'},inplace=True)
sig_both = sig_both.assign(tot_true=dat_cpt_sig_mdl.is_sig.sum(), tot_false=np.sum(~dat_cpt_sig_mdl.is_sig))
sig_both = sig_both.assign(false0=lambda x: x.tot_false-x.false1, true0=lambda x: x.tot_true-x.true1)
sig_both = sig_both.assign(lOR=lambda x: np.log(x.true1*x.false0/(x.true0*x.false1)),
                se=lambda x: np.sqrt(1/x.false1+1/x.true1+1/x.false0+1/x.true0))
sig_both = sig_both.assign(zscore=lambda x: x.lOR/x.se).assign(pval=lambda x: 2*(1-stats.norm.cdf(np.abs(x.zscore))))
sig_both.term = pd.Categorical(sig_both.term,sig_both.sort_values('lOR',ascending=False).term.values)

##################################
# ----- (7) DISCRETIZATION ----- #

mdl_nsqip = df_nsqip.merge(best_mdl)
mdl_nsqip.to_csv(os.path.join(dir_output, 'best_eta.csv'),index=False)
# Get the different bins so we can do the cuts
p_seq = np.append(np.append(np.array([0]),np.round(np.arange(0.69,0.98,0.01),2)),np.arange(0.99,1.001,0.001))
# Match the percentile to the quantile
dat_pp_qq = mdl_nsqip.groupby('outcome').apply(lambda x: x.preds.quantile(p_seq))
dat_pp_qq = dat_pp_qq.reset_index().melt('outcome',None,'pp','qq')
dat_pp_qq = dat_pp_qq.assign(outcome=lambda x: x.outcome.map(di_outcome), pp = lambda x: x.pp.astype(float))
# Calculate the different bins
tmp_p = mdl_nsqip.groupby('outcome').apply(lambda x: 
            pd.cut(x.preds, bins=x.preds.quantile(p_seq),right=True,labels=p_seq[1:],duplicates='drop')).reset_index()
tmp_p.rename(columns={'level_1':'idx','preds':'pp'},inplace=True)
mdl_nsqip = mdl_nsqip.rename_axis('idx').reset_index().merge(tmp_p,'left',['idx','outcome'])
# Calculate the precision across the different percentiles
res_ppv = mdl_nsqip.groupby(['outcome','pp','y']).size().reset_index().pivot_table(0,['outcome','pp'],'y').reset_index()
res_ppv = res_ppv.rename(columns={0:'y0',1:'y1'}).assign(outcome=lambda x: x.outcome.map(di_outcome),pp=lambda x: x.pp.astype(float))
res_ppv = res_ppv.merge(res_ppv.groupby('outcome').y1.sum().reset_index().rename(columns={'y1':'y1tot'}))
res_ppv = res_ppv.query('pp>@p_seq[1]').sort_values(['outcome','pp'],ascending=False).reset_index(None,True)
res_ppv[['y0','y1']] = res_ppv.groupby('outcome')[['y0','y1']].cumsum()
res_ppv = res_ppv.assign(ppv = lambda x: x.y1/(x.y1+x.y0), sens=lambda x: x.y1/x.y1tot)
res_ppv = res_ppv.melt(['outcome','pp'],['ppv','sens'],'metric')
res_ppv = res_ppv.merge(dat_pp_qq,'left',['outcome','pp'])
res_ppv.to_csv(os.path.join(dir_output, 'res_ppv.csv'),index=False)

###############################
# ----- (8) CPT-SIG RHO ----- #

n_bs, n_s = 1000, 1000

if 'df_rho_outcome.csv' in os.listdir(dir_output):
    df_rho_outcome = pd.read_csv(os.path.join(dir_output,'df_rho_outcome.csv'))
else:
    # Get the pairwise concordance
    dat_cpt_pair = df_within_cpt.merge(best_mdl).pivot('g','outcome','auc')
    cn_pair = dat_cpt_pair.columns.to_list()
    holder = []
    for i in range(len(cn_pair)):
        for j in range(len(cn_pair)):
            cn1, cn2 = cn_pair[i], cn_pair[j]
            v1, v2 = dat_cpt_pair[cn1], dat_cpt_pair[cn2]
            idx_keep = v1.notnull() & v2.notnull()
            v1, v2 = v1[idx_keep].values, v2[idx_keep].values
            tmp_df = bs_student_spearman(v1, v2, n_bs, n_s, alpha=0.05).assign(cn1=cn1, cn2=cn2)
            holder.append(tmp_df)
    df_rho_outcome = pd.concat(holder)
    df_rho_outcome.to_csv(os.path.join(dir_output,'df_rho_outcome.csv'),index=False)
df_rho_outcome = df_rho_outcome.assign(cn1=lambda x: x.cn1.map(di_outcome), cn2=lambda x: x.cn2.map(di_outcome), is_sig = lambda x: np.sign(x.lb)==np.sign(x.ub))
df_rho_outcome = df_rho_outcome.query('tt=="student"').reset_index(None, True)
df_rho_long = df_rho_outcome.pivot_table('rho',['cn1'],'cn2').reset_index().melt('cn1',None,None,'rho')
df_rho_long = df_rho_long.merge(df_rho_outcome[['cn1','cn2','is_sig']])


############################
# ----- (8) FIGURES  ----- #

tmp = pd.DataFrame({'yi':[0.05,0.10],'metric':'ppv'})
gg_ppv = (ggplot(res_ppv, aes(x='pp',y='value',color='outcome')) + 
          theme_bw() + geom_line() + 
          labs(x='Score percentile',y='PPV/Sensitivity') + 
          scale_color_discrete(name='Label') + 
          scale_x_continuous(breaks=list(np.arange(0.7,1.01,0.05))) + 
          facet_wrap('~metric',scales='free_y',labeller=labeller(metric={'ppv':'PPV','sens':'Sensitivity'})) + 
          theme(subplots_adjust={'wspace':0.15}) + 
         geom_hline(aes(yintercept='yi'),linetype='--',data=tmp))
gg_save('gg_ppv.png',dir_figures,gg_ppv,8,4)

"""(8.A) Variation between models/years
There is little variation between models in performance. Exception for a few kidney models in some years. Also shows that for any given model, there is very little variation between years"""

tmp = df_within_year_inf.query('tt=="within"').assign(model=lambda x: x.model.map(di_model)).copy()
posd = position_dodge(0.5)
gg_auc_within = (ggplot(tmp, aes(x='test_year.astype(str)',y='auc',color='model')) + 
                 theme_bw() + geom_point(position=posd) + 
                 geom_linerange(aes(ymin='lb',ymax='ub'),position=posd) + 
                 facet_wrap('~outcome',labeller=labeller(outcome=di_outcome)) + 
                 theme(axis_text_x=element_text(angle=90)) + #,axis_ticks_minor_y=element_blank()
                 labs(y='Within-CPT AUROC',x='Test year') + 
                 scale_y_continuous(limits=[0.25,1],breaks=list(np.arange(0.25,1.01,0.25))) + 
                 geom_hline(yintercept=0.5,linetype='--') + 
                 scale_color_discrete(name='Model') + 
                 ggtitle('Linerange shows 95% bootstrap CI'))
gg_save('gg_auc_within.png',dir_figures,gg_auc_within,8,4)

# (8.B) Variation between decomp/years
tmp = df_within_year_inf.merge(best_mdl)
posd = position_dodge(0.5)
gg_between = (ggplot(tmp,aes(x='test_year.astype(str)',y='auc',color='tt')) + 
              theme_bw() + geom_point(position=posd) + 
              facet_wrap('~outcome',labeller=labeller(outcome=di_outcome)) + 
              scale_color_discrete(name='AUC type',labels=['Total','Within']) + 
              geom_linerange(aes(ymin='lb',ymax='ub'),position=posd) + 
              labs(y='AUROC',x='Test year') + 
              geom_hline(yintercept=0.5,linetype='--') + 
              scale_y_continuous(limits=[0.25,1],breaks=list(np.arange(0.25,1.01,0.25))))
gg_save('gg_between.png',dir_figures,gg_auc_within,8,4)

# (8.C) Distribution within the "within"
tmp1 = df_within.merge(best_mdl).query('tt=="within"').reset_index(None,True).copy()
tmp2 = df_within_cpt.merge(best_mdl).reset_index(None,True).copy()
gg_dist_cpt = (ggplot(tmp2, aes(x='auc')) + theme_bw() + 
              geom_histogram(bins=18,color='black',fill='red',alpha=1/3) + 
              facet_wrap('~outcome',scales='free_y',labeller=labeller(outcome=di_outcome)) + 
              labs(x='AUROC', y='Relative Frequency') + 
              scale_x_continuous(limits=[-0.1,1.1],breaks=list(np.arange(0,1.1,0.25))) + 
              theme(axis_text_y=element_blank(),axis_ticks_minor_y=element_blank(),
                    axis_ticks_major_y=element_blank()) + 
              geom_vline(aes(xintercept='auc'),data=tmp1,color='blue') + 
              ggtitle('Blue line shows within-AUROC') + 
              geom_vline(xintercept=0.5,linetype='--'))
gg_save('gg_dist_cpt.png',dir_figures,gg_dist_cpt,8,4)

# (8.4) CORRELATION BETWEEN CPT RESULTS OVER LABELS
tmp = df_rho_long.query('cn1!=cn2')
tmp = tmp[~tmp[['rho','is_sig']].duplicated()]

gg_rho = (ggplot(tmp, aes(x='cn1',y='cn2',fill='rho',alpha='is_sig',color='is_sig')) + 
          geom_tile(size=1) + theme_bw() + 
          theme(axis_text_x=element_text(angle=90),axis_title=element_blank(),
                panel_grid_major=element_blank(),panel_grid_minor=element_blank()) + 
          scale_fill_gradient(name='Spearman (rho)',low='blue',high='red') + 
          scale_alpha_manual(values=[0.25,1]) + guides(color=False,alpha=False) + 
          scale_color_manual(values=['white','black']) + 
          ggtitle('Highlighted areas are statistically significant'))
gg_save('gg_rho.png',dir_figures,gg_rho,5,4)

# (8.5) CPT signficance
tmp = agg_cpt_sig.query('is_sig==True').assign(outcome=lambda x: 
       pd.Categorical(x.outcome,dat_cpt_sig.outcome.unique()).map(di_outcome),
      model=lambda x: x.model.map(di_model))

posd = position_dodge(0.5)
gg_sig = (ggplot(tmp,aes(x='outcome',y='pct',color='model')) + theme_bw() + 
          geom_point(position=posd) + 
         theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
         scale_color_discrete(name='Model') + 
         labs(y='% of CPTs significant') + 
         ggtitle('AUROCs calculated over all years') + 
         scale_y_continuous(limits=[0,1]))
gg_save('gg_sig.png',dir_figures,gg_sig,4,3)

# (8.6) Label count
n_cpts_mdl = df_within_cpt.merge(best_mdl).groupby(['outcome']).size().reset_index().rename(columns={0:'tot'})
n_cpts_mdl.outcome = n_cpts_mdl.outcome.map(di_outcome)

gg_n_cpts_mdl = (ggplot(n_cpts_mdl,aes(x='outcome',y='tot')) + theme_bw() + 
                  geom_bar(stat='identity',color='black',fill='red',alpha=0.5,width=0.8) + 
                 theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
                 labs(y='# of CPTs') + 
                 scale_y_continuous(limits=[0,150]) + 
                 ggtitle('Only CPTs with y==1') + 
                geom_hline(yintercept=134))
gg_save('gg_n_cpts_mdl.png',dir_figures,gg_n_cpts_mdl,4,3)

# (8.7) CPT Category
gg_sig = (ggplot(sig_both,aes(x='term',y='lOR',color='pval<0.05')) + theme_bw() + 
          geom_point() + geom_linerange(aes(ymin='lOR-1.96*se',ymax='lOR+2*se')) + 
          facet_wrap('~tt',scales='free_x',labeller=labeller(tt={'group':'Category','organ':'Organ'})) + 
         theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
         geom_hline(yintercept=0) + 
         scale_y_continuous(limits=[-4.1,4.1],breaks=list(np.arange(-4,4.1,1))) + 
         labs(y='log(OR) CPT enrichment') + 
         ggtitle('Linerange shows 95% CI') + 
         scale_color_discrete(name='Significant'))
gg_save('gg_sig.png',dir_figures,gg_sig,8, 3.5)


