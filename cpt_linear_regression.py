import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot
from support.naive_bayes import mbatch_NB
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns
from sklearn import preprocessing
from support.support_funs import stopifnot
from support.mdl_funs import normalize, idx_iter


# missing
# readmission1
###############################
# ---- STEP 1: LOAD DATA ---- #

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_figures = os.path.join(dir_base,'..','figures')

fn_X = 'X_imputed.csv'
fn_Y = 'y_agg.csv'

dat_X = pd.read_csv(os.path.join(dir_output,fn_X))
dat_Y = pd.read_csv(os.path.join(dir_output,fn_Y))

dat_X = pd.get_dummies(dat_X)


# !! ENCODE CPT AS CATEGORICAL !! #
dat_X['cpt'] = 'c'+dat_X.cpt.astype(str)

# get cpt with most data
top_cpts = dat_X.groupby('cpt').size().sort_values(ascending=False)
top_cpts = pd.DataFrame({'cpt':top_cpts.index, 'count':top_cpts.values})
# any cpts with over 40 observations
top_cpts = top_cpts[top_cpts['count'] > 1000]
top_cpts = top_cpts.cpt.unique()

# subset dat_X and dat_Y by top cpts
dat_X = dat_X[dat_X.cpt.isin(top_cpts)].reset_index(drop=True)
dat_Y= dat_Y[dat_Y.caseid.isin(dat_X.caseid)].reset_index(drop=True)

# get columns
cn_X = list(dat_X.columns[2:])
cn_Y = list(dat_Y.columns[2:])

# remove cpt
cn_X = np.delete(cn_X, 1)


###############################################
# ---- STEP 2: LEAVE-ONE-YEAR - ALL VARIABLES ---- #

# start cpt loop
holder_auc = []
for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii+1, len(cn_Y)))
    tmp_ii = pd.concat([dat_Y.operyr, dat_Y[vv]==-1],axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv:'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]

    holder_score = []
    for yy in tmp_train_years:
        print('Train Year %i' % (yy))
        idx_train = dat_X.operyr.isin(tmp_years) & (dat_X.operyr < yy)
        idx_test = dat_X.operyr.isin(tmp_years) & (dat_X.operyr == yy)
        # get dummies
        Xtrain, Xtest = dat_X.loc[idx_train, cn_X].reset_index(drop=True), \
                        dat_X.loc[idx_test, cn_X].reset_index(drop=True)
        ytrain, ytest = dat_Y.loc[idx_train,[vv]].reset_index(drop=True), \
                        dat_Y.loc[idx_test,[vv]].reset_index(drop=True)


        # --- train model --- #


        logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
        from sklearn.model_selection import train_test_split

        logit_fit = logisticreg.fit(Xtrain, ytrain.values.ravel())
        logit_preds = logit_fit.predict_proba(Xtest)[:,1]
        # predict
        holder_score.append(pd.DataFrame({'auc':metrics.roc_auc_score(ytest.values, logit_preds),
            'pr':metrics.average_precision_score(ytest.values, logit_preds), 'test_year':yy},index=[0]))
    holder_auc.append(pd.concat(holder_score).assign(outcome=vv))


res_auc = pd.concat(holder_auc).reset_index(drop=True)
res_auc.to_csv(os.path.join(dir_output,'logit_auc_all_cpt.csv'),index=False)


####################################################
# ---- STEP 3: LEAVE-ONE-YEAR - ALL VARIABLES, FOR EACH CPT CODE ---- #


# start cpt loop
holder_cpt = []
for jj, tt in enumerate(top_cpts):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (tt, jj + 1, len(top_cpts)))
    # subset by cpt
    sub_X = dat_X[dat_X['cpt'] == tt].reset_index(drop=True)
    sub_Y = dat_Y[dat_Y.caseid.isin(sub_X.caseid)].reset_index(drop=True)

    holder_auc = []
    for ii, vv in enumerate(cn_Y):
        print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii+1, len(cn_Y)))
        tmp_ii = pd.concat([sub_Y.operyr, sub_Y[vv]==-1],axis=1)
        tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv:'n'})
        tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
        tmp_years = tmp_years.astype(int)
        tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]

        # keep on the years where y has two labels
        tmp_out = sub_Y.groupby(['operyr', vv]).size().reset_index().rename(columns={'0':'n'})
        tmp_out_years = tmp_out[tmp_out[vv]==1].operyr.values
        tmp_train_years = np.intersect1d(tmp_train_years, tmp_out_years)

        holder_score = []
        if len(tmp_train_years) <=1:
            holder_score.append(pd.DataFrame({'auc': 'NA',
                                              'pr': 'NA',
                                              'test_year': 'NA'}, index=[0]))
        else:
            # now remove max value
            tmp_test_years = np.intersect1d(tmp_train_years, tmp_out_years)
            max_year = tmp_train_years.max()
            tmp_train_years = tmp_train_years[tmp_train_years != max_year]
            all_years = sub_X.operyr.unique()


            for yy in tmp_train_years:
                print('Train Year %i' % (yy))
                test_years = tmp_test_years[tmp_test_years> yy]
                test_year = test_years[0]
                idx_train = sub_X.operyr.isin(tmp_years) & (sub_X.operyr == yy)
                idx_test = sub_X.operyr.isin(tmp_years) & (sub_X.operyr == test_year)
            # get dummies
                Xtrain, Xtest = sub_X.loc[idx_train, cn_X].reset_index(drop=True), \
                                sub_X.loc[idx_test, cn_X].reset_index(drop=True)
                ytrain, ytest = sub_Y.loc[idx_train,[vv]].reset_index(drop=True), \
                                sub_Y.loc[idx_test,[vv]].reset_index(drop=True)


                # --- train model --- #
                logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
                from sklearn.model_selection import train_test_split

                logit_fit = logisticreg.fit(Xtrain, ytrain.values.ravel())
                logit_preds = logit_fit.predict_proba(Xtest)[:,1]
            # predict
                holder_score.append(pd.DataFrame({'auc':metrics.roc_auc_score(ytest.values, logit_preds),
                    'pr':metrics.average_precision_score(ytest.values, logit_preds), 'test_year':test_year},index=[0]))
        holder_auc.append(pd.concat(holder_score).assign(outcome=vv))
    holder_cpt.append(pd.concat(holder_auc).assign(cpt=tt))

res_auc_cpt = pd.concat(holder_cpt).reset_index(drop=True)
res_auc_cpt.to_csv(os.path.join(dir_output,'logit_auc_each_cpt.csv'),index=False)


##############################################
# ---- STEP 4: ON ALL CPT SCORES (LIKE STEP 1, BUT WITH SCORE INSTEAD OF INTERCEPT) ---- #
# GET NB PHAT DATA
file_name = 'nbayes_phat.csv'
# read in phat values from naive bayes
nb_phat = pd.read_csv(os.path.join(dir_output,file_name))

# add 'phat' to cn_X list
cn_X = np.append(cn_X, 'phat')
# start cpt loop
holder_auc = []
for ii, vv in enumerate(cn_Y):
    print('##### ------- Outcome %s (%i of %i) -------- #####' % (vv, ii+1, len(cn_Y)))
    # subset np_phat by vv
    sub_phat = nb_phat[nb_phat['outcome']==vv].reset_index(drop=True)
    sub_X = pd.merge(dat_X, sub_phat, on='caseid').reset_index(drop=True)
    sub_X = pd.DataFrame(sub_X).rename(columns={"operyr_x": "operyr"})
    # only keep needed columns
    # subset y by sub_X.caseid
    sub_Y = dat_Y.loc[dat_Y['caseid'].isin(sub_X.caseid)].reset_index(drop=True)
    tmp_ii = pd.concat([sub_Y.operyr, dat_Y[vv] == -1], axis=1)
    tmp_ii = tmp_ii.groupby('operyr')[vv].apply(np.sum).reset_index().rename(columns={vv: 'n'})
    tmp_years = tmp_ii[tmp_ii.n == 0].operyr.values
    tmp_years = tmp_years.astype(int)
    tmp_train_years = tmp_years[tmp_years > (tmp_years.min())]

    # keep on the years where y has two labels
    tmp_out = sub_Y.groupby(['operyr', vv]).size().reset_index().rename(columns={'0': 'n'})
    tmp_out_years = tmp_out[tmp_out[vv] == 1].operyr.values
    tmp_train_years = np.intersect1d(tmp_train_years, tmp_out_years)

    holder_score = []
    if len(tmp_train_years) <=1:
        holder_score.append(pd.DataFrame({'auc': 'NA',
                                          'pr': 'NA',
                                          'test_year': 'NA'}, index=[0]))
    else:
        # now remove max value
        tmp_test_years = np.intersect1d(tmp_train_years, tmp_out_years)
        max_year = tmp_train_years.max()
        tmp_train_years = tmp_train_years[tmp_train_years != max_year]
        all_years = sub_X.operyr.unique()


        for yy in tmp_train_years:
            print('Train Year %i' % (yy))
            idx_train = sub_X.operyr.isin(tmp_years) & (sub_X.operyr < yy)
            idx_test = sub_X.operyr.isin(tmp_years) & (sub_X.operyr == yy)
            # get dummies
            Xtrain, Xtest = sub_X.loc[idx_train, cn_X].reset_index(drop=True), \
                            sub_X.loc[idx_test, cn_X].reset_index(drop=True)
            ytrain, ytest = sub_Y.loc[idx_train,[vv]].reset_index(drop=True), \
                            sub_Y.loc[idx_test,[vv]].reset_index(drop=True)


            # --- train model --- #

            logisticreg = LogisticRegression(solver='liblinear', max_iter=200)
            from sklearn.model_selection import train_test_split

            logit_fit = logisticreg.fit(Xtrain, ytrain.values.ravel())
            logit_preds = logit_fit.predict_proba(Xtest)[:,1]
            # predict
            holder_score.append(pd.DataFrame({'auc':metrics.roc_auc_score(ytest.values, logit_preds),
                'pr':metrics.average_precision_score(ytest.values, logit_preds), 'test_year':yy},index=[0]))
    holder_auc.append(pd.concat(holder_score).assign(outcome=vv))


res_auc = pd.concat(holder_auc).reset_index(drop=True)
res_auc.to_csv(os.path.join(dir_output,'logit_auc_all_cpt_scores.csv'),index=False)










##############################################
# ---- STEP 4: COMBINE RESULTS AND SAVE ---- #

res_both = pd.concat([res_cpt, res_all],axis=0).reset_index(drop=True)
res_both.to_csv(os.path.join(dir_output,'naivebayes_results.csv'),index=False)

# Calculate percentage
res_pct = res_both.pivot_table('auc',['outcome','operyr'],'tt').reset_index()
res_pct['pct'] = 1 - (res_pct.cpt-0.5) / (res_pct['all']-0.5)
res_pct.pct = np.where(res_pct.pct < 0 , 0 ,res_pct.pct)

g = sns.FacetGrid(data=res_both,col='outcome',col_wrap=5,hue='tt')
g.map(sns.scatterplot,'operyr','auc')
g.add_legend()
g.savefig(os.path.join(dir_figures,'auc_naivebayes1.png'))

g = sns.FacetGrid(data=res_pct,col='outcome',col_wrap=5)
g.map(sns.scatterplot,'operyr','pct')
g.savefig(os.path.join(dir_figures,'auc_naivebayes2.png'))

