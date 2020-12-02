import pandas as pd
import os
from support.acc_funs import auc_decomp

dir_base = os.getcwd()
dir_output =os.path.join(dir_base, '..', 'output')
dir_figures = os.path.join(dir_base, '..', 'figures')
fn_Y = 'y_agg.csv'
dat_Y = pd.read_csv(os.path.join(dir_output, fn_Y))
cn_Y = list(dat_Y.columns[25:37])

############### LOGIT
# AGG
read_file_1 = 'best_agg_logit.csv'
res_y_all = pd.read_csv(os.path.join(dir_output, read_file_1))
res_y_all = res_y_all.dropna().reset_index(drop=True)
result_list = []
for i in cn_Y:
    print(i)
    temp_list = []
    sub_res = res_y_all[res_y_all['outcome']==i].reset_index(drop=True)
    y_values = sub_res.y.values
    pred_values = sub_res.preds.values
    group_values = sub_res.cpt.values
    agg_auc_decomp = auc_decomp(y=y_values, score=pred_values, group=group_values, rand=False)
    temp_list.append(pd.DataFrame({'tt': agg_auc_decomp.tt, 'auc': agg_auc_decomp.auc,
                                       'den': agg_auc_decomp.den}))
    result_list.append(pd.concat(temp_list).assign(outcome=i))
agg_model_auc = pd.concat(result_list).reset_index(drop=True)
agg_model_auc.to_csv(os.path.join(dir_output, 'logit_agg_model_auc_decomposed.csv'), index=False)

###############################################
# decompose auc and save
read_file_1 = 'best_sub_logit.csv'
res_y_all = pd.read_csv(os.path.join(dir_output, read_file_1))
res_y_all = res_y_all.dropna().reset_index(drop=True)
result_list = []
for i in cn_Y:
    print(i)
    temp_list = []
    sub_res = res_y_all[res_y_all['outcome']==i].reset_index(drop=True)
    y_values = sub_res.y.values
    pred_values = sub_res.preds.values
    group_values = sub_res.cpt.values
    sub_auc_decomp = auc_decomp(y=y_values, score=pred_values, group=group_values, rand=False)
    temp_list.append(pd.DataFrame({'tt': sub_auc_decomp.tt, 'auc': sub_auc_decomp.auc,
                                       'den': sub_auc_decomp.den}))
    result_list.append(pd.concat(temp_list).assign(outcome=i))
sub_model_auc = pd.concat(result_list).reset_index(drop=True)
sub_model_auc.to_csv(os.path.join(dir_output, 'logit_sub_model_auc_decomposed.csv'), index=False)


############### random forest

###############################################
# decompose auc and save
read_file_1 = 'best_agg_rf.csv'
res_y_all = pd.read_csv(os.path.join(dir_output, read_file_1))
res_y_all = res_y_all.dropna().reset_index(drop=True)

result_list = []
for i in cn_Y:
    print(i)
    temp_list = []
    sub_res = res_y_all[res_y_all['outcome']==i].reset_index(drop=True)
    y_values = sub_res.y.values
    pred_values = sub_res.preds.values
    group_values = sub_res.cpt.values
    agg_auc_decomp = auc_decomp(y=y_values, score=pred_values, group=group_values, rand=False)
    temp_list.append(pd.DataFrame({'tt': agg_auc_decomp.tt, 'auc': agg_auc_decomp.auc,
                                       'den': agg_auc_decomp.den}))
    result_list.append(pd.concat(temp_list).assign(outcome=i))

agg_model_auc = pd.concat(result_list).reset_index(drop=True)
agg_model_auc.to_csv(os.path.join(dir_output, 'rf_agg_model_auc_decomposed.csv'), index=False)

###############################################
# decompose auc and save

read_file_1 = 'best_sub_rf.csv'
res_y_all = pd.read_csv(os.path.join(dir_output, read_file_1))
res_y_all = res_y_all.dropna().reset_index(drop=True)

result_list = []
for i in cn_Y:
    print(i)
    temp_list = []
    sub_res = res_y_all[res_y_all['outcome']==i].reset_index(drop=True)
    y_values = sub_res.y.values
    pred_values = sub_res.preds.values
    group_values = sub_res.cpt.values
    sub_auc_decomp = auc_decomp(y=y_values, score=pred_values, group=group_values, rand=False)
    temp_list.append(pd.DataFrame({'tt': sub_auc_decomp.tt, 'auc': sub_auc_decomp.auc,
                                       'den': sub_auc_decomp.den}))
    result_list.append(pd.concat(temp_list).assign(outcome=i))

sub_model_auc = pd.concat(result_list).reset_index(drop=True)
sub_model_auc.to_csv(os.path.join(dir_output, 'rf_sub_model_auc_decomposed.csv'), index=False)

############### xgboost

###############################################
# decompose auc and save
read_file_1 = 'best_agg_xgb.csv'
res_y_all = pd.read_csv(os.path.join(dir_output, read_file_1))
res_y_all = res_y_all.dropna().reset_index(drop=True)

result_list = []
for i in cn_Y:
    print(i)
    temp_list = []
    sub_res = res_y_all[res_y_all['outcome']==i].reset_index(drop=True)
    y_values = sub_res.y.values
    pred_values = sub_res.preds.values
    group_values = sub_res.cpt.values
    agg_auc_decomp = auc_decomp(y=y_values, score=pred_values, group=group_values, rand=False)
    temp_list.append(pd.DataFrame({'tt': agg_auc_decomp.tt, 'auc': agg_auc_decomp.auc,
                                       'den': agg_auc_decomp.den}))
    result_list.append(pd.concat(temp_list).assign(outcome=i))

agg_model_auc = pd.concat(result_list).reset_index(drop=True)
agg_model_auc.to_csv(os.path.join(dir_output, 'xgb_agg_model_auc_decomposed.csv'), index=False)

###############################################
# decompose auc and save

read_file_1 = 'best_sub_xgb.csv'
res_y_all = pd.read_csv(os.path.join(dir_output, read_file_1))
res_y_all = res_y_all.dropna().reset_index(drop=True)

result_list = []
for i in cn_Y:
    print(i)
    temp_list = []
    sub_res = res_y_all[res_y_all['outcome']==i].reset_index(drop=True)
    y_values = sub_res.y.values
    pred_values = sub_res.preds.values
    group_values = sub_res.cpt.values
    sub_auc_decomp = auc_decomp(y=y_values, score=pred_values, group=group_values, rand=False)
    temp_list.append(pd.DataFrame({'tt': sub_auc_decomp.tt, 'auc': sub_auc_decomp.auc,
                                       'den': sub_auc_decomp.den}))
    result_list.append(pd.concat(temp_list).assign(outcome=i))

sub_model_auc = pd.concat(result_list).reset_index(drop=True)
sub_model_auc.to_csv(os.path.join(dir_output, 'xgb_sub_model_auc_decomposed.csv'), index=False)


