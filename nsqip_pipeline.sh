#!/bin/bash

# -------------------------------------- #
# Shell script to process the NSQIP data #
# -------------------------------------- #

conda activate NSQIP

echo "----- Step 1: Combine raw data from Stata format -----"
python 01a_combine_data.py
# Output: combined_raw.csv and yr_vars.csv

echo "----- Step 2A: Process data to produce y and X matrices -----"
python yX_process.py
# Output: y_bin.csv and X_preop.csv

echo "----- Step 2B: Create aggregated labels -----"
python y_agg.py
# Output: y_agg.csv

echo "----- Step 3: Test X-imputation -----"
python X_imputation.py
# Output: X_imputed.csv

echo "----- Step 4: Establish CPT Baseline -----"
python cpt_baseline.py
# Output: naivebayes_results.csv, auc_naivebayes1.png, auc_naivebayes2.png

#echo "----- Step 5: Run Multitask NNet -----"
python mtask_pytorch.py
Output: naivebayes_results.csv, auc_naivebayes1.png, auc_naivebayes2.png

#echo "----- Step 6: Evaluate Results -----"
python auc_eval.py
Output: df_decomp_nnet.csv, various figures!

echo "----- logistic regression on aggregate and submodels -----"
python cpt_logit.py
# Output: logit_agg.csv, logit_sub.csv, logit_agg_phat.csv, logit_sub_phat.csv

echo "----- model for cpt annotation file on aggregate and submodels -----"
python cpt_anno.py
# Output: logit_agg_title.csv, logit_sub_title.csv

echo "----- model for cpt main groups file on aggregate and submodels -----"
python cpt_main_group.py
# Output: logit_agg_main.csv, logit_sub_main.csv

echo "----- risk quintiles on aggregate and submodels -----"
python cpt_logit_quintiles.py
# Output: logit_agg_quin_cpt.csv, logit_agg_quin_bin.csv, logit_agg_quin_coef.csv, logit_sub_quin_cpt.csv, logit_sub_quin_bin.csv, logit_sub_quin_coef_bin.csv, logit_sub_quin_coef_cpt.csv

#echo "----- random forest on aggregate and submodels -----"
#python cpt_rf.py
# Output: rf_agg.csv, rf_sub.csv,

#echo "----- xgbBoost on aggregate and submodels -----"
#python cpt_xgb.py
# Output: xgb_agg.csv, xgb_sub.csv

echo "----- logistic regression with bootstrapped AUCs on aggregate and submodels -----"
python cpt_logit_bootstrap.py
# Output: logit_boot_agg.csv, logit_boot_sub.csv, logit_sig_cpts.csv

echo "----- plot results from models on cpt, quintiles, and cpt annotations -----"
python generate_plots.py
# Output: plots to save in figures folder

echo "----- analyze BEST models -----"
python analyze_best_perf.py
# Output: df_best.csv, best_outcome.csv, best_mdl.csv, res_ppv.csv, df_rho_outcome.csv, and 8 figures)


echo "----- process the sickkids validation data -----"
python validation_process.py
# Output: val_Y.csv, val_Yagg.csv, prop_impute.csv, dat_Xmap.csv

echo "----- analyze the sickkids validation data -----"
python validation_run.py
# Output: dup_test.csv, dat_matcher.csv, dat_suspect.csv, df_within_sk_inf.csv
