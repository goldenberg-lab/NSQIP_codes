import numpy as np
import pandas as pd
import os
from support.support_funs import stopifnot

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')

###############################
# ---- STEP 1: LOAD DATA ---- #

fn_X = 'X_imputed.csv'
fn_Y = 'y_bin.csv'
dat_X = pd.read_csv(os.path.join(dir_output,fn_X))
dat_Y = pd.read_csv(os.path.join(dir_output,fn_Y))
print(dat_X.shape); print(dat_Y.shape)
stopifnot(all(dat_X.caseid == dat_Y.caseid))

