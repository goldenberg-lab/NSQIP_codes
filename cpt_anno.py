import numpy as np
import pandas as pd
import os

# get working directory and file name
dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
file_name = 'cpt_anno.csv'

# read name data
dat = pd.read_csv(os.path.join(dir_output, file_name))
