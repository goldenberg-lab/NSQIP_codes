
"""
01. extract_combine_data.py
Reads in NSQIP-P PUFs statas (2012 - 2017) and text file (2018), concatenating the individual years together
into a single dataframe and csv, 'combined_raw.csv'.
For each year, also extracts the variable names (columns) and saves into 'yr_vars.csv'
TODO: implement command line version given a root directory...
TODO: also don't really like the use of placeholders because of memory
"""
import os
import re
import pandas as pd
# the root of the nsqip directory
main_dir = '/Users/delvin/Documents/OneDrive - SickKids/nsqip/'
# assumes data sits in  'nsqip/data/raw/*'
data_dir = os.path.join(main_dir, 'data')
raw_data_dir = os.path.join(data_dir, 'raw')

df_ph = []
vars_ph = []
# ---- read in the data -----
# traverses directory, read in each relevant file, convert all variable names to lower case and concatenate
for dirname, dirnames, filenames in os.walk(raw_data_dir, topdown = True):
    for filename in filenames:
        if re.search('[0-9]{2}.dta|18.txt$|15_v2.dta', filename): # raw statas and 2018 file
            fn = os.path.join(dirname, filename)                  # input file absolute path
            # print(dirname)
            # print(filename)
            if fn.endswith('dta'):
                print("Reading in {}".format(fn))
                df = pd.read_stata(fn)
                print("Done...")
                # 'ya'
            else:
                print("Reading in {}".format(fn))
                df = pd.read_csv(fn)
                df.drop(df.columns[[0]], axis = 1, inplace = True) # drop pesky first column (looks like a pd index),
                                                                   # axis = 1 is column and inplace drops without re-assigning
                print("Done...")
            df.columns = df.columns.str.lower()

            # puf_yr = test[('operyr')].unique().repeat(test.columns.shape[0])  # repeat year for same length as # of variables
            res = re.search(r'[0-9]{4}', fn)                                    # extract current puf year, safer to just get it from the file name
            puf_yr = res.group(0)
            names_df = pd.DataFrame(data={'year': puf_yr, 'vars': df.columns},
                                    columns=['year', 'vars'])

            vars_ph.append(names_df)
            df_ph.append(df)

df = pd.concat(df_ph)
# df.shape # (602584, 399)
# df.head


yr_vars = pd.concat(vars_ph)
# yr_vars.shape # 2536,2
# yr_vars
del(df_ph)
del(vars_ph)

# ---- save down the files -----
# outdir = os.path.join(os.path.dirname(os.path.dirname(dirname)), 'output')
outdir = os.path.join(main_dir, 'output')           # output directory
out_dat = os.path.join(outdir, 'combined_raw.csv')  # the combined, uncleaned data
out_vars = os.path.join(outdir, 'yr_vars.csv')      # year by variable vars

if not os.path.exists(out_dat):
    print('{} does not exist yet... saving..'.format(out_dat))
    df.to_csv(out_dat, index = False)
if not os.path.exists(out_vars):
    print('{} does not exist yet... saving...'.format(out_vars))
    yr_vars.to_csv(out_vars, index = False)

# NULL can mean Unknown, Did not Occur (eg. for
# -99 can mean either no response or patient did not experience this
# apparently pandas reads these in as null so we gud? (actually not since i'm going to use r for the eda)
