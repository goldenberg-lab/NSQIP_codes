"""
SCRIPT TO SCRAPE THE CPT CODES INTERPRETATION
"""

import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

dir_base = os.getcwd()
dir_data = os.path.join(dir_base, '..', 'data')

# Load the tabular count for CPT codes
df_cpt = pd.read_csv(os.path.join(dir_data,'df_cpt_year.csv'))
ucpt = list(df_cpt.cpt.unique())

def cpt_scrape(lnk):
    page = requests.get(lnk)
    if page.ok:
        soup = BeautifulSoup(page.text, 'lxml')
        txt = soup.select('.left')[-1].text
    else:
        txt = ''
    return(txt)

# Base url
burl = 'https://coder.aapc.com/cpt-codes'

holder = []
for ii, cpt in enumerate(ucpt):
    print('%i of %i' % (ii+1,len(ucpt)))
    holder.append(cpt_scrape(burl + '/' + str(cpt)))
# combine
df_anno = pd.Series(holder).str.split('\\,',expand=True,n=1)
df_anno.columns = ['code','title']
df_anno.code = df_anno.code.str.replace('CPT\\s','')
df_anno.insert(0,'cpt',ucpt)
# Find the problem cases
print('The following codes have deleted values: %s' %
      ', '.join(df_anno[df_anno.code.str.contains('[^0-9]')].cpt.astype(str).to_list()))

# Code 31582 as 31551: https://www.medtronsoftware.com/pdf/2016/2017_CPT_Updates_Revisions_Deletions_by_Specialty/Respiratory.pdf
df_anno.loc[df_anno.cpt == 31582,'title'] = df_anno[df_anno.cpt == 31551].title.to_list()[0]
df_anno[df_anno.cpt.isin([31582,31551])].T
# Define 61542 as Hemispherectomy: https://www.aapc.com/blog/23869-brain-surgery/
# ala Craniotomy http://ec4medic.com/l5rap/public/cpt?page=444
df_anno.loc[df_anno.cpt == 61542,'title'] = 'Under Craniectomy or Craniotomy Procedures' # 'Hemispherectomy'
df_anno.loc[df_anno.cpt == 43324,'title'] = 'Under Repair Procedures on the Esophagus'
df_anno.loc[df_anno.cpt == 39400,'title'] =  'Under Excision/Resection Procedures on the Mediastinum'

df_anno.drop(columns='code',inplace=True)
print('There are a total of %i unique titles' % (df_anno.title.unique().shape[0]))
# Save for later
df_anno.to_csv(os.path.join(dir_data,'cpt_anno.csv'),index=False)


