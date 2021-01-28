import pandas as pd
import os

class cpt_desciptions():
    def __init__(self):
        fold = os.path.join(os.getcwd(), '..', 'output')
        # read in annotation csv
        #self.cpt_anno = pd.read_csv(os.path.join(fold, 'cpt_anno.csv'))
        cpt_group = pd.read_csv(os.path.join(fold, 'cpt_anno_group.csv'))
        cpt_group = cpt_group.drop(columns='Unnamed: 0').rename(columns={'main_group':'group'})
        cpt_group.title = cpt_group.title.str.strip()
        cpt_organ = pd.read_csv(os.path.join(fold, 'cpt_anno_organ.csv')).drop(columns='Unnamed: 0')
        self.df_cpt = cpt_group.merge(cpt_organ.drop(columns='title'),'left','cpt')
        self.df_cpt.cpt = 'c' + self.df_cpt.cpt.astype(str)
        
    def trans(self,x):
        if isinstance(x, str):
            x = pd.Series([x])
        return pd.DataFrame({'cpt':x}).merge(self.df_cpt,'left','cpt')
