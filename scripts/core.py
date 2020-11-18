import pandas as pd
import numpy as np
from scipy import stats
import os

DATA_DIR = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0],'..','data'))
GENE_DIR = os.path.join(DATA_DIR,'annotation')

gene_info = pd.read_csv(os.path.join(GENE_DIR,'gene_info.csv'),index_col=0)
num2name = gene_info.gene_name.to_dict()
trn = pd.read_csv(os.path.join(GENE_DIR,'TRN.csv'),index_col=0)

class IcaData():
    def __init__(self,M,A,X,metadata,cutoff=None):
        self.M = pd.read_csv(M,index_col=0)
        self.M.columns = self.M.columns.astype(int)
        self.A = pd.read_csv(A,index_col=0)
        self.A.index = self.A.index.astype(int)
        self.X = pd.read_csv(X,index_col=0)
        self.metadata = pd.read_csv(metadata,index_col=0)
        if cutoff is not None:
            self.thresholds = {k:self.get_threshold(k,cutoff) for k in self.M.columns}
    
    def show_enriched(self,k):
        gene_table = gene_info.copy()
        gene_table = gene_table.reindex(self.M.index)
        in_imod = abs(self.M[k]) > self.thresholds[k]
        weights = self.M.loc[in_imod,k]
        weights.name = 'weight'
        rows = gene_table.loc[in_imod]
        final_rows = pd.concat([weights,rows],axis=1)
        return final_rows.sort_values('weight')
    
    def get_threshold(self,k,cutoff):
        i=0
        genes = abs(self.M[k]).sort_values()
        k2,p = stats.normaltest(self.M[k])
        while k2 > cutoff:
            i -= 1
            k2,p = stats.normaltest(self.M.loc[genes.index[:i],k])
        imod_genes = genes.iloc[i:]
        return np.mean([genes.iloc[i], genes.iloc[i - 1]])
        
