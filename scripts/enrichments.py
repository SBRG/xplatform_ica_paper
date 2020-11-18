import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


def FDR(p_values,fdr,total=None):
    """Runs false detection correction over a pandas Dataframe
        p_values: Pandas Dataframe with 'pvalue' column
        fdr: False detection rate
        total: Total number of tests (for multi-enrichment)
    """
    
    if total is not None:
        pvals = p_values.pvalue.values.tolist() + [1]*(total-len(p_values))
        idx = p_values.pvalue.index.tolist() + [None]*(total-len(p_values))
    else:
        pvals = p_values.pvalue.values
        idx = p_values.pvalue.index

    keep,qvals = fdrcorrection(pvals,alpha=fdr)
    
    result = p_values.copy()
    result['qvalue'] = qvals[:len(p_values)]
    result = result[keep[:len(p_values)]]
   
    return result.sort_values('qvalue')
    
def contingency(set1,set2,all_genes):
    """Creates contingency table for gene enrichment
        set1: Set of genes (e.g. regulon)
        set2: Set of genes (e.g. i-modulon)
        all_genes: Set of all genes
    """
        
    tp = len(set1 & set2)
    fp = len(set2 - set1)
    tn = len(all_genes - set1 - set2)
    fn = len(set1 - set2)
    return [[tp,fp],[fn,tn]]
    
def single_enrichment_helper(reg_genes,ic_genes,all_genes,reg_name):
    """ Calculates the enrichment of a set of genes in an i-modulon
        reg_genes: Genes in regulon
        ic_genes: Genes in an independent component
        all_genes: All genes in genome
        reg_name: Regulator name
    """
    # Compute enrichments
    ((tp,fp),(fn,tn)) = contingency(reg_genes,ic_genes,all_genes)
        
    # Handle edge cases
    if tp == 0:
        res = [0,1,0,0,0]
    elif fp == 0 and fn == 0:
        res = [np.inf,0,1,1,len(ic_genes)]
    else:
        odds,pval = stats.fisher_exact([[tp,fp],[fn,tn]],alternative='greater')
        recall = np.true_divide(tp,tp+fn)
        precision = np.true_divide(tp,tp+fp)
        res = [np.log(odds),pval,recall,precision,tp]
    return pd.Series(res,index=['log_odds','pvalue','recall','precision','TP'],
                     name=reg_name)

def compute_enrichments(ic_genes,all_genes,trn,fdr=1e-5):
    """Calculate regulon enrichments for a set of genes
        ic_genes: Set of genes to check for regulon enrichments
        all_genes: All genes in genome
        trn: Dataframe of gene_id and TF
        fdr: False detection rate
    """
    
    enrichment_list = []
    for tf in trn.TF.unique():   
        reg_genes = set(trn[trn.TF == tf].gene_id)
        enrichment_list.append(single_enrichment_helper(reg_genes,ic_genes,
                                               all_genes,tf))
    DF_enriched = pd.concat(enrichment_list,axis=1).T

    # Run false detection
    df = FDR(DF_enriched,fdr)
    df['f1score'] = 2*(df['precision']*df['recall'])/(df['precision']+df['recall'])
    return df.sort_values(['f1score'],ascending=[False])
    
def compute_threshold(S,k,cutoff):
    """Computes kurtosis-based threshold for a component of an S matrix
        S: Component matrix with gene weights
        k: Component name
        cutoff: Minimum test statistic value to determine threshold
    """
    i = 0
    # Sort genes based on absolute value
    ordered_genes = abs(S[k]).sort_values()
    K,p = stats.normaltest(S.loc[:,k])
    while K > cutoff:
        i -= 1
        # Check if K statistic is below cutoff
        K,p = stats.normaltest(S.loc[ordered_genes.index[:i],k])
    comp_genes = ordered_genes.iloc[i:]
    if len(comp_genes) == len(S.index):
        return max(comp_genes)+.05
    else:
        return np.mean([ordered_genes.iloc[i],ordered_genes.iloc[i-1]])
