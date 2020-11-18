import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_rec_var(ica_data,ref_conds=None,genes=None,samples=None,modulons=None,plot=True):
    
    # This uses the formula of Cumulative Explained Variance that is described in Sastry et al., Nat. Comm., 2019
    # For this to be accurate, you must include reference conditions if you did not load ica_data with the centered X
    # For PRECISE, ref_conds are ['baseline__wt_glc__1','baseline__wt_glc__2']
    
    # Check inputs
    if genes is None:
        genes = ica_data.X.index
    elif isinstance(genes,str):
        genes = [genes]
    if samples is None:
        samples = ica_data.X.columns
    elif isinstance(samples,str):
        samples = [samples]
    if modulons is None:
        modulons = ica_data.M.columns
    elif isinstance(modulons,str):
        modulons = [modulons]

    # Account for normalization procedures before ICA (X=MA-x_mean)
    if ref_conds is None:
        X = ica_data.X
    else:
        X = ica_data.X.subtract(ica_data.X[ref_conds].mean(axis=1),axis=0)
    
    baseline = pd.DataFrame(np.subtract(X,X.values.mean(axis=0,keepdims=True)),
                    index=ica_data.M.index,columns=ica_data.A.columns)
    baseline = baseline.loc[genes]
    
    # Initialize variables
    base_err = np.linalg.norm(baseline)**2
    MA = np.zeros(baseline.shape)
    rec_var = [0]
    ma_arrs = {}
    ma_weights = {}
    
    # Get individual modulon contributions
    for k in modulons:
        ma_arr = np.dot(ica_data.M.loc[genes,k].values.reshape(len(genes),1),
                         ica_data.A.loc[k,samples].values.reshape(1,len(samples)))
        ma_arrs[k] = ma_arr
        ma_weights[k] = np.sum(ma_arr**2)
    
    # Sum components in order of most important component first
    sorted_mods = sorted(ma_weights,key=ma_weights.get,reverse=True)
    # Compute reconstructed variance
    for k in sorted_mods:
        MA = MA + ma_arrs[k]
        ma_err = np.linalg.norm(MA-baseline)**2
        rec_var.append((1-ma_err/base_err)*100)

    if plot:
        fig,ax = plt.subplots()
        ax.plot(rec_var)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Explained Variance')
        ax.set_ylim([0,100])
        return ax
    else:
        return pd.Series(rec_var[1:],index=sorted_mods)
