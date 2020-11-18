import numpy as np
import pandas as pd

def cos_dist(s1,s2):
    return np.dot(s1.T,s2)/np.sqrt((s1**2).sum())[:, np.newaxis]/np.sqrt((s2**2).sum())[np.newaxis,:]

def spearmanr(s1,s2):
    res = pd.DataFrame(index=s1.columns,columns=s2.columns)
    for k1 in s1.columns:
        for k2 in s2.columns:
            res.loc[k1,k2] = stats.spearmanr(s1[k1],s2[k2])[0]
    return res.values

def rbh(s1,s2):
    dist = 1-abs(cos_dist(s1,s2))

    colmins = dist.argmin(axis=0)
    rowmins = dist.argmin(axis=1)

    res = []
    for idx1,idx2 in enumerate(rowmins):
        if colmins[idx2] == idx1:
            res.append([idx1,idx2,dist[idx1,idx2]])
    return res
    
def get_edge_color(dist):
    hex_code = '{:2x}'.format(int(dist*255)).replace(' ','0')
    return '#'+hex_code*3
