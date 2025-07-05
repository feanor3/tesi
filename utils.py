import numpy as np
import pandas as pd

def resize_data(L, lt, ut, t_step):

    df = pd.read_csv(f"./data_generation/data/{L}/s3_cfg_L{L}_A0_mc100000_burn1_tl{lt:.3f}_tu{ut:.3f}.csv", header = None)
    t = df[L]# last column of dataframe, corresponding to temperatures
    df.drop(labels=L, axis=1, inplace=True)


    # Creating numpy array where each element is a lattice configuration
    # c++ code creates (L*temperatures, L) csv file

    # n = configurations generated at different temperatures
    n = int(df.shape[0] / L) - 1
    # data where each row is data-point and there are L*L columns corresponding to the whole lattice
    data = np.zeros((n, L*L))
    temp = np.zeros(n)
    k = 0 # index for filling data

    # n*L rows, step=L every L row a new configuration starts
    for i in range(0,n*L, L):
        z = df.iloc[i:i+L].to_numpy()
        data[k] = z.flatten()
        temp[k] = t[i]
        k += 1
    
    np.save(f"data/{L}", data)
    np.save(f"data/{L}_temp", temp)

