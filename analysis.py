import numpy as np
import pandas as pd



def resize_data(L, lt, ut, t_step):



    df = pd.read_csv(f"./data_generation/data/{L}/s3_cfg_L10_A0_mc100000_burn1_tl{lt:.3f}_tu{ut:.3f}.csv", header = None)
    t = df[10]
    df.drop(labels=10, axis=1, inplace=True)


    # Creating numpy array where each element is a lattice configuration
    # c++ code creates (L*temperatures, L) csv file

    # n = configurations generated at different temperatures
    n = int((ut-lt) / t_step)
    # data where each row is data-point and there are L*L columns corresponding to the whole lattice
    data = np.zeros((n, L*L))
    temp = np.zeros(n)
    k = 0 # index for filling data

    for i in range(0,n*L, 10):
        z = df.iloc[i:i+L].to_numpy()
        data[k] = z.flatten()
        temp[k] = t[i]
        k += 1
    
    np.save(f"{L}", data)
    np.save(f"{L}_temp", temp)




