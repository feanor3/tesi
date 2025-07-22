import numpy as np
import pandas as pd

def resize_data(L, lt, ut):

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


def resize_data_test(L):
    '''
    Resize test data. Metropolis algorithm is run 10 time on the same temperatures
    and intervals in order to get statistical quantities out of the data.
    In L_test there are 10 simulations concatenated: the function divide the file in 10 pieces and each of the pieces is resized like in resize_data()
    '''  
    
    N = 10 # times algorithm has been run
    a = []
    for i in range(N):
        dff =  pd.read_csv(f"./data_generation/data/tmp/{i}.csv", header = None)
        a.append(dff)


    df = pd.concat(a)
    t = df[L]# last column of dataframe, corresponding to temperatures
    df.drop(labels=L, axis=1, inplace=True)

    
    ''' # imporved version of loops from chatgpt
    # Number of configurations generated at different temperatures
    n = df.shape[0] // (L * N)

    # Reshape the DataFrame into (N, n, L, L)
    data_reshaped = df.to_numpy().reshape(N, n, L, L)
    data = data_reshaped.reshape(N, n, L * L)

    # Reshape temperature array accordingly
    temp = np.array(t).reshape(N, n, L)[:, :, 0]  # take the first row per configuration as representative

    '''
    # Creating numpy array where each element is a lattice configuration
    # c++ code creates (L*temperatures, L) csv file

    # n = configurations generated at different temperatures
    n = int(df.shape[0] / L / N) 
    # data where each row is data-point and there are L*L columns corresponding to the whole lattice
    data = np.zeros((N,n, L*L))
    temp = np.zeros((N,n))
    
    kk = 0 # index for filling data again
    for j in range(0, N*n*L, n*L):
        k=0
        w = df.iloc[j:j+n*L] # select first configuration
        y = t[j:j+n*L]
        # n*L rows, step=L every L row a new configuration starts
        for i in range(0,n*L, L):
            z = w.iloc[i:i+L].to_numpy()
            data[kk, k, :] = z.flatten()
            temp[kk, k] = y[i]
            k += 1
        
        kk += 1
    
    np.save(f"data/{L}_test", data)
    np.save(f"data/{L}_test_temp", temp)

def load_train_data(L, mode='all'):
    """
    load training data and returns data_train, t_train, data_val, t_val
    input: L= size, mode='all', 'cut'
    """
    if mode == 'all':
        T_CRIT = 2.2691853 # k_b * T_C / J  with k_b=1, J = interaction constant
        data = np.load(f"data/{L}_test_tanti.npy").reshape(-1, 100)
        temps = np.load(f"data/{L}_temp_tanti.npy").reshape(-1,)
        n = data.shape[0]
        # target value
        t = (temps > T_CRIT).astype(int)

        # DATA Shuffling
        rng = np.random.default_rng()
        indices = np.arange(data.shape[0])
        rng.shuffle(indices)

        data = data[indices]
        t = t[indices]
        temps = temps[indices]

        # splitting data in 80% training, 20% validation
        a = int(0.8*n)
        data_train = data[:a]
        data_val = data[a:]

        t_train = t[:a]
        t_val = t[a:]

        return data_train, t_train, data_val, t_val