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

        # loading test data
        temp_test = np.load(f"./data/{L}_test_temp.npy")
        data_test = np.load(f"./data/{L}_test.npy")


        # sort data into ascending order according to temperatures
        # potrei metterlo in utils

        index = np.argsort(temp_test, axis=1)
        temp_test = np.sort(temp_test, axis=1)
        index_expanded = index[..., np.newaxis]
        data_test = np.take_along_axis(data_test, index_expanded, axis=1)

        t_test = (temp_test > T_CRIT).astype(int)

        return data_train, t_train, data_val, t_val, data_test, t_test
    
    if mode == 'cut':
        min_bound = 1.9
        max_bound = 2.6
        T_CRIT = 2.2691853 # k_b * T_C / J  with k_b=1, J = interaction constant
        data = np.load(f"data/{L}_test_tanti.npy")#.reshape(-1, 100)
        temps = np.load(f"data/{L}_temp_tanti.npy")#.reshape(-1, 1)

        # target value
        t = ((temps > max_bound) | (temps < min_bound)).astype(int)

        # removing data close to T_CRIT
        idx = np.where(t==1)
        data = data[idx]
        temps = temps[idx]
        n = data.shape[0]
        t = (temps > T_CRIT).astype(int)

        # DATA Shuffling
        rng = np.random.default_rng()
        indices = np.arange(data.shape[0])
        rng.shuffle(indices)

        data = data[indices]
        t = t[indices]
        temps = temps[indices]

        # splitting data in 80% training, 20% validation, 10% test
        a = int(0.8*n)
        data_train = data[:a]
        data_val = data[a:]

        t_train = t[:a]
        t_val = t[a:]

        # loading data for testing
        temp_test = np.load(f"./data/{L}_test_temp.npy")    # shape (10, 24)
        data_test = np.load(f"./data/{L}_test.npy")         # shape (10, 24, 100)

        # Create mask for filtering along axis 1
        mask = (temp_test > max_bound) | (temp_test < 2.2)        # shape (10, 24), bool

        # Filter each group, preserving the first dimension
        filtered_temps = [temp_test[i][mask[i]] for i in range(temp_test.shape[0])]  # list of arrays, each (something,)
        filtered_data = [data_test[i][mask[i]] for i in range(data_test.shape[0])]   # list of arrays, each (something, 100)

        # Sort within each group
        for i in range(len(filtered_temps)):
            sort_idx = np.argsort(filtered_temps[i])
            filtered_temps[i] = filtered_temps[i][sort_idx]
            filtered_data[i] = filtered_data[i][sort_idx]

        # Optionally, convert lists to arrays for further processing
        filtered_temps = np.array(filtered_temps)  # shape (10, something)
        filtered_data = np.array(filtered_data)    # shape (10, something, 100)
        temp_test = filtered_temps
        data_test = filtered_data
        t_test = (temp_test > T_CRIT).astype(int)

        return data_train, t_train, data_val, t_val, data_test, t_test
    

def get_training_data(data, t, fraction):
    "data, t = label, fraction = franction of validation set"
    # DATA Shuffling
    rng = np.random.default_rng()
    indices = np.arange(data.shape[0])
    rng.shuffle(indices)
    n = data.shape[0]
    data = data[indices]
    t = t[indices]
    #temps = temps[indices]

    # splitting data in 80% training, 20% validation, 10% test
    a = int(fraction*n)
    data_train = data[a:]
    data_val = data[:a]

    t_train = t[a:]
    t_val = t[:a]
    
    return data_train, t_train, data_val, t_val