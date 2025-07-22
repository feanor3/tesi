import numpy as np
import pandas as pd
import sys
from classifier import MLPBinary

# LOADING TEST DATA
L = sys.argv[1]
print(L)
T_CRIT = 2.2691853 # k_b * T_C / J  with k_b=1, J = interaction constant

data = np.load(f"data/{L}_test_tanti.npy").reshape(-1, 100)
temps = np.load(f"data/{L}_temp_tanti.npy").reshape(-1,)
n = data.shape[0]
t = (temps > T_CRIT).astype(int) # target value

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

# loading data for testing
temp_test = np.load(f"./data/{L}_test_temp.npy")
data_test = np.load(f"./data/{L}_test.npy")


# sort data into ascending order according to temperatures
# potrei metterlo in utils

index = np.argsort(temp_test, axis=1)
temp_test = np.sort(temp_test, axis=1)
index_expanded = index[..., np.newaxis]
data_test = np.take_along_axis(data_test, index_expanded, axis=1)

t_test = (temp_test > T_CRIT).astype(int)



dimensions = [20, 32, 40, 56, 68, 80, 88, 96, 100]
tolerance = 1e-4
activation = 'relu'
lr = 0.01
batch_size = 200
momentum = 0.8
solver = 'sgd' 
alpha = 0.1 
power_t = 0.5
n_epochs_no_update = 5


accuracy = []
acc_std = []

for dim in dimensions:
    clf = MLPBinary(dim_hidden=dim,tolerance=tolerance,activation='relu', lr=lr, batch_size=batch_size, momentum=momentum, solver='sgd', alpha=alpha, power_t=power_t, n_epochs_no_update=n_epochs_no_update)
    
    clf.fit(data_train, t_train, X_val=data_val, t_val=t_val)

    # PREDICTION ON DATA SET to have mean
    flattened = data_test.reshape(-1, L*L)

    acc = clf.score(flattened, t_test.reshape(-1,))

    accuracy.append(np.mean(acc)) # append mean and std of the 
    acc_std.append(np.std(acc, ddof=1) / np.sqrt(len(acc)))

    print(f"Trained classifer dim_hidden = {dim}")

df = pd.DataFrame( 
    {"acc mean": accuracy,
    "acc std": acc_std},
    index=dimensions
)

df.to_csv('accuracy - dimensions')






