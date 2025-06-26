import numpy as np
from numpy import random as rnd
'''
lattice as linear array -> feeded in linera classifier like this
'''


# lattice size
L = [10, 15, 20]




rng = rnd.default_rng()
sample_size = 10000
# completely random
for length in L:
    
    
    # create square lattice of 01 of size=length
    l = length*length

          
    lattice = rng.integers(0,2,size=(sample_size, l)) #potevo specificare 100*10 e avevo la matrice di punit 
    z = np.zeros((sample_size,l))
    for i in range(sample_size):
        lattice1 = np.ones(l)

        # create indices to be modified to 0
        indices = rng.integers(0,l, size=int(0.05*l))
        lattice1[indices] = 0

        lattice1 = lattice1.astype(int)
        z[i]=lattice1
        

   

    data = np.concatenate((lattice, z), axis=0)   

    labels = np.concatenate((np.ones(sample_size), np.zeros(sample_size)))

    indices = np.arange(data.shape[0])
    
    rng.shuffle(indices)
    


    np.save(f"./dummy_data/{length}", data[indices])
    np.save(f"./dummy_data/{length}_label", labels[indices])


    
    