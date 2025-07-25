# First, we define the logistic function and its derivative
import numpy as np
import time 

def accuracy(predicted, gold):
    return np.mean(predicted == gold)

def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_diff(y):
    return y * (1 - y)

def relu(x):
    return np.where(x<0, 0, x)

def relu_diff(x):
    return np.where(x<0, 0, 1)

def pool(x):
    return x/2

def pool_diff(x):
    return 1/2

def add_bias(X, bias):
    """X is a NxM matrix: N datapoints, M features
    bias is a bias term, -1 or 1, or any other scalar. Use 0 for no bias
    Return a Nx(M+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias # Make a N*1 matrix of biases
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 

class MLPBinary():
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 6, tolerance=1e-4, activation='relu', solver='sgd', lr = 1e-4, batch_size=100, epochs=100, n_epochs_no_update = 2, momentum = 0.9, alpha=1e-3, power_t = 0.001, symmetric_weights=False):
        """Intialize the hyperparameters"""
        self.bias = bias
        # Dimensionality of the hidden layer
        self.dim_hidden = dim_hidden
        self.epochs = 0
        self.tol = tolerance
        self.lr = lr
        self.lr_initial = lr
        self.epochs = epochs
        self.n_epochs_no_update = n_epochs_no_update
        self.solver = solver
        self.batch_size = batch_size
        self.momentum = momentum # momentum coefficient
        self.alpha = alpha # strength of regularization term
        # self.tau= tau
        self.power_t = power_t
        self.symmetric_weights = symmetric_weights
        self.t_ = 0

        self.activ = [relu, pool, logistic]
        self.activ_diff = [relu_diff, pool_diff, logistic_diff]
    
        
        '''if activation == 'relu':
            self.activ = relu     
            self.activ_diff = relu_diff
        elif activation == 'logistic':
            self.activ = logistic     
            self.activ_diff = logistic_diff'''
        

    def fit(self, X_train, t_train, X_val=None, t_val=None):
        """Initialize the weights. Train *epochs* many epochs.
        
        X_train is a NxM matrix, N data points, M features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        lr is the learning rate
        """
        
        # create loss to store and inspect
        self.train_loss = train_loss = []
        self.train_acc = []
        self.val_loss = val_loss = []
        self.val_acc = val_acc = []

        current_epochs_no_update = 0
        t_start = time.time()
        
        # Turn t_train into a column vector, a N*1 matrix:
        T_train = t_train.reshape(-1,1)
        X_train_bias = add_bias(X_train, self.bias)
        self.N = X_train_bias.shape[0] # number datapoints

        # dimensions of hidden and output layer
        self.dim_in = dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Initialize the weights
        # accoding to Glorot et al.
        factor = 6.0
        np.random.seed(25080)

        init_bound = np.sqrt(factor / (dim_in + self.dim_hidden))
        a = np.random.uniform(-init_bound, init_bound, (dim_in, self.dim_hidden // 2))
        b = np.random.uniform(-init_bound, init_bound, (1, dim_in)) # bias
        c = np.concatenate((a,-a), axis = 1)
        self.weights1 = np.concatenate((b,c)) # [tutti pesi neurone, fino al neurone]

        # i
        dim_out2 = self.dim_hidden // 2
        a = np.identity(dim_out2)
        self.weights2 = np.concatenate((a, a))

        init_bound = np.sqrt(factor / (dim_out2 + dim_out))                               
        self.weights3 = np.random.uniform(-init_bound, init_bound, (dim_out2 +1, dim_out))


        # initialize momentum vector to 0
        self.m1 = np.zeros_like(self.weights1) 
        self.m3 = np.zeros_like(self.weights3)
        
        for e in range(self.epochs):

            if time.time()-t_start > 60*3:
                print("time exceeded")
                break
            
            if self.solver == 'sgd':
                # selecting random elements of X_train_bias as batch
                idx = np.random.permutation(self.N)
                
                # this ensures that training is done over all data per epoch
                for i in range(0, self.N, self.batch_size):
                    X_batch = X_train[idx[i:i+self.batch_size], :]
                    T_batch = T_train[idx[i:i+self.batch_size]]

                    self.update(X_batch, T_batch)
                
            else:
                self.update(X_train, T_train)

            # Loss and accuracy
            acc = accuracy(self.predict(X_train), t_train)
            self.train_acc.append(acc)

            loss = self.compute_loss(self.predict_probabilities(X_train), t_train)
            self.train_loss.append(loss)

            if X_val is not None:
                z = self.compute_loss(self.predict_probabilities(X_val), t_val)
                val_loss.append(z)
                val_acc.append(accuracy(self.predict(X_val), t_val))

                if e > 0 and abs(val_loss[e-1]-val_loss[e]) < self.tol:
                    current_epochs_no_update += 1
                    if current_epochs_no_update > self.n_epochs_no_update:
                        #print("Classifier trained for epochs: ", #self.epochs)
                        break
                else: 
                    current_epochs_no_update = 0
                
            if e > 10 and val_loss[e] > min(val_loss) + 1e-3:
                break
            
            
            self.t_ += self.N # adding datapoints
            self.update_lr(self.t_)
            self.epochs += 1

    def update(self, X, t_train):
        ''''X unbiased data, T_train gold value'''




        X_bias = add_bias(X, self.bias)
        # One epoch
        # The forward step:
        z1 = X_bias @ self.weights1
        a1 = self.activ[0](z1)
        #a1_bias = add_bias(a1, self.bias)

        # in questo caso no bias perchè faccio solo pooling
        dim2 = self.dim_in // 2
        z2 = a1 @ self.weights2
        a2 = self.activ[1](z2) # pooling
        a2 = add_bias(a2, self.bias)

        z3 = a2 @ self.weights3
        a3 = outputs = self.activ[2](z3)  # final output with logistic

        
        # The delta term on the output node:
        delta3 = (outputs - t_train) # last layer
        delta2 = (delta3 @ self.weights3.T)[:,1:] * self.activ_diff[2](z3)
        delta1 = (delta2 @ self.weights2.T) * self.activ_diff[1](z2)

        ''' # The delta terms at the output of the hidden layer:
        hiddenout_diffs = out_deltas @ self.weights2.T
        # The deltas at the input to the hidden layer:
        #hiddenout_diffs = add_bias(hiddenout_diffs, self.bias)
        hiddenact_deltas = (hiddenout_diffs[:, 1:] * 
        self.activ_diff(hidden_outs[:, 1:]))'''

        grad3 = a2.T @ delta3
        #grad2 = a1.T @ delta2
        grad1 = X_bias.T @ delta1



        # Update the weights:
        if self.solver == 'sgd':
            self.m1 = self.momentum * self.m1 - self.lr * (grad1 / self.batch_size + self.alpha*self.weights1)
            self.m3 = self.momentum * self.m3 - self.lr * (grad3 / self.batch_size + self.alpha*self.weights3)
            # second layer not updated, is jus pooling
            self.weights1 +=  self.m1
            self.weights3 +=  self.m3
            
        else:                
            self.weights1 -= self.lr * grad1
            self.weights3 -= self.lr * grad3

            

    def update_lr(self, time_step):
        """
        decrease learnign rate after patience"""
        #self.lr = self.lr /  (self.epochs ** self.power_t)
        #tau = self.tau
        #patience = 5
        #if self.epochs < tau and self.epochs > patience:
            #self.lr = (1-self.epochs / tau) * self.lr_initial + self.epochs/tau * 0.01*self.lr_initial
        self.lr = self.lr_initial / pow(time_step,  self.power_t)
        self.lr = self.lr_initial / pow(time_step,  self.power_t)

    def forward(self, X):
        """ 
        Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        
      # One epoch
        # The forward step:
        z1 = X @ self.weights1
        a1 = self.activ[0](z1)
        #a1_bias = add_bias(a1, self.bias)

        # in questo caso no bias perchè faccio solo pooling
        dim2 = self.dim_in // 2
        z2 = a1 @ self.weights2
        a2 = self.activ[1](z2) # pooling
        a2 = add_bias(a2, self.bias)

        z3 = a2 @ self.weights3
        a3 = outputs = self.activ[2](z3)  # final output with logistic
       
        return 0, outputs

    def compute_loss(self, y, t):
        """
        Compute the binary cross-entropy loss.
        y: Predicted probabilities (after sigmoid), shape (N, 1)
        t: Target values (0 or 1)
        """
        eps = 1e-12  # to avoid log(0)
        y = np.clip(y, eps, 1 - eps)
        return -np.mean(t * np.log(y) + (1 - t) * np.log(1 - y))

    def predict_probabilities(self, x):
        _, prob = self.forward(add_bias(x, self.bias))
        return prob
    
    def predict(self, X):
        """Predict the class for the members of X"""
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score = forw[:, 0]
        return (score > 0.5)
    
    def score(self, X, t):
        '''return accuracy of X against label t'''
        # predicted values:
        y = self.predict(X)
        return accuracy(y, t)


