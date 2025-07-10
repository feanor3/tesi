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
    
    def __init__(self, bias=-1, dim_hidden = 6, tolerance=1e-4, activation='relu', solver='sgd', lr = 1e-4, batch_size=100, epochs=100, n_epochs_no_update = 2, momentum = 0.9):
        """Intialize the hyperparameters"""
        self.bias = bias
        # Dimensionality of the hidden layer
        self.dim_hidden = dim_hidden
        self.epochs = 0
        self.tol = tolerance
        self.lr = lr
        self.epochs = epochs
        self.n_epochs_no_update = n_epochs_no_update
        self.solver = solver
        self.batch_size = batch_size
        self.momentum = momentum
        
        if activation == 'relu':
            self.activ = relu     
            self.activ_diff = relu_diff
        elif activation == 'logistic':
            self.activ = logistic     
            self.activ_diff = logistic_diff

        



        
    def forward(self, X):
        """ 
        Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        
        hidden_outs = self.activ(X @ self.weights1)
        hidden_outs_bias = add_bias(hidden_outs, self.bias)
        outputs = logistic(hidden_outs_bias @ self.weights2)
       
        return hidden_outs_bias, outputs

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
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Initialize the weights
        self.weights1 = (np.random.rand(
            dim_in + 1, 
            self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand( #final layer
            self.dim_hidden+1, 
            dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        
        
        for e in range(self.epochs):

            if time.time()-t_start > 60*2:
                print("time exceedeed")
                break
            
            if self.solver == 'sgd':
                # selecting random elements of X_train_bias as batch
                idx = np.random.randint(0, self.N, self.batch_size)
                X_batch = X_train[idx, :]
                T_batch = T_train[idx]

                self.update(X_batch, T_batch)
            else:
                self.update(X_train, t_train)

            

            if X_val is not None:
                z = self.compute_loss(self.predict_probabilities(X_val), t_val)
                val_loss.append(z)
                val_acc.append(accuracy(self.predict(X_val), t_val))

                if e > 0 and abs(val_loss[e-1]-val_loss[e]) < self.tol:
                    current_epochs_no_update += 1
                    if current_epochs_no_update > self.n_epochs_no_update:
                        #print("Classifier trained for epochs: ", #self.epochs)
                        break
                    
            self.epochs += 1


    def update(self, X, t_train):
            ''''X unbiased data, T_train gold value'''

            X_bias = add_bias(X, self.bias)
            # One epoch
            # The forward step:
            hidden_outs, outputs = self.forward(X_bias)
            # The delta term on the output node:
            out_deltas = (outputs - t_train)
            # The delta terms at the output of the hidden layer:
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The deltas at the input to the hidden layer:
            #hiddenout_diffs = add_bias(hiddenout_diffs, self.bias)
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * 
                                self.activ_diff(hidden_outs[:, 1:]))  

            # Update the weights:
            self.weights2 -= self.lr * hidden_outs.T @ out_deltas
            self.weights1 -= self.lr * X_bias.T @ hiddenact_deltas 

            # Loss and accuracy
            acc = accuracy(self.predict(X), t_train)
            self.train_acc.append(acc)

            loss = self.compute_loss(self.predict_probabilities(X), t_train)
            self.train_loss.append(loss)

    def compute_loss(self, y, t):
        """
        Compute the mean squared error loss.
        y: Predicted values
        t: Target values
        """
        # Mean Squared Error, t = target, y = predicted
        a = 0.5 * np.mean((y - t) ** 2)
        return a

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
        '''return accuracy of X againt label t'''
        # predicted values:
        y = self.predict(X)
        return accuracy(y, t)

    
