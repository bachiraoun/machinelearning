# standard distribution imports
import copy

# external packages imports
import numpy as np
import theano
import theano.tensor as T



class PolynomialRegression(object):
    """
    Polynomial regression model and data training engine using Theano package.
    
    :Parameters:
        #. trX (numpy.array): training features array of dimension (m,n).
        #. trX (numpy.array): training results array of dimension (m,).
        #. learningRate (number): the learning rate.
        #. normalize (boolean): whether to normalize the features.
    """
    def __init__(self, trX, trY, learningRate=1e-5, normalize=False):
        # append bias term, 1 vector to trX
        self.trX = trX
        self.trY = trY
        # normalize
        self.__normalize(normalize)
        # set learning rate
        self.learningRate = learningRate
        # init training model
        self.weights = ()
        self.model   = None
        self.train   = None
        self.costEvolution = []
        self.totalNumberOfEpochs = 0
        # initialize to linear
        self.set_polynomial()

    def __set_weight(self, shape, w=None, scale=0.01):
        """Creates a random weight array of shape shape and scale it with the scale vector.
        assign the array to a theano shared variables.
        """
        return theano.shared( np.asarray(np.random.randn(*shape)*scale, dtype=theano.config.floatX) )
    
    def __normalize(self, normalize):
        """
        not fully implemented yet ...
        """
        return
        self.normalized = normalize
        if not self.normalized:
            self.__subtract = 0.
            self.__divide   = 1. 
        elif self.normalized == "minmax":
            self.__subtract = np.min(self.trX)
            self.__divide = np.max(self.trX)-self.__subtract
        elif self.normalized == "meanstd":
            self.__subtract = np.mean(self.trX)
            self.__divide = np.std(self.trX)
        else:
            raise Exception("Unkown normalization method %s"%str(self.normalized ))
        self.trX = (self.trX - self.__subtract)/self.__divide
        
    def compute_linear_regression_weights(self):
        """
        Compute linear regression weights.
        linear regression model is the following:
        model = W0 + w1.X .
        
        weights are computed using the formula:
        W = inverse(transpose(X).X).transpose(X).Y
        
        :Returns:
            #. weights (integer): the weights array of dimension (m,2)
        """
        L = self.trX.shape[0]
        newX = np.hstack( (np.ones(L).reshape(L, 1),  self.trX) )
        transpose_x = np.transpose(newX)
        transpose_x_dot_x = np.dot(transpose_x, newX)
        transpose_x_dot_x_inv = np.linalg.inv(transpose_x_dot_x)
        transpose_x_dot_y = np.dot( transpose_x, self.trY )
        return np.dot(transpose_x_dot_x_inv, transpose_x_dot_y)
    
    def guess_using_linear_regression(self, X = None):
        """
        Compute the values of features X using the linear regression model.
        linear regression model is the following:
        model = W0 + w1.X .
        
        :Parameters:
            #. X (numpy.array): The features array X.

        :Parameters:
            #. weights (integer): the weights array of dimension (m,2) computed using compute_linear_regression_weights method.
            #. Y (numpy.array): the calculated outputs array of dimension (m,1)
        """
        if X is None:
            X = self.trX
        # get weights
        thetas = self.compute_linear_regression_weights()
        L = X.shape[0]
        newX = np.hstack( (np.ones(L).reshape(L, 1),  X) )
        # get guessed values
        guessedValues = []
        for idx in range(L):
            guessedValues.append( np.dot(thetas, newX[idx,:]) )
        return thetas, guessedValues
            
    def set_polynomial(self, learningRate=None, weights=None, deg=2):
        """
        Sets the model polynomial as the following form with no cross terms.
        model = W0 + w1.X + W2.X**2 + W3.X**3 + ...
    
        :Parameters:
            #. learningRate (number, list): the learning rate number of a list equal to the deg+1.
            #. weights (None, list of numpy.array): if None weight will be initialized, 
               if list of arrays, the length of the list defines the deg of the polynomial
            #. deg (integer>1):  the deg of the polynomial.
        """
        # declare variables
        X = T.vector()
        Y = T.scalar()
        # set weights
        if weights is None:
            w0 = self.__set_weight( (1,) )
            weights =[ self.__set_weight( (self.trX.shape[1],) ) for _ in range(deg)]
            weights.insert(0, w0)
        else:
            weights = tuple([theano.shared( np.asarray(w, dtype=theano.config.floatX) ) for w in weights])
        self.weights = weights
        # create model
        self.model = lambda X, W: sum( [W[idx] if idx == 0 else T.dot(X**idx,W[idx]) for idx in range(len(W)) ] )
        # set predict y given x model
        guessedY = self.model(X, self.weights)
        # set learning rate
        if learningRate is not None:
            if not isinstance(learningRate, float):
                assert len(learningRate) == len(self.weights), "training rate must have the same dimension as weights %i"%len(self.weights)
                LR = [float(lr) for lr in learningRate]
            else:
                LR = [learningRate for _ in self.weights]
        elif isinstance(self.learningRate, float):
            LR = [self.learningRate for _ in self.weights] 
        elif len(self.learningRate) == len(self.weights):
            LR = self.learningRate
        else:
            raise Exception("learning must be a float or a list with length equal to the number of weights")
        self.learningRate = LR
        # cost function
        cost = T.mean( T.sqr( Y-guessedY ) )
        # gradient
        grads = T.grad(cost=cost, wrt=self.weights)
        # update
        update = [  [self.weights[idx],self.weights[idx]-grads[idx]*LR[idx] ] for idx in range(len(self.weights)) ]
        # create train function
        self.train = theano.function( inputs=[X,Y],
                                      outputs=cost,
                                      updates=update,
                                      allow_input_downcast=True)
                            
    def get_weights_values(self):
        """
        get a list of all weights values
        
        :Returns:
            #. weights (list): list of weights used in the learning process.
        """
        return [copy.deepcopy(w.get_value())for w in self.weights]
        
    def guess_values(self, X):
        """
        Guess the the values of the given features using the model in its current state.
        
        :Returns:
            #. X (nump.array): the array of features
        """
        # compute guessed data
        thetas = self.get_weights_values()
        guessed = self.model(X, thetas).eval().reshape((-1,))
        # return values
        return guessed
        
    def train_model(self, nepoch=1000, stopIfDiverging=True, stopWhenDiffCost=1e-5):
        """
        Train the model using the training data nepoch time.
        
        :Parameters:
            #. nepoch (integer): the number of epochs to runs.
            #. stopIfDiverging (bool): controls whether the cost is diverging and
               stops the training if happened.
            #. stopWhenDiffCost (float): stop learning when learning speed
               defined as diff(lastCost-cost) < stopWhenDiffCost
        """
        
        ### When all the training data can't be loaded in memory,
        ### an easy solution can be pulling data randomly by chunks from a database.
        ### 
        ### e.g. lets assume training data are stored in a SQL like database of 1e10 entries.
        ### 
        ### # lets pull randomly 5000 data to train at every epoch
        ### ids = np.random.randint(low=1, high=1e10, size=5000)
        ### ids = str(tuple(ids))
        ### # create select query command
        ### command  = 'SELECT * FROM TABLENAME WHERE id IN %s'%ids
        ### features = databaseCursor.execute(command).fetchall() 
        ### # then run the training epochs on those features
        ### for _ in xrange(nepoch): 
        ###     for idx in range(len(features)): 
        ###         ...
        
        self.costEvolution = []
        self.totalNumberOfEpochs += nepoch
        # run epochs
        for _ in xrange(nepoch):
            trainedCost = 0
            for idx in range(len(self.trY)):
                x = self.trX[idx,:]
                y = self.trY[idx]
                trainedCost += self.train(x,y)
            CE = trainedCost/len(self.trY)    
            print "epoch %i (%.8f)"%(_ ,CE )
            self.costEvolution.append(CE)
            # check cost 
            if np.isnan(self.costEvolution[-1]):
                print "Weights diverged, now way to recover."
                break
            if len(self.costEvolution) == 1:
                continue
            costDiff = self.costEvolution[-2]-self.costEvolution[-1]
            if stopIfDiverging and costDiff < 0:
                print "Divergence detected. training stopped"
                break
            elif costDiff < stopWhenDiffCost:
                print "Slow learning. training stopped"
                break
    
    
