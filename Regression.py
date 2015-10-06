# standard distribution imports
import copy
import warnings

# external packages imports
import numpy as np
import theano
import theano.tensor as T

# TestUtils imports
from Collection import is_number, is_integer

class PolynomialRegression(object):
    """
    Polynomial regression model and data training engine using Theano package.
    
    :Parameters:
        #. features (numpy.array): training features array of dimension (m,n).
        #. features (numpy.array): training results array of dimension (m,).
        #. learningRate (number): the learning rate.
        #. deg (integer>1, list):  the degree of the polynomial.\n
               If Integer an polynomail of degree deg is created.\n
               If list, a list of all degrees of the polynomial.
    """
    def __init__(self, features, target, learningRate=1e-5, deg=2):
        # append bias term, 1 vector to features
        self.features = features
        self.target   = target
        # set learning rate
        self.learningRate = learningRate
        # init training model
        self.weights = ()
        self.model   = None
        self.train   = None
        self.costEvolution = []
        self.totalNumberOfEpochs = 0
        # initialize to linear
        self.set_polynomial(deg=deg)

    def _create_weight(self, shape=None, scale=0.01):
        """Creates a random weight array of shape shape and scale it with the scale vector.
        assign the array to a theano shared variables.
        """
        if shape is None:
            shape = (self.features.shape[1],)
        if isinstance(shape, np.ndarray):
            return theano.shared( np.asarray(shape, dtype=np.float64) )
        else:
            return theano.shared( np.asarray(np.random.randn(*shape)*scale, dtype=np.float64) )
     
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
        L = self.features.shape[0]
        newX = np.hstack( (np.ones(L).reshape(L, 1),  self.features) )
        transpose_x = np.transpose(newX)
        transpose_x_dot_x = np.dot(transpose_x, newX)
        transpose_x_dot_x_inv = np.linalg.inv(transpose_x_dot_x)
        transpose_x_dot_y = np.dot( transpose_x, self.target )
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
            X = self.features
        # get weights
        thetas = self.compute_linear_regression_weights()
        L = X.shape[0]
        newX = np.hstack( (np.ones(L).reshape(L, 1),  X) )
        # get guessed values
        guessedValues = []
        for idx in range(L):
            guessedValues.append( np.dot(thetas, newX[idx,:]) )
        return thetas, np.array(guessedValues)
            
    
    def set_polynomial(self, learningRate=None, weights=None, scale = None, fixedWeights=None, deg=(1,2) ):
        """
        Sets the model polynomial as the following form with no cross terms.\n
         model = W0 + W10*X[:,0]    + W11*X[:,1]    + W12*X[:,2]    + ... + W1n*X[:,n] 
                    + W20*X[:,0]**2 + W21*X[:,1]**2 + W22*X[:,2]**2 + ... + W2n*X[:,n]**2 
                    + ... 
    
        :Parameters:
            #. learningRate (number, list): the learning rate number of a list equal to the deg+1.
            #. weights (None, list of numpy.array): if None weight will be initialized, 
               if list of arrays, the length of the list defines the deg of the polynomial
            #. scale (number): Only used when weights are None. It is used to scale weights upon generation.
            #. deg (integer>1, list):  the degree of the polynomial.\n
               If Integer an polynomail of degree deg is created.\n
               If list, a list of all degrees of the polynomial.
        """
        # set degrees
        if isinstance(deg, (list, set, tuple)):
            deg = sorted(deg)
            self.degrees = []
            for d in deg:
                assert is_number(d), "a degree must be a numbder"
                d = float(d)
                assert d>0, "a degree must be positive bigger than 0"
                self.degrees.append(d)
        else:
            assert is_integer(deg), "deg must be a list or a numbder"
            deg = int(deg)
            assert deg>=0, "deg must be positive integer"
            self.degrees = range(1,deg+1)
        # declare variables
        X = T.vector()
        Y = T.scalar()
        # set weights
        if weights is None:
            w0 = self._create_weight( (1,) )
            weights = [w0]
            for deg in self.degrees:
                if scale is None:
                    s = 1./np.max(np.abs(self.features**deg), 0)
                    w = self._create_weight( shape=s.reshape((-1,)) )
                else:
                    w = self._create_weight( (self.features.shape[1],), scale=scale )
                weights.append(w)
        else:
            for idx, w in enumerate(weights):
                w = np.asarray(w, dtype=np.float64)
                if idx == 0:
                    assert len(w) == 1, "first weight must be for the offset"
                else:
                    assert len(w) == self.features.shape[1] 
                weights[idx] = theano.shared( w ) 
        self.weights = weights
        assert len(self.degrees) == len(self.weights)-1, "degrees and weights length is not matching"
        # set fixed model
        if fixedWeights is None:
            fixedWeights={}
        assert isinstance(fixedWeights, dict), "fixedWeights must be None or a dictionary"
        for deg, w in fixedWeights.items():
            assert is_number(deg), "fixedWeights keys must be numbers"
            deg = float(deg)
            assert deg > 0, "fixedWeights keys must be positive"
            if deg in self.degrees:
                warnings.warn("fixed degree %i is also set as a degree to fit"%deg)
            w = np.asarray(w, dtype=np.float64)
            assert w.shape == (self.features.shape[1],), "w must be of shape %s"%str( (self.features.shape[1],) )
            fixedWeights[deg] = w
        self.fixedWeights = fixedWeights
        # create model
        if not len(self.fixedWeights):
            self.model = lambda X, W: sum( [ W[idx] if idx == 0 else T.dot(X**self.degrees[idx-1],W[idx]) for idx in range(len(W)) ] )
        else:
            self.model = lambda X, W: sum( [T.dot(X**deg,w) for deg, w in self.fixedWeights.items()] ) + sum( [ W[idx] if idx == 0 else T.dot(X**self.degrees[idx-1],W[idx]) for idx in range(len(W)) ] )
        guessedY = self.model(X, self.weights)
        # set learning rate
        if learningRate is not None:
            if not isinstance(learningRate, float):
                assert len(learningRate) == len(self.weights), "learning rate must have the same dimension as weights %i"%len(self.weights)
                LR = [float(lr) for lr in learningRate]
            else:
                LR = [learningRate for _ in self.weights]
        elif isinstance(self.learningRate, float):
            LR = [self.learningRate for _ in self.weights] 
        elif len(self.learningRate) == len(self.weights):
            LR = self.learningRate
        else:
            raise Exception("learning rate must be a float or a list with length equal to the number of weights")
        self.learningRate = LR
        # cost function
        cost = T.mean( T.sqr( Y-guessedY ) )
        # gradient
        grads = T.grad(cost=cost, wrt=self.weights)
        # update
        update = [ [ self.weights[idx],self.weights[idx]-grads[idx]*LR[idx] ] 
                    for idx in range(len(self.weights)) ]
        # create train function
        self.train = theano.function( inputs=[X,Y],
                                      outputs=cost,
                                      updates=update,
                                      allow_input_downcast=True)
                                      
    def get_fitted_weights_values(self):
        """
        get a list of all fitted weights values
        
        :Returns:
            #. weights (list): list of weights used in the learning process.
        """
        return [copy.deepcopy(w.get_value())for w in self.weights]
    
    def get_all_weights_dict(self):
        """
        get a list of all weights values
        
        :Returns:
            #. weights (dict): dictionary of two keys ('fixed', 'fitted').\n
               values are dictionaries as well conposed of {deg:values, ...}
        """
        weights = {'fixed':copy.deepcopy(self.fixedWeights), "fitted":{}}
        for idx, w in enumerate(self.weights):
            if idx == 0:
                weights["fitted"][0] = copy.deepcopy(w.get_value())
            else:
                weights["fitted"][self.degrees[idx-1]] = copy.deepcopy(w.get_value())            
        return weights
        
    def guess_values(self, X):
        """
        Guess the the values of the given features using the model in its current state.
        
        :Returns:
            #. X (numpy.array): the array of features
        """
        # compute guessed data
        thetas  = self.get_fitted_weights_values()
        guessed = self.model(X, thetas).eval().reshape((-1,))
        # return values
        return guessed
    
    
    def chi_square(self, guessed, target=None):
        """
        Get chi square of the given target values
        
        :Parameters:
            #. guessed (None, numpy.array): the array to compare with the target.
            #. target (None, numpy.array): the target array to compare with.
               If None, it will be set to target array.
            
        :Returns:
            #. chisquare (float): the chi square value.
        """
        if target is None:
            target = self.target
        return np.sqrt( np.sum((target-guessed)**2)/len(target))
        
    def r_squared(self, guessed, target=None):
        """
        Get r squared of the given target values
        
        :Parameters:
            #. guessed (None, numpy.array): the array to compare with the target.
            #. target (None, numpy.array): the target array to compare with.
               If None, it will be set to target array.
            
        :Returns:
            #. rsquared (float): the rsquared value.
        """
        if target is None:
            target = self.target
        meanY                = np.mean(target)
        totalSumOfSquares    = np.sum( (target-meanY)**2 )
        residualSumOfSquares = np.sum( (target-guessed)**2 )
        return 1-(residualSumOfSquares/totalSumOfSquares)

    def train_model(self, nepoch=1000, stopIfDiverging=True, minLearningSpeed=1e-5):
        """
        Train the model using the training data nepoch time.
        
        :Parameters:
            #. nepoch (integer): the number of epochs to runs.
            #. stopIfDiverging (bool): controls whether the cost is diverging and
               stops the training if happened.
            #. minLearningSpeed (float): stop learning when learning speed
               defined as diff(lastCost-cost) < minLearningSpeed
        """
        
        """When all the training data can't be loaded in memory,
        an easy solution can be pulling data randomly by chunks from a database.
        
        e.g. lets assume training data are stored in a SQL like database of 1e10 entries.
        
        # lets pull randomly 5000 data to train at every epoch
        ids = np.random.randint(low=1, high=1e10, size=5000)
        ids = str(tuple(ids))
        # create select query command
        command  = 'SELECT * FROM TABLENAME WHERE id IN %s'%ids
        features = databaseCursor.execute(command).fetchall() 
        # then run the training epochs on those features
        for _ in xrange(nepoch): 
            for idx in range(len(features)): 
                ...
        """    
        # initialize variables
        self.costEvolution = []
        self.totalNumberOfEpochs += nepoch
        # run epochs
        for _ in xrange(nepoch):
            trainedCost = 0
            for idx, y in enumerate(self.target):
                x = self.features[idx,:]
                cost = self.train(x,y)
                trainedCost += cost
            CE = trainedCost/len(self.target)    
            print "epoch %i (%.8f)"%(_ ,CE )
            self.costEvolution.append(CE)
            # check cost 
            if np.isnan(self.costEvolution[-1]):
                print "Weights diverged, no way to recover."
                break
            if len(self.costEvolution) == 1:
                continue
            costDiff = self.costEvolution[-2]-self.costEvolution[-1]
            if stopIfDiverging and costDiff < 0:
                print "Divergence detected. training stopped"
                break
            elif costDiff < minLearningSpeed:
                print "Slow learning. training stopped"
                break
        
    
