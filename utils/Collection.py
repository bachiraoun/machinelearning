# standard libraries imports
from random import random  as generate_random_float   
from random import randint as generate_random_integer 

# external libraries imports
import numpy as np     
     
   
def is_number(number):
    """
    check if number is convertible to float.
    
    :Parameters:
        #. number (str, number): input number
                   
    :Returns:
        #. result (bool): True if convertible, False otherwise
    """
    if isinstance(number, (int, long, float, complex)):
        return True
    try:
        float(number)
    except:
        return False
    else:
        return True
        
def is_integer(number, precision=10e-10):
    """
    check if number is convertible to integer.
    
    :Parameters:
        #. number (str, number): input number
        #. precision (number): To avoid floating errors, a precision should be given.
                   
    :Returns:
        #. result (bool): True if convertible, False otherwise
    """
    if isinstance(number, (int, long)):
        return True
    try:
        number = float(number)
    except:
        return False
    else:
        if np.abs(number-int(number)) < precision:
            return True
        else:
            return False
       
           
        
class RandomFloatGenerator(object):
    """
    Generate random float number between a lower and an upper limit.
    
    :Parameters:
        #. lowerLimit (number): The lower limit allowed.
        #. upperLimit (number): The upper limit allowed.
    """
    def __init__(self, lowerLimit, upperLimit):
         self.__lowerLimit = None
         self.__upperLimit = None
         self.set_lower_limit(lowerLimit)
         self.set_upper_limit(upperLimit)
         
    @property
    def lowerLimit(self):
        """The lower limit of the number generation."""
        return self.__lowerLimit
        
    @property
    def upperLimit(self):
        """The upper limit of the number generation."""
        return self.__upperLimit
        
    @property
    def rang(self):
        """The range defined as upperLimit-lowerLimit."""
        return self.__rang
        
    def set_lower_limit(self, lowerLimit):   
        """
        Set lower limit.
        
        :Parameters:
            #. lowerLimit (number): The lower limit allowed.
        """
        assert is_number(lowerLimit)
        self.__lowerLimit = np.float(lowerLimit) 
        if self.__upperLimit is not None:
            assert self.__lowerLimit<self.__upperLimit
            self.__rang = np.float(self.__upperLimit-self.__lowerLimit)
    
    def set_upper_limit(self, upperLimit):
        """
        Set upper limit.
        
        :Parameters:
            #. upperLimit (number): The upper limit allowed.
        """
        assert is_number(upperLimit)
        self.__upperLimit = np.float(upperLimit)
        if self.__lowerLimit is not None:
            assert self.__lowerLimit<self.__upperLimit
            self.__rang = np.float(self.__upperLimit-self.__lowerLimit)
        
    def generate(self):
        """Generate a random float number between lowerLimit and upperLimit."""
        return np.float(self.__lowerLimit+generate_random_float()*self.__rang)


def gaussian(x, center=0, FWHM=1, normalize=True, check=True):
    """
    Compute the normal distribution or gaussian distribution of a given vector.
    The probability density of the gaussian distribution is:
    :math:`f(x,\\mu,\\sigma) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{\\frac{-(x-\\mu)^{2}}{2\\sigma^2}}`
     
    Where:\n
    * :math:`\\mu` is the center of the gaussian, it is the mean or expectation of the distribution it is called the distribution's median or mode. 
    * :math:`\\sigma` is its standard deviation.
    * :math:`FWHM=2\\sqrt{2 ln 2} \\sigma` is the Full Width at Half Maximum of the gaussian. 
    
    :Parameters:
        #. x (numpy.ndarray): The vector to compute the gaussian
        #. center (number): The center of the gaussian.
        #. FWHM (number): The Full Width at Half Maximum of the gaussian.
        #. normalize(boolean): Whether to normalize the generated gaussian by :math:`\\frac{1}{\\sigma\\sqrt{2\\pi}}` so the integral is equal to 1. 
        #. check (boolean): whether to check arguments before generating vectors.
    """
    if check:
        assert is_number(center)
        center = np.float(center)
        assert is_number(FWHM)
        FWHM = np.float(FWHM)
        assert FWHM>0
        assert isinstance(normalize, bool)
    sigma       = FWHM/(2.*np.sqrt(2*np.log(2)))
    x = np.array(x)
    expKernel   = ((x-center)**2) / (-2*sigma**2)
    exp         = np.exp(expKernel)
    scaleFactor = 1.
    if normalize:
        scaleFactor /= sigma*np.sqrt(2*np.pi)
    return (scaleFactor * exp).astype(np.float)
    

def step_function(x, center=0, FWHM=0.1, height=1, check=True):
    """
    Compute a step function as the cumulative summation of a gaussian distribution of a given vector.
    
    :Parameters:
        #. x (numpy.ndarray): The vector to compute the gaussian
        #. center (number): The center of the step function which is the the center of the gaussian.
        #. FWHM (number): The Full Width at Half Maximum of the gaussian.
        #. height (number): The height of the step function.
        #. check (boolean): whether to check arguments before generating vectors.
    """
    if check:
        assert is_number(height)
        height = np.float(height)
    g  = gaussian(x, center=center, FWHM=FWHM, normalize=False, check=check)
    sf = np.cumsum(g)
    sf /= sf[-1]
    return (sf*height).astype(np.float)           
           
class BiasedRandomFloatGenerator(RandomFloatGenerator):
    """ 
    Generate biased random float number between a lower and an upper limit.
    To bias the generator at a certain number, a bias gaussian is added to the 
    weights scheme at the position of this particular number.
    
    .. image:: biasedFloatGenerator.png   
       :align: center 
       
    :Parameters:
        #. lowerLimit (number): The lower limit allowed.
        #. upperLimit (number): The upper limit allowed.
        #. weights (None, list, numpy.ndarray): The weights scheme. The length defines the number of bins and the edges.
           The length of weights array defines the resolution of the biased numbers generation.
           If None is given, ones array of length 10000 is automatically generated.
        #. biasRange(None, number): The bias gaussian range. 
           It must be smaller than half of limits range which is equal to (upperLimit-lowerLimit)/2
           If None, it will be automatically set to (upperLimit-lowerLimit)/5
        #. biasFWHM(None, number): The bias gaussian Full Width at Half Maximum. 
           It must be smaller than half of biasRange.
           If None, it will be automatically set to biasRange/10
        #. biasHeight(number): The bias gaussian maximum intensity.
        #. unbiasRange(None, number): The bias gaussian range. 
           It must be smaller than half of limits range which is equal to (upperLimit-lowerLimit)/2
           If None, it will be automatically set to biasRange.
        #. unbiasFWHM(None, number): The bias gaussian Full Width at Half Maximum. 
           It must be smaller than half of biasRange.
           If None, it will be automatically set to biasFWHM.
        #. unbiasHeight(number): The unbias gaussian maximum intensity.
           If None, it will be automatically set to biasHeight.
        #. unbiasThreshold(number): unbias is only applied at a certain position only when the position weight is above unbiasThreshold.
           It must be a positive number.
    """
    def __init__(self, lowerLimit, upperLimit, 
                       weights=None, 
                       biasRange=None, biasFWHM=None, biasHeight=1,
                       unbiasRange=None, unbiasFWHM=None, unbiasHeight=None, unbiasThreshold=1):
         # initialize random generator              
         super(BiasedRandomFloatGenerator, self).__init__(lowerLimit=lowerLimit, upperLimit=upperLimit)
         # set scheme 
         self.set_weights(weights)
         # set bias function
         self.set_bias(biasRange=biasRange, biasFWHM=biasFWHM, biasHeight=biasHeight)
         # set unbias function
         self.set_unbias(unbiasRange=unbiasRange, unbiasFWHM=unbiasFWHM, unbiasHeight=unbiasHeight, unbiasThreshold=unbiasThreshold)
    
    @property
    def originalWeights(self):
        """The original weights as initialized."""
        return self.__originalWeights
    
    @property
    def weights(self):
        """The current value weights vector."""
        weights = self.__scheme[1:]-self.__scheme[:-1]
        weights = list(weights)
        weights.insert(0,self.__scheme[0])
        return weights
        
    @property
    def scheme(self):
        """The numbers generation scheme."""
        return self.__scheme 
    
    @property
    def bins(self):
        """The number of bins that is equal to the length of weights vector."""
        return self.__bins
    
    @property
    def binWidth(self):
        """The bin width defining the resolution of the biased random number generation."""
        return self.__binWidth 
          
    @property
    def bias(self):
        """The bias step-function."""
        return self.__bias
    
    @property
    def biasGuassian(self):
        """The bias gaussian function."""
        return self.__biasGuassian
        
    @property
    def biasRange(self):
        """The bias gaussian extent range."""
        return self.__biasRange    
        
    @property
    def biasBins(self):
        """The bias gaussian number of bins."""
        return self.__biasBins
    
    @property
    def biasFWHM(self):
        """The bias gaussian Full Width at Half Maximum."""
        return self.__biasFWHM 

    @property
    def biasFWHMBins(self):
        """The bias gaussian Full Width at Half Maximum number of bins."""
        return self.__biasFWHMBins

    @property
    def unbias(self):
        """The unbias step-function."""
        return self.__unbias
    
    @property
    def unbiasGuassian(self):
        """The unbias gaussian function."""
        return self.__unbiasGuassian
        
    @property
    def unbiasRange(self):
        """The unbias gaussian extent range."""
        return self.__unbiasRange    
        
    @property
    def unbiasBins(self):
        """The unbias gaussian number of bins."""
        return self.__unbiasBins
    
    @property
    def unbiasFWHM(self):
        """The unbias gaussian Full Width at Half Maximum."""
        return self.__unbiasFWHM 

    @property
    def unbiasFWHMBins(self):
        """The unbias gaussian Full Width at Half Maximum number of bins."""
        return self.__unbiasFWHMBins

    def set_weights(self, weights=None):
        """
        Set generator's weights.
        
        :Parameters:
            #. weights (None, list, numpy.ndarray): The weights scheme. The length defines the number of bins and the edges.
               The length of weights array defines the resolution of the biased numbers generation.
               If None is given, ones array of length 10000 is automatically generated.
        """
        # set original weights
        if weights is None:
           self.__bins = 10000
           self.__originalWeights = np.ones(self.__bins)
        else:
            assert isinstance(weights, (list, set, tuple, np.ndarray))
            if isinstance(weights,  np.ndarray):
                assert len(weights.shape)==1
            wgts = []
            assert len(weights)>=100
            for w in weights:
                assert is_number(w)
                w = np.float(w)
                assert w>=0
                wgts.append(w)
            self.__originalWeights = np.array(wgts, dtype=np.float)
            self.__bins = len(self.__originalWeights)
        # set bin width
        self.__binWidth     = np.float(self.rang/self.__bins)
        self.__halfBinWidth = np.float(self.__binWidth/2.)
        # set scheme    
        self.__scheme = np.cumsum( self.__originalWeights )
    
    def set_bias(self, biasRange, biasFWHM, biasHeight):
        """
        Set generator's bias gaussian function
        
        :Parameters:
            #. biasRange(None, number): The bias gaussian range. 
               It must be smaller than half of limits range which is equal to (upperLimit-lowerLimit)/2
               If None, it will be automatically set to (upperLimit-lowerLimit)/5
            #. biasFWHM(None, number): The bias gaussian Full Width at Half Maximum. 
               It must be smaller than half of biasRange.
               If None, it will be automatically set to biasRange/10
            #. biasHeight(number): The bias gaussian maximum intensity.
        """
        # check biasRange
        if biasRange is None:
            biasRange = np.float(self.rang/5.)
        else:
            assert is_number(biasRange)
            biasRange = np.float(biasRange)
            assert biasRange>0
            assert biasRange<=self.rang/2.
        self.__biasRange = np.float(biasRange)
        self.__biasBins  = np.int(self.bins*self.__biasRange/self.rang)
        # check biasFWHM
        if biasFWHM is None:
            biasFWHM = np.float(self.__biasRange/10.)
        else:
            assert is_number(biasFWHM)
            biasFWHM = np.float(biasFWHM)
            assert biasFWHM>=0
            assert biasFWHM<=self.__biasRange/2.
        self.__biasFWHM     = np.float(biasFWHM) 
        self.__biasFWHMBins = np.int(self.bins*self.__biasFWHM/self.rang)
        # check height
        assert is_number(biasHeight)
        self.__biasHeight = np.float(biasHeight)
        assert self.__biasHeight>=0
        # create bias step function
        b = self.__biasRange/self.__biasBins
        x = [-self.__biasRange/2.+idx*b for idx in range(self.__biasBins) ]
        self.__biasGuassian = gaussian(x, center=0, FWHM=self.__biasFWHM, normalize=False)
        self.__biasGuassian -= self.__biasGuassian[0]
        self.__biasGuassian /= np.max(self.__biasGuassian)
        self.__biasGuassian *= self.__biasHeight
        self.__bias = np.cumsum(self.__biasGuassian)
    
    def set_unbias(self, unbiasRange, unbiasFWHM, unbiasHeight, unbiasThreshold):
        """
        Set generator's unbias gaussian function
        
        :Parameters:
            #. unbiasRange(None, number): The bias gaussian range. 
               It must be smaller than half of limits range which is equal to (upperLimit-lowerLimit)/2
               If None, it will be automatically set to biasRange.
            #. unbiasFWHM(None, number): The bias gaussian Full Width at Half Maximum. 
               It must be smaller than half of biasRange.
               If None, it will be automatically set to biasFWHM.
            #. unbiasHeight(number): The unbias gaussian maximum intensity.
               If None, it will be automatically set to biasHeight.
            #. unbiasThreshold(number): unbias is only applied at a certain position only when the position weight is above unbiasThreshold.
               It must be a positive number.
        """
        # check biasRange
        if unbiasRange is None:
            unbiasRange = self.__biasRange
        else:
            assert is_number(unbiasRange)
            unbiasRange = np.float(unbiasRange)
            assert unbiasRange>0
            assert unbiasRange<=self.rang/2.
        self.__unbiasRange = np.float(unbiasRange)
        self.__unbiasBins  = np.int(self.bins*self.__unbiasRange/self.rang)
        # check biasFWHM
        if unbiasFWHM is None:
            unbiasFWHM = self.__biasFWHM
        else:
            assert is_number(unbiasFWHM)
            unbiasFWHM = np.float(unbiasFWHM)
            assert unbiasFWHM>=0
            assert unbiasFWHM<=self.__unbiasRange/2.
        self.__unbiasFWHM     = np.float(unbiasFWHM) 
        self.__unbiasFWHMBins = np.int(self.bins*self.__unbiasFWHM/self.rang)
        # check height
        if unbiasHeight is None:
            unbiasHeight = self.__biasHeight
        assert is_number(unbiasHeight)
        self.__unbiasHeight = np.float(unbiasHeight)
        assert self.__unbiasHeight>=0
        # check unbiasThreshold
        assert is_number(unbiasThreshold)
        self.__unbiasThreshold = np.float(unbiasThreshold)
        assert self.__unbiasThreshold>=0
        # create bias step function
        b = self.__unbiasRange/self.__unbiasBins
        x = [-self.__unbiasRange/2.+idx*b for idx in range(self.__unbiasBins) ]
        self.__unbiasGuassian = gaussian(x, center=0, FWHM=self.__unbiasFWHM, normalize=False)
        self.__unbiasGuassian -= self.__unbiasGuassian[0]
        self.__unbiasGuassian /= np.max(self.__unbiasGuassian)
        self.__unbiasGuassian *= -self.__unbiasHeight
        self.__unbias = np.cumsum(self.__unbiasGuassian)
         
    def bias_scheme_by_index(self, index, scaleFactor=None, check=True):
        """
        Bias the generator's scheme using the defined bias gaussian function at the given index.
        
        :Parameters:
            #. index(integer): The index of the position to bias
            #. scaleFactor(None, number): Whether to scale the bias gaussian before biasing the scheme.
               If None, bias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if not self.__biasHeight>0: return
        if check:
            assert is_integer(index)
            index = np.int(index)
            assert index>=0
            assert index<=self.__bins
            if scaleFactor is not None:
                assert is_number(scaleFactor)
                scaleFactor = np.float(scaleFactor)
                assert scaleFactor>=0
        # get start indexes
        startIdx = index-int(self.__biasBins/2)
        if startIdx < 0:
            biasStartIdx = -startIdx
            startIdx = 0
            bias = np.cumsum(self.__biasGuassian[biasStartIdx:]).astype(np.float)
        else:
            biasStartIdx = 0
            bias = self.__bias
        # scale bias
        if scaleFactor is None:
            scaledBias = bias
        else:
            scaledBias = bias*scaleFactor         
        # get end indexes
        endIdx = startIdx+self.__biasBins-biasStartIdx
        biasEndIdx = len(scaledBias)
        if endIdx > self.__bins-1:
            biasEndIdx -= endIdx-self.__bins
            endIdx = self.__bins
        # bias scheme
        self.__scheme[startIdx:endIdx] += scaledBias[0:biasEndIdx]
        self.__scheme[endIdx:] += scaledBias[biasEndIdx-1]
        
    def bias_scheme_at_position(self, position, scaleFactor=None, check=True):
        """
        Bias the generator's scheme using the defined bias gaussian function at the given number.
        
        :Parameters:
            #. position(number): The number to bias.
            #. scaleFactor(None, number): Whether to scale the bias gaussian before biasing the scheme.
               If None, bias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if check:
            assert is_number(position)
            position = np.float(position)
            assert position>=self.lowerLimit
            assert position<=self.upperLimit
        index = np.int(self.__bins*(position-self.lowerLimit)/self.rang) 
        # bias scheme by index
        self.bias_scheme_by_index(index=index, scaleFactor=scaleFactor, check=check)
    
    def unbias_scheme_by_index(self, index, scaleFactor=None, check=True):
        """
        Unbias the generator's scheme using the defined bias gaussian function at the given index.
        
        :Parameters:
            #. index(integer): The index of the position to unbias
            #. scaleFactor(None, number): Whether to scale the unbias gaussian before unbiasing the scheme.
               If None, unbias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if not self.__unbiasHeight>0: return
        if check:
            assert is_integer(index)
            index = np.int(index)
            assert index>=0
            assert index<=self.__bins
            if scaleFactor is not None:
                assert is_number(scaleFactor)
                scaleFactor = np.float(scaleFactor)
                assert scaleFactor>=0
        # get start indexes
        startIdx = index-int(self.__unbiasBins/2)
        if startIdx < 0:
            biasStartIdx = -startIdx
            startIdx = 0
            unbias = self.__unbiasGuassian[biasStartIdx:]
        else:
            biasStartIdx = 0
            unbias = self.__unbiasGuassian
        # get end indexes
        endIdx = startIdx+self.__unbiasBins-biasStartIdx
        biasEndIdx = len(unbias)
        if endIdx > self.__bins-1:
            biasEndIdx -= endIdx-self.__bins
            endIdx = self.__bins
        # scale unbias
        if scaleFactor is None:
            scaledUnbias = unbias 
        else:
            scaledUnbias = unbias*scaleFactor
        # unbias weights
        weights = np.array(self.weights)
        weights[startIdx:endIdx] += scaledUnbias[0:biasEndIdx]
        # correct for negatives
        weights[np.where(weights<self.__unbiasThreshold)] = self.__unbiasThreshold
        # set unbiased scheme
        self.__scheme = np.cumsum(weights)
                                    
    def unbias_scheme_at_position(self, position, scaleFactor=None, check=True):
        """
        Unbias the generator's scheme using the defined bias gaussian function at the given number.
        
        :Parameters:
            #. position(number): The number to unbias.
            #. scaleFactor(None, number): Whether to scale the unbias gaussian before unbiasing the scheme.
               If None, unbias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if check:
            assert is_number(position)
            position = np.float(position)
            assert position>=self.lowerLimit
            assert position<=self.upperLimit
        index = np.int(self.__bins*(position-self.lowerLimit)/self.rang) 
        # bias scheme by index
        self.unbias_scheme_by_index(index=index, scaleFactor=scaleFactor, check=check)

    def generate(self):
        """Generate a random float number between the biased range lowerLimit and upperLimit."""
        # get position
        position = self.lowerLimit + self.__binWidth*np.searchsorted(self.__scheme, generate_random_float()*self.__scheme[-1]) + self.__halfBinWidth
        # find limits
        minLim = max(self.lowerLimit, position-self.__halfBinWidth)
        maxLim = min(self.upperLimit, position+self.__halfBinWidth)
        # generate number
        return minLim+generate_random_float()*(maxLim-minLim) + self.__halfBinWidth    
        

class RandomIntegerGenerator(object):
    """
    Generate random integer number between a lower and an upper limit.
    
    :Parameters:
        #. lowerLimit (number): The lower limit allowed.
        #. upperLimit (number): The upper limit allowed.
    """
    def __init__(self, lowerLimit, upperLimit):
         self.__lowerLimit = None
         self.__upperLimit = None
         self.set_lower_limit(lowerLimit)
         self.set_upper_limit(upperLimit)
    
    @property
    def lowerLimit(self):
        """The lower limit of the number generation."""
        return self.__lowerLimit
        
    @property
    def upperLimit(self):
        """The upper limit of the number generation."""
        return self.__upperLimit
        
    @property
    def rang(self):
        """The range defined as upperLimit-lowerLimit"""
        return self.__rang
            
    def set_lower_limit(self, lowerLimit):    
        """
        Set lower limit.
        
        :Parameters:
            #. lowerLimit (number): The lower limit allowed.
        """
        assert is_integer(lowerLimit)
        self.__lowerLimit = np.int(lowerLimit) 
        if self.__upperLimit is not None:
            assert self.__lowerLimit<self.__upperLimit
            self.__rang = self.__upperLimit-self.__lowerLimit+1
    
    def set_upper_limit(self, upperLimit):
        """
        Set upper limit.
        
        :Parameters:
            #. upperLimit (number): The upper limit allowed.
        """
        assert is_integer(upperLimit)
        self.__upperLimit = np.int(upperLimit) 
        if self.__lowerLimit is not None:
            assert self.__lowerLimit<self.__upperLimit
            self.__rang = self.__upperLimit-self.__lowerLimit+1
        
    def generate(self):
        """Generate a random integer number between lowerLimit and upperLimit."""
        return generate_random_integer(self.__lowerLimit, self.__upperLimit)

        
class BiasedRandomIntegerGenerator(RandomIntegerGenerator):
    """ 
    Generate biased random integer number between a lower and an upper limit.
    To bias the generator at a certain number, a bias height is added to the 
    weights scheme at the position of this particular number.
    
    .. image:: biasedIntegerGenerator.png   
       :align: center 
       
    :Parameters:
        #. lowerLimit (integer): The lower limit allowed.
        #. upperLimit (integer): The upper limit allowed.
        #. weights (None, list, numpy.ndarray): The weights scheme. The length must be equal to the range between lowerLimit and upperLimit.
           If None is given, ones array of length upperLimit-lowerLimit+1 is automatically generated.
        #. biasHeight(number): The weight bias intensity.
        #. unbiasHeight(None, number): The weight unbias intensity.
           If None, it will be automatically set to biasHeight.
        #. unbiasThreshold(number): unbias is only applied at a certain position only when the position weight is above unbiasThreshold.
           It must be a positive number.
    """
    def __init__(self, lowerLimit, upperLimit, 
                       weights=None, 
                       biasHeight=1, unbiasHeight=None, unbiasThreshold=1):
        # initialize random generator              
        super(BiasedRandomIntegerGenerator, self).__init__(lowerLimit=lowerLimit, upperLimit=upperLimit)
        # set weights
        self.set_weights(weights=weights)
        # set bias height
        self.set_bias_height(biasHeight=biasHeight)
        # set bias height
        self.set_unbias_height(unbiasHeight=unbiasHeight)
        # set bias height
        self.set_unbias_threshold(unbiasThreshold=unbiasThreshold)
    
    @property
    def originalWeights(self):
        """The original weights as initialized."""
        return self.__originalWeights
    
    @property
    def weights(self):
        """The current value weights vector."""
        weights = self.__scheme[1:]-self.__scheme[:-1]
        weights = list(weights)
        weights.insert(0,self.__scheme[0])
        return weights
        
    @property
    def scheme(self):
        """The numbers generation scheme."""
        return self.__scheme 
    
    @property
    def bins(self):
        """The number of bins that is equal to the length of weights vector."""
        return self.__bins
        
    def set_weights(self, weights):
        """
        Set the generator integer numbers weights.
        
        #. weights (None, list, numpy.ndarray): The weights scheme. The length must be equal to the range between lowerLimit and upperLimit.
           If None is given, ones array of length upperLimit-lowerLimit+1 is automatically generated.
        """
        if weights is None:
            self.__originalWeights = np.ones(self.upperLimit-self.lowerLimit+1)
        else:
            assert isinstance(weights, (list, set, tuple, np.ndarray))
            if isinstance(weights,  np.ndarray):
                assert len(weights.shape)==1
            wgts = []
            assert len(weights)==self.upperLimit-self.lowerLimit+1
            for w in weights:
                assert is_number(w)
                w = np.float(w)
                assert w>=0
                wgts.append(w)
            self.__originalWeights = np.array(wgts, dtype=np.float)
        # set bins
        self.__bins = len( self.__originalWeights )
        # set scheme    
        self.__scheme = np.cumsum( self.__originalWeights )
        
    def set_bias_height(self, biasHeight):
        """
        Set weight bias intensity.
        
        :Parameters:
            #. biasHeight(number): The weight bias intensity.
        """
        assert is_number(biasHeight)
        self.__biasHeight = np.float(biasHeight)
        assert self.__biasHeight>0
        
    def set_unbias_height(self, unbiasHeight):
        """
        Set weight unbias intensity.
        
        :Parameters:
            #. unbiasHeight(None, number): The weight unbias intensity.
               If None, it will be automatically set to biasHeight.
        """
        if unbiasHeight is None:
            unbiasHeight = self.__biasHeight
        assert is_number(unbiasHeight)
        self.__unbiasHeight = np.float(unbiasHeight)
        assert self.__unbiasHeight>=0
        
    def set_unbias_threshold(self, unbiasThreshold):
        """
        Set weight unbias threshold.
        
        :Parameters:
            #. unbiasThreshold(number): unbias is only applied at a certain position only when the position weight is above unbiasThreshold.
               It must be a positive number.
        """
        assert is_number(unbiasThreshold)
        self.__unbiasThreshold = np.float(unbiasThreshold)
        assert self.__unbiasThreshold>=0

    def bias_scheme_by_index(self, index, scaleFactor=None, check=True):
        """
        Bias the generator's scheme at the given index.
        
        :Parameters:
            #. index(integer): The index of the position to bias
            #. scaleFactor(None, number): Whether to scale the bias gaussian before biasing the scheme.
               If None, bias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if not self.__biasHeight>0: return
        if check:
            assert is_integer(index)
            index = np.int(index)
            assert index>=0
            assert index<=self.__bins
            if scaleFactor is not None:
                assert is_number(scaleFactor)
                scaleFactor = np.float(scaleFactor)
                assert scaleFactor>=0
        # scale bias
        if scaleFactor is None:
            scaledBias = self.__biasHeight
        else:
            scaledBias = self.__biasHeight*scaleFactor         
        # bias scheme
        self.__scheme[index:] += scaledBias
          
    def bias_scheme_at_position(self, position, scaleFactor=None, check=True):
        """
        Bias the generator's scheme at the given number.
        
        :Parameters:
            #. position(number): The number to bias.
            #. scaleFactor(None, number): Whether to scale the bias gaussian before biasing the scheme.
               If None, bias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if check:
            assert is_integer(position)
            position = np.int(position)
            assert position>=self.lowerLimit
            assert position<=self.upperLimit
        index = position-self.lowerLimit
        # bias scheme by index
        self.bias_scheme_by_index(index=index, scaleFactor=scaleFactor, check=check)

    def unbias_scheme_by_index(self, index, scaleFactor=None, check=True):
        """
        Unbias the generator's scheme at the given index.
        
        :Parameters:
            #. index(integer): The index of the position to unbias
            #. scaleFactor(None, number): Whether to scale the unbias gaussian before unbiasing the scheme.
               If None, unbias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if not self.__unbiasHeight>0: return
        if check:
            assert is_integer(index)
            index = np.int(index)
            assert index>=0
            assert index<=self.__bins
            if scaleFactor is not None:
                assert is_number(scaleFactor)
                scaleFactor = np.float(scaleFactor)
                assert scaleFactor>=0
        # scale unbias
        if scaleFactor is None:
            scaledUnbias = self.__unbiasHeight 
        else:
            scaledUnbias = self.__unbiasHeight*scaleFactor
        # check threshold
        if index == 0:
            scaledUnbias = max(scaledUnbias, self.__scheme[index]-self.__unbiasThreshold)   
        elif self.__scheme[index]-scaledUnbias < self.__scheme[index-1]+self.__unbiasThreshold:
            scaledUnbias = self.__scheme[index]-self.__scheme[index-1]-self.__unbiasThreshold
        # unbias scheme
        self.__scheme[index:] -= scaledUnbias
                                   
    def unbias_scheme_at_position(self, position, scaleFactor=None, check=True):
        """
        Unbias the generator's scheme using the defined bias gaussian function at the given number.
        
        :Parameters:
            #. position(number): The number to unbias.
            #. scaleFactor(None, number): Whether to scale the unbias gaussian before unbiasing the scheme.
               If None, unbias gaussian is used as defined.
            #. check(boolean): Whether to check arguments.
        """
        if check:
            assert is_integer(position)
            position = np.int(position)
            assert position>=self.lowerLimit
            assert position<=self.upperLimit
        index = position-self.lowerLimit
        # unbias scheme by index
        self.unbias_scheme_by_index(index=index, scaleFactor=scaleFactor, check=check)
        
    def generate(self):
        """Generate a random intger number between the biased range lowerLimit and upperLimit."""
        index = np.int( np.searchsorted(self.__scheme, generate_random_float()*self.__scheme[-1]) )
        return self.lowerLimit + index
    
