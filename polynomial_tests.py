# standard distribution imports
import random

# external packages imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# TestUtils imports
from utils.Regression import PolynomialRegression

FEATURES = 25
NOISE    = 1.5
LEN_X    = 1000
COEFF0   = 2.5


def mean_chisquare(Y, guessed):
    return  np.sqrt( np.sum((Y-guessed)**2)/len(Y))

def plot(X, Y, guessedY, title=""):
    # compute residuals
    residuals = Y-guessedY

    # create figure
    FIG = plt.figure()
    FIG.patch.set_facecolor('white')
    grid = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[2,2,1])
    grid.update(left=0.1, right=0.95, wspace=0.05)
    sortAx = plt.subplot(grid[0, 0])
    dataAx = plt.subplot(grid[1, 0])
    resiAx  = plt.subplot(grid[2, 0])

    if X.shape[1] ==1:
        X = X.flatten()
        indexes = np.argsort(X)
        dataAx.plot(X[indexes], Y[indexes],'o',          label="Y")
        dataAx.plot(X[indexes], guessedY[indexes],       label="guessed")
        resiAx.plot(X[indexes], residuals[indexes], 'o', label='model residuals')
    else:
        dataAx.plot(Y,         'o', label="Y")
        dataAx.plot(guessedY,  'o', label="guessed")
        resiAx.plot(residuals, 'o', label='model residuals')
    # plot sorted axis
    indexes = np.argsort(Y)
    sortAx.plot(Y[indexes],'o',    label="sorted Y")
    sortAx.plot(guessedY[indexes], label="sorted guessed")
    # set legends
    dataAx.legend(frameon=False, loc='best')
    sortAx.legend(frameon=False, loc='best')
    resiAx.legend(frameon=False, loc='best')
    dataAx.set_ylim([None,max(Y)+1.25*(max(Y)-min(Y))])
    resiAx.set_ylim([min(residuals),max(residuals)+1.25*(max(residuals)-min(residuals))])
    # title
    FIG.suptitle(title)
    # show
    plt.show()


# ###################################################################################### #
# #################################### CREATE MODEL #################################### #
    
# training functions with some added functionalities
def train_poly1(X,Y, nepocs = 200, weights=None, minLearningSpeed= 1e-5, learningRate = [1e-3, 1e-5]):
    PR = PolynomialRegression(X,Y) 
    PR.set_polynomial(deg=1, weights=weights, learningRate = learningRate)
    PR.train_model(nepocs, minLearningSpeed=minLearningSpeed)
    guessedY = PR.guess_values(PR.features)
    weights  = PR.get_fitted_weights_values()
    return PR, weights, guessedY
    
def train_poly2(X,Y, nepocs = 200, weights=None, minLearningSpeed= 1e-5, learningRate = [1e-3, 1e-4, 1e-5]):
    PR = PolynomialRegression(X,Y) 
    PR.set_polynomial(deg=2, weights=weights, learningRate = learningRate)
    PR.train_model(nepocs, minLearningSpeed=minLearningSpeed)
    guessedY = PR.guess_values(PR.features)
    weights  = PR.get_fitted_weights_values()
    return PR, weights, guessedY
        
def train_poly3(X,Y, nepocs = 200, weights=None, minLearningSpeed= 1e-5, learningRate = [1e-3, 1e-4, 1e-5, 1e-6]):
    PR = PolynomialRegression(X,Y) 
    PR.set_polynomial(deg=3, weights=weights, learningRate = learningRate)
    PR.train_model(nepocs, minLearningSpeed=minLearningSpeed)
    guessedY = PR.guess_values(PR.features)
    weights  = PR.get_fitted_weights_values()
    return PR, weights, guessedY
        
def train_poly4(X,Y, nepocs = 200, weights=None, minLearningSpeed= 1e-5, learningRate = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]):
    PR = PolynomialRegression(X,Y) 
    PR.set_polynomial(deg=4, weights=weights, learningRate = learningRate)
    PR.train_model(nepocs, minLearningSpeed=minLearningSpeed)
    guessedY = PR.guess_values(PR.features)
    weights  = PR.get_fitted_weights_values()
    return PR, weights, guessedY

def train_poly5(X,Y, nepocs = 200, weights=None, minLearningSpeed= 1e-5, learningRate = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]):
    PR = PolynomialRegression(X,Y) 
    PR.set_polynomial(deg=5, weights=weights, learningRate = learningRate)
    PR.train_model(nepocs, minLearningSpeed=minLearningSpeed)
    guessedY = PR.guess_values(PR.features)
    weights  = PR.get_fitted_weights_values()
    return PR, weights, guessedY

    
# ###################################################################################### #
# ############################ CREATE 2nd DEGREE POLYNOMIAL ############################ #
degrees  = 2
# create coefficients 
coeffs = [ idx+1+np.arange(degrees) for idx in range(FEATURES)]
coeffs.insert(0, np.array([COEFF0]))

# create X and Y
X = 10.*( 1-2*np.random.random((LEN_X, FEATURES)) )
Y = coeffs[0][0] + sum( [ coeffs[fidx+1][degIdx]*X[:,fidx]**(degIdx+1)
                          for fidx in range(FEATURES)
                          for degIdx in range(degrees) ] )                      

# add noise to Y
Y += NOISE*(1-2*np.random.random(Y.shape))
# train model and get weights
PR, weights, guessedY = train_poly2(X, Y, nepocs = 1000, minLearningSpeed=1e-5)#, learningRate = [1e-4, 1e-6, 1e-8])

meanChiSquare = mean_chisquare(Y, guessedY)
print "\nmean ChiSquare: ", meanChiSquare
print "coefficients-->weights:\n========================="
print "offset constant: " + str(coeffs[0][0]) + "-->" + str(weights[0][0])
for degIdx in range(1, degrees+1):
    W = weights[degIdx]
    for fidx in range(FEATURES):
        print "deg %i, feature %i: "%(degIdx,fidx) + str(coeffs[fidx+1][degIdx-1]) + "-->" + str(W[fidx])

# plot
meanChiSquare = PR.chi_square(guessed = guessedY, target=Y)
rSquared      = PR.r_squared(guessed = guessedY, target=Y)
plot(X, Y, guessedY, title="%i features - deg %i polynomial ($\chi^{2}=%.6f ; R^{2}=%.6f$)"%(FEATURES, degrees, meanChiSquare, rSquared) )


# ###################################################################################### #
# ############################ CREATE 3rd DEGREE POLYNOMIAL ############################ #
degrees  = 3
# create coefficients 
coeffs = [ idx+1+np.arange(degrees) for idx in range(FEATURES)]
coeffs.insert(0, np.array([COEFF0]))

# create X and Y
X = 10.*( 1-2*np.random.random((LEN_X, FEATURES)) )
Y = coeffs[0][0] + sum( [ coeffs[fidx+1][degIdx]*X[:,fidx]**(degIdx+1)
                          for fidx in range(FEATURES)
                          for degIdx in range(degrees) ] )           
# add noise to Y
Y += NOISE*(1-2*np.random.random(Y.shape))
# train model and get weights
PR, weights, guessedY = train_poly3(X, Y, nepocs = 1000, minLearningSpeed=1e-5, learningRate = [1e-4, 1e-5, 1e-6, 1e-8])

meanChiSquare = mean_chisquare(Y, guessedY)
print "\nmean ChiSquare: ", meanChiSquare
print "coefficients-->weights:\n========================="
print "offset constant: " + str(coeffs[0][0]) + "-->" + str(weights[0][0])
for degIdx in range(1, degrees+1):
    W = weights[degIdx]
    for fidx in range(FEATURES):
        print "deg %i, feature %i: "%(degIdx,fidx) + str(coeffs[fidx+1][degIdx-1]) + "-->" + str(W[fidx])

# plot
meanChiSquare = PR.chi_square(guessed = guessedY, target=Y)
rSquared      = PR.r_squared(guessed = guessedY, target=Y)
plot(X, Y, guessedY, title="%i features - deg %i polynomial ($\chi^{2}=%.6f ; R^{2}=%.6f$)"%(FEATURES, degrees, meanChiSquare, rSquared) )

   
  
# ###################################################################################### #
# ############################ CREATE 4th DEGREE POLYNOMIAL ############################ #
degrees  = 4
# create coefficients 
coeffs = [ idx+1+np.arange(degrees) for idx in range(FEATURES)]
coeffs.insert(0, np.array([COEFF0]))

# create X and Y
X = 10.*( 1-2*np.random.random((LEN_X, FEATURES)) )
Y = coeffs[0][0] + sum( [ coeffs[fidx+1][degIdx]*X[:,fidx]**(degIdx+1)
                          for fidx in range(FEATURES)
                          for degIdx in range(degrees) ] )           
# add noise to Y
Y += NOISE*(1-2*np.random.random(Y.shape))
# train model and get weights
PR, weights, guessedY = train_poly4(X, Y, nepocs = 1000, minLearningSpeed=1e-5, learningRate = [1e-4, 1e-5, 1e-6, 1e-8, 1e-9])
#PR, weights, guessedY = train_poly3(X, Y, nepocs = 1000, minLearningSpeed=1e-5, learningRate = [1e-4, 1e-5, 1e-6, 1e-8])


meanChiSquare = mean_chisquare(Y, guessedY)
print "\nmean ChiSquare: ", meanChiSquare
print "coefficients-->weights:\n========================="
print "offset constant: " + str(coeffs[0][0]) + "-->" + str(weights[0][0])
for degIdx in range(1, degrees+1):
    W = weights[degIdx]
    for fidx in range(FEATURES):
        print "deg %i, feature %i: "%(degIdx,fidx) + str(coeffs[fidx+1][degIdx-1]) + "-->" + str(W[fidx])

# plot
meanChiSquare = PR.chi_square(guessed = guessedY, target=Y)
rSquared      = PR.r_squared(guessed = guessedY, target=Y)
plot(X, Y, guessedY, title="%i features - deg %i polynomial ($\chi^{2}=%.6f ; R^{2}=%.6f$)"%(FEATURES, degrees, meanChiSquare, rSquared) )

 
 
 
# ###################################################################################### #
# ############################ CREATE 5th DEGREE POLYNOMIAL ############################ #
degrees  = 5
# create coefficients 
coeffs = [ idx+1+np.arange(degrees) for idx in range(FEATURES)]
coeffs.insert(0, np.array([COEFF0]))

# create X and Y
X = 10.*( 1-2*np.random.random((LEN_X, FEATURES)) )
Y = coeffs[0][0] + sum( [ coeffs[fidx+1][degIdx]*X[:,fidx]**(degIdx+1)
                          for fidx in range(FEATURES)
                          for degIdx in range(degrees) ] )                      
# add noise to Y
Y += NOISE*(1-2*np.random.random(Y.shape))
# train model and get weights
PR, weights, guessedY = train_poly5(X, Y, nepocs = 1000, minLearningSpeed=1e-5, learningRate = [1e-6,1e-7,1e-8,1e-9,1e-10, 1e-11])
# print results
meanChiSquare = mean_chisquare(Y, guessedY)
print "\nmean ChiSquare: ", meanChiSquare
print "coefficients-->weights:\n========================="
print "offset constant: " + str(coeffs[0][0]) + "-->" + str(weights[0][0])
for degIdx in range(1, degrees+1):
    W = weights[degIdx]
    for fidx in range(FEATURES):
        print "deg %i, feature %i: "%(degIdx,fidx) + str(coeffs[fidx+1][degIdx-1]) + "-->" + str(W[fidx])

# plot
meanChiSquare = PR.chi_square(guessed = guessedY, target=Y)
rSquared      = PR.r_squared(guessed = guessedY, target=Y)
plot(X, Y, guessedY, title="%i features - deg %i polynomial ($\chi^{2}=%.6f ; R^{2}=%.6f$)"%(FEATURES, degrees, meanChiSquare, rSquared) )

 
