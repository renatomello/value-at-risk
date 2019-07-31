import csv
import numpy as np

from numpy import sqrt, exp, pi

from scipy.special import erfcinv, betaincinv
from scipy.stats.distributions import  t

#################################################################
#################################################################

path = '/home/renato/Dropbox/Machine Learning/PLD/VaR/'

#################################################################
#################################################################
def csv_to_array(filename, type_):
	with open(filename, 'r') as f:
		next(f)
		reader = csv.reader(f)
		list_ = list(reader)
	f.close()
	
	return np.array(list_).astype(type_)

#################################################################
#################################################################

def gaussian(x, K, x0, sigma):
    return K * exp(-(x-x0)**2/(2*sigma**2))

#################################################################
#################################################################

def VaR_gaussian(quantity, mean, stddev, alpha, log_returns = True):
    var_log = mean - sqrt(2) * stddev * erfcinv(2 * alpha)
    
    if log_returns == False:
        var = var_log
    else:
        var = quantity * (exp(var_log) -1)
    
    return var

#################################################################
#################################################################

def CVaR_gaussian(quantity, mean, stddev, alpha, log_returns = True):
    
    K = 2 * mean - 2 * alpha * mean + sqrt(2 / pi) * stddev * exp(erfcinv(2 * alpha)**2)
    N = 2 - 2 * alpha
    
    cvar_log = K / N
    
    if log_returns == False:
        cvar = cvar_log
    else:
        cvar = quantity * (exp(cvar_log) - 1)
    
    return cvar 

#################################################################
#################################################################

def student_t(x, K, mean, stddev, nu):
    return K * t.pdf(x, nu, mean, stddev)

#################################################################
#################################################################

def VaR_T(mean, stddev, nu, alpha):
    var = mean - stddev * sqrt(nu) * sqrt(-1 + 1/betaincinv(nu / 2, 1/2, 2 * alpha))
    return var

#################################################################
#################################################################