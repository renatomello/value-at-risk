#%%
import pandas as pd
import numpy as np

from math import log

from scipy.optimize import curve_fit
from scipy.stats.distributions import t
from scipy.special import erfcinv

from functions import *

import matplotlib.pyplot as plt
plt.style.use('ggplot')

#%%
preco_raw = csv_to_array(path + 'preco_medio_2018.csv', 'str')

datas = [preco_raw[k,0] for k in range(len(preco_raw[:,0]))][::-1]
preco = [] 
for k in preco_raw[:,-1]:
    preco.append(float(k))

log_returns = [log(preco[k] / preco[k - 1]) \
    for k in range(1, len(preco))][::-1]

del preco_raw

#%%
topos, bins, plot = plt.hist(log_returns, bins = 11, cumulative=False)

bins_center = [(bins[k] + bins[k+1])/2 for k in range(len(bins) - 1) ]

#%%
N = len(log_returns)
K, avg, stddev = 1, np.mean(log_returns), np.std(log_returns)

p0 = [K, avg, stddev]
popt, pcov = curve_fit(gaussian, bins_center, topos, p0 = p0)
perr = np.sqrt(np.diag(pcov))

#%%
K = popt[0]
mean = popt[1]
stddev = popt[2]

x = np.linspace(-0.5, 0.5, 100)

plt.plot(x, gaussian(x, *popt)) 
plt.hist(log_returns, bins = 11, cumulative = False)[2]

#%%
alpha = 0.01
quantity = 1000

var = VaR_gaussian(quantity, mean, stddev, alpha)
cvar = CVaR_gaussian(quantity, mean, stddev, alpha)
print(var, cvar)

#%%
#K_t, avg, stddev, nu = 1, np.mean(log_returns), np.std(log_returns), 1
p0_t = [1, np.mean(log_returns), np.std(log_returns), 1]
#p0_t = [K_t, avg, stddev, nu]
pop_t, pcov = curve_fit(student_t, bins_center, topos, p0 = p0_t)
perr = np.sqrt(np.diag(pcov))

#%%
mean_t = popt[1]
stddev_t = popt[2]

x = np.linspace(-0.5, 0.5, 100)

plt.plot(x, gaussian(x, *popt)) 
plt.hist(log_returns, bins = 11, cumulative = False)[2]
plt.plot(x, student_t(x, *pop_t)) 

#%%
