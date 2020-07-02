'''
Standalone function to plot c+ and c- as functions of m and \Delta
'''

import numpy as np
import matplotlib.pyplot as plt

m = 4 #6


c_bar = np.linspace(0.0001,0.9999,100)

#check function for c_0
def compute_c0(c):
    #return (1 - c) * np.exp( - Delta / 7) + (1 - np.exp( - Delta / 7)) * (1 - np.exp(-2 * (1-c) * m))
    return (np.exp(c*Delta) - 1) / (np.exp(Delta) - 1)


# check function for delta_0:
def compute_delta0(c):
    return (1 - c ** m) / (1 - c)

def compute_s(c,Delta):
    s = np.exp(-Delta/7)*((np.exp(Delta/7) - 1) * np.exp(2 * (c-1) * m) + c)
    return s


# compute the values for c_minus
def compute_c_minus(c,Delta):
    # Eq. 40
    this_s = compute_s(c,Delta)
    this_delta_0 = compute_delta0(this_s)

    c_min = (np.exp(c * this_delta_0 * Delta) -1) / (np.exp(this_delta_0*Delta) - 1)
    return c_min


def compute_c_m(xi):
    return (1+ np.exp(-m*xi))**(-1/m)

def compute_xi_m(c):
    return 1/m * np.log(c**m /(1-c**m))

def compute_c_plus(c,Delta):
    '''
    :param c: c_minus
    :return:
    Eq. 13
    '''
    this_xi_m = compute_xi_m(c)

    this_c_plus = compute_c_m(this_xi_m+Delta)

    return this_c_plus

style = ['-','.','-.']

# define figure size
plt.figure(figsize=(11,6))
plt.rc('font', size=15)

i=0
for Delta in [0.1,2,10]:

    c_minus = compute_c_minus(c_bar, Delta)
    c_plus = compute_c_plus(c_minus, Delta)

    plt.plot(c_bar, c_minus, 'b'+style[i])
    plt.plot(c_bar, c_plus, 'r'+style[i])
    i=i+1

plt.show()







