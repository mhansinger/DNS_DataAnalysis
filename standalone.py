'''
Standalone function to plot c+ and c- as functions of m and \Delta
'''

import numpy as np
import matplotlib.pyplot as plt

m = 4 #6


c_bar = np.linspace(0.0001,0.9999,100)

# #check function for c_0
# def compute_c0(c):
#     #return (1 - c) * np.exp( - Delta / 7) + (1 - np.exp( - Delta / 7)) * (1 - np.exp(-2 * (1-c) * m))
#     return (np.exp(c*Delta) - 1) / (np.exp(Delta) - 1)

# c_0 = compute_c0(c_bar)

# plt.figure()
# plt.plot(c_bar,c_0)
# plt.title('c_0')

# check function for delta_0:
def compute_delta0(c):
    return (1 - c ** m) / (1 - c)

def compute_s(c,Delta):
    s = np.exp(-Delta/7)*((np.exp(Delta/7) - 1) * np.exp(2 * (c-1) * m) + c)
    return s

#s = compute_s(Delta,c_bar,m)



#delta_0 = compute_delta0(c_0)

# plt.figure()
# plt.plot(c_bar,delta_0)
# plt.title('delta_0')


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

#style = ['b','b--','r','r--']
style = ['--','.','-.']

# define figure size
plt.figure(figsize=(11,6))
plt.rc('font', size=15)

i=0
for Delta in [0.5,2,10]:

    c_minus = compute_c_minus(c_bar, Delta)
    c_plus = compute_c_plus(c_minus, Delta)

    plt.plot(c_bar, c_minus, 'b'+style[i])
    plt.plot(c_bar, c_plus, 'r'+style[i])
    i=i+1

#plt.xlabel('c')
#plt.title('Vergleich mit Fig. 4')
plt.plot(c_bar,c_bar,'k-')

plt.savefig('plots_paper/c_plus_c_minus.eps')
plt.show()

#c_plus, c_minus = compute_c_plus(c_bar,Delta)

# plt.figure()
# plt.plot(c_bar, c_minus)
# plt.title('C_minus')
#
# plt.figure()
# plt.plot(c_bar, c_plus)
# plt.title('C_plus')

# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# ax = axs[0 ,0]
# ax.plot(c_bar,c_0)
# ax.set_title('c_0')
# ax.set_xlabel('c')
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# ax.grid()
#
# ax = axs[0,1]
# ax.plot(c_bar,delta_0)
# ax.set_title('delta_0')
# ax.set_xlabel('c')
# ax.set_xlim([0, 1])
# #ax.set_ylim([0, 1])
# ax.grid()
#
# ax = axs[1,0]
# ax.plot(c_bar,c_minus)
# ax.set_title('c_minus')
# ax.set_xlabel('c')
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# ax.grid()
#
# ax = axs[1,1]
# ax.plot(c_bar,c_plus)
# ax.set_title('c_plus')
# ax.set_xlabel('c')
# # ax.set_xlim([0, 1])
# # ax.set_ylim([0, 1])
# ax.grid()

# plt.tight_layout()
# # plt.grid()
#
# plt.show()









