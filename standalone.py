import numpy as np
import matplotlib.pyplot as plt

m = 4.8
Delta = 90

c_bar = np.linspace(0,1,50)

# check function for c_0
def compute_c0(c, Delta):
    return (1 - c) * np.exp( - Delta / 7) + (1 - np.exp( - Delta / 7)) * (1 - np.exp(-2 * (1-c) * m))

c_0 = compute_c0(c_bar, Delta)

plt.figure()
plt.plot(c_bar,c_0)
plt.title('c_0')

# check function for delta_0:
def compute_delta0(c):
    return (1 - c ** m) / (1 - c)

delta_0 = compute_delta0(c_0)

plt.figure()
plt.plot(c_bar,delta_0)
plt.title('delta_0')


# compute the values for c_minus
def compute_c_minus(c,Delta):
    this_c_0 = compute_c0(c,Delta)
    this_delta_0 = compute_delta0(c= 1-this_c_0)

    return (np.exp(c * this_delta_0 * Delta) -1 ) / (np.exp(this_delta_0*Delta) - 1)


def compute_c_plus(c,Delta):

    this_c_minus = compute_c_minus(c,Delta)

    this_c_plus = (this_c_minus* np.exp(Delta)) / (1 + this_c_minus * (np.exp(Delta) - 1))

    return this_c_plus, this_c_minus


c_plus, c_minus = compute_c_plus(c_bar,Delta)

plt.figure()
plt.plot(c_bar, c_minus)
plt.title('C_minus')

plt.figure()
plt.plot(c_bar, c_plus)
plt.title('C_plus')

plt.show()









