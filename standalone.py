import numpy as np
import matplotlib.pyplot as plt

m = 4.5
Delta = 2.8

c_bar = np.linspace(0,1,50)

# check function for c_0
def compute_c0(c, Delta):
    return (1 - c) * np.exp( - Delta / 7) + (1 - np.exp( - Delta / 7)) * (1 - np.exp(-2 * (1-c) * m))

c_0 = compute_c0(c_bar, Delta)

# plt.figure()
# plt.plot(c_bar,c_0)
# plt.title('c_0')

# check function for delta_0:
def compute_delta0(c):
    return (1 - c ** m) / (1 - c)

delta_0 = compute_delta0(c_0)

# plt.figure()
# plt.plot(c_bar,delta_0)
# plt.title('delta_0')


# compute the values for c_minus
def compute_c_minus(c,Delta):
    this_c_0 = compute_c0(c,Delta)
    this_delta_0 = compute_delta0(c= 1-this_c_0)

    return (np.exp(c * this_delta_0 * Delta) -1 ) / (np.exp(this_delta_0*Delta) - 1)


def compute_c_plus(c,Delta):

    this_c_minus = compute_c_minus(c,Delta)

    this_c_plus = (1 + (-1 + this_c_minus ** (-m)) * np.exp(-Delta * m)) ** (-1/m)

    #(this_c_minus* np.exp(Delta)) / (1 + this_c_minus * (np.exp(Delta) - 1))

    return this_c_plus, this_c_minus


c_plus, c_minus = compute_c_plus(c_bar,Delta)

# plt.figure()
# plt.plot(c_bar, c_minus)
# plt.title('C_minus')
#
# plt.figure()
# plt.plot(c_bar, c_plus)
# plt.title('C_plus')

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
ax = axs[0 ,0]
ax.plot(c_bar,c_0)
ax.set_title('c_0')
ax.set_xlabel('c')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid()

ax = axs[0,1]
ax.plot(c_bar,delta_0)
ax.set_title('delta_0')
ax.set_xlabel('c')
ax.set_xlim([0, 1])
#ax.set_ylim([0, 1])
ax.grid()

ax = axs[1,0]
ax.plot(c_bar,c_minus)
ax.set_title('c_minus')
ax.set_xlabel('c')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid()

ax = axs[1,1]
ax.plot(c_bar,c_plus)
ax.set_title('c_plus')
ax.set_xlabel('c')
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
ax.grid()

plt.tight_layout()
# plt.grid()

plt.show()









