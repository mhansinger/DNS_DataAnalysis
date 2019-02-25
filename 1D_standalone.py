import numpy as np
import matplotlib.pyplot as plt

m = 4

try:
    c_verlauf = np.loadtxt('C_verlauf.txt')
except:
    c_path=input('Give the path to C_verlauf.txt')
    c_verlauf = np.loadtxt(c_path)

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

style = ['b','b--','r','r--']


def analytical_omega(alpha,beta,c):
    exponent = - beta * (1 - c) / (1 - alpha * (1 - c))
    Eigenval = beta**2 / 2 + beta*(3*alpha - 1.344)

    print('Lambda:', Eigenval)

    return Eigenval*(1-c)*np.exp(exponent)

def model_omega(c_plus,c_minus,Delta):
    '''
    :param c_plus:
    :param c_minus:
    :param Delta:
    :return: omega Eq. 29
    '''

    return (c_plus**(m+1) - c_minus**(m+1))/Delta


plt.close('all')

Deltas = [1,2,3,4,5,6,7,8,9,10,15,20]

omega_verlauf = analytical_omega(alpha = 0.818, beta = 6, c = c_verlauf)

fig, ax1 = plt.subplots(ncols=1, figsize=(6, 6))
ax2 = ax1.twinx()

ax1.plot(c_verlauf,'b')
ax2.plot(omega_verlauf,'r')
ax2.set_ylabel('Reaction Rate', color='r')
ax1.set_ylabel('Verlauf C', color='b')
plt.title('C und omega 1D')
plt.show(block=False)

# loop over the different Filters
for Delta in Deltas:

    omega_analytic_list = []
    omega_model_list = []

    for i in range(0,len(c_verlauf)-Delta):

        this_c_bar = c_verlauf[i:i+Delta].mean()
        this_analytical_omega_bar = omega_verlauf[i:i+Delta].mean()

            # compute the boundaries:
        this_c_minus = compute_c_minus(this_c_bar,Delta)
        this_c_plus = compute_c_plus(this_c_minus,Delta)

        this_model_omega_bar = model_omega(this_c_plus,this_c_minus, Delta )

        print(' ')
        print('c_bar: %.2f  c_minus: %.2f  c_plus: %.2f  analytical_omega: %.2f  model_omega: %.2f' %
              (this_c_bar, this_c_minus, this_c_plus, this_analytical_omega_bar, this_model_omega_bar))

        omega_analytic_list.append(this_analytical_omega_bar)
        omega_model_list.append(this_model_omega_bar)

    plt.figure()
    plt.title('Vergleich gefilterter omega für Delta = %i' % Delta)
    plt.plot(omega_analytic_list)
    plt.plot(omega_model_list)
    plt.legend(['omega_bar_analytical','omega_bar_model'])
    plt.savefig('Vergleich_Delta%i.png' % Delta)

    plt.show(block=False)












# i=0
# for Delta in [0.1,1,5,10]:
#
#     c_minus = compute_c_minus(c_bar, Delta)
#     c_plus = compute_c_plus(c_minus, Delta)
#
#     plt.plot(c_bar, c_minus, style[i])
#     plt.plot(c_bar, c_plus, style[i])
#     i=i+1
#
# plt.xlabel('c')
# plt.title('Vergleich mit Fig. 4')
# plt.show()











