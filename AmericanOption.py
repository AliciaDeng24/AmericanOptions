import numpy as np
import pandas as pd
import scipy


# This is to replicate the algorithm proposed in a paper on Valuing American Options via
# Simulation using Least Square approach

# The code will be broken down into several parts:
# - Simulation part
# - Numerical Estimation using basis functions
# - Determination of Exercise point
# - Backward Computation for fair valuations

# Here assume that the American Option can ONLY be exercised at discrete time point (Not continuously).
# i.e. the time horizon will be partitioned into small subintervals

# According to the paper, we will use Laguerre polynomials as basis functions for illustration

# First, define basis functions
def L0(x):
    ret = np.exp(-x/2)
    ret[ret == 1] = 0
    return ret

def L1(x):
    ret = np.exp(-x/2) * (1-x)
    ret[ret == 1] = 0
    return ret

def L2(x):
    ret = np.exp(-x/2) * (1 - 2*x + (x**2)/2)
    ret[ret == 1] = 0
    return ret

# Linear regression expression
def linear(x1,x2,x3,a1,a2,a3):
    return x1*a1 + x2*a2 + x3*a3

# Regress explanatory x's with y and find coefficients
def fit_reg_ls(x, y):
    '''
    x - An array of length n (this n depends on the payoff)
    y - An array of length n

    return - An array of length 3 (4?) as coefficients of fitted regression
    '''

    l0 = L0(x)
    l1 = L1(x)
    l2 = L2(x)

    x_stack = np.stack((l0, l1, l2), axis=1)

    a1, a2, a3 = np.linalg.lstsq(x_stack, y)[0]

    return linear(l0,l1,l2,a1,a2,a3)

def payoff(S, K):
    '''
    S - Simulated Price path of underlying stock

    Specify the payoff of such Options
    In this case, payoff function for american put will be used.
    '''
    payoff = K-S
    payoff[payoff < 0] = 0
    return payoff

def determine_exercise(S, curr_payoff_matrix, delta, r, K):
    '''
    Determine whether or not to exercise American Put option prior to expiration
    at current time;

    The decision is made based on current exercise value versus the expected value
    from continuation;
    If current exercise value is less than expected value from continuation, then
    the option will be exercised later, but not necessarily at expiration

    return:
    updated payoff matrix
    '''

    padding = curr_payoff_matrix.shape[1]

    St = np.array([v if K-v > 0 else 0 for v in S])
    Pt_related = np.asarray([curr_payoff_matrix[i,:] if v > 0 else np.zeros(padding) for i, v in enumerate(St)])

    # Form linear regression and solve for coefficients via LS
    # Discount factors
    discounts = np.array([np.exp(-(i+1)*r*delta) for i in range(padding)])
    # Discount all future expected value to current time, and sum to form response variable
    y = np.sum(Pt_related * discounts, axis=1)

    y_hat = fit_reg_ls(St, y)

    # Estimate payoff value from continuation
    Pt = payoff(S, K)
    P_continuation = y_hat

    payoff_t = np.array([])
    # Update payoff arrays to decide whether to exercise the option immediately or not
    for i in range(len(Pt)):
        if Pt[i] > P_continuation[i]:
            # Immediately exercise
            payoff_t = np.append(payoff_t, Pt[i])

            # Erase previously added payoff
            curr_payoff_matrix[i,:][curr_payoff_matrix[i,:] > 0] = 0

        else:
            # Do not exercise, keep it for future
            payoff_t = np.append(payoff_t, 0)

    curr_payoff_matrix = np.c_[payoff_t.reshape(-1,1), curr_payoff_matrix]

    return curr_payoff_matrix


# Back propagate through time to arrive at inception
def backward_computation(S_paths, r, K, T, delta):
    '''
    Backward compute the optimal exercise strategy for each simulated price S_paths
    S_paths: A M-by-N numpy matrix where M indicates simulated senarios and N represents
             number of timesteps
    r: Risk-free interest rate
    K: Strike Price of option
    T: Expiration date of option
    delta: Size of timestep

    return:
    Payoff matrix of size M-by-N
    '''
    # Amount of time intervals
    N = T/delta

    # Backward looping
    for n in np.arange(N,0,-1):
        n = int(n)
        print('..... Currently at timestep {} ......'.format(n))
        S = S_paths[:,n]
        if n == N:
            # At expiration, payoff matrix would be the immediate exercise value at T
            Mpayoff = payoff(S, K).reshape(-1,1)
        else:
            Mpayoff = determine_exercise(S, Mpayoff, delta, r, K)

    return Mpayoff


def valuation(payoff, delta, r):
    '''
    Discount all future payoffs back to time 0 and average them out to arrive at
    the valuation for american options
    '''
    M, N = payoff.shape
    discounts = np.exp([-(i+1)*r*delta for i in range(N)])

    value = np.sum(payoff * discounts)/M

    return value
