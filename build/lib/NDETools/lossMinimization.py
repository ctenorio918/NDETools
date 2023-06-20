import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from scipy.optimize import line_search, minimize_scalar
from scipy import signal as sig

########################################################################################################################
# Loss Functions                                                                                                       #
########################################################################################################################
"""
Function: toneBurstLoss()
Author: Nate Tenorio
Date: 3/17/2023
Purpose: Calculates squared error between the provided signal and a tone burst that arrives at fitted parameter t0
Arguments:
- signal(N x 1), the current signal you are performing a line search for
- t0, the current arrival time you are investigating
- frequency, the frequency of the tone burst of note
- cycles, the number of cycles in the tone burst
- timeRes, the time resolution of the sample
Returns:
- L, the value of the total loss
"""

def toneBurstLoss_t0(t0, timeVec, signal, frequency, cycles):
    y = np.sin(2 * np.pi * frequency * timeVec) * (
            np.heaviside(timeVec - t0, 0.5) - np.heaviside(timeVec - t0 - cycles / frequency, 0.5))
    L = np.sum((y - signal) ** 2)
    return L


"""
Function: toneBurstGrad()
Author: Nate Tenorio
Date: 3/17/2023
Purpose: Calculates the gradient of the LOSS FUNCTION of squared error between the given signal and an ideal tone burst
Arguments: See Above
Returns: The gradient of the loss function taken *with respect to the time offset*
"""


def toneBurstGrad_t0(t0, timeVec, signal, frequency, cycles):
    t0_loc = np.argmin(np.abs(timeVec - t0))
    tf_loc = np.argmin(np.abs(timeVec - t0 - cycles / frequency))
    L_prime_func = (2 * np.sin(2 * np.pi * frequency * timeVec)
                    * (np.heaviside(timeVec - t0, 0.5) - np.heaviside(timeVec - t0 - cycles / frequency, 0.5)) - signal) \
                   * np.sin(2 * np.pi * frequency * timeVec) \
                   * (sig.unit_impulse(len(timeVec), tf_loc)).reshape(-1, 1) - sig.unit_impulse(len(timeVec),
                                                                                                t0_loc).reshape(-1, 1)
    L_prime = np.sum(L_prime_func)
    return L_prime

def translate_excitation(t0: float,
                         shape: np.ndarray,
                         timeVec: np.ndarray,) -> np.ndarray:
    """
    Function: translate_excitation

    Purpose: This takes an arbitrary waveform shape, the time vector of your signal, and a time offset to
    reconstruct the excitation signal for loss minimization. This is designed to aid in the automatic
    windowing of signals with unusual excitation shapes.

    :param t0: float, the time offset of your waveform, given in seconds
    :param shape: np.ndarray, points representing signal locations for your waveform vector
    :param timeVec: np.ndarray, the time vector of your measurement
    :return: shifted_waveform: np.ndarray, output waveform padded with zeroes in accordance with given t0
    """
    dummyVec = np.zeros(np.shape(timeVec))  # Creating a dummy time vector to inject our shape into
    waveformLen = len(shape)  # Getting the length of our waveform for later use
    startIndex = np.argmin(timeVec-t0)
    try:
        dummyVec[startIndex:(waveformLen+startIndex-1)] = shape
    except IndexError:
        print('Given t0 creates out of range solution - creating highly lossy function.')
        dummyVec[startIndex:] = 999
    exShift = dummyVec
    return exShift

def arbitrary_loss_function(*modifiers,
                            shiftFunc: Callable,
                            timeVec: np.ndarray,
                            shape: np.ndarray,
                            signal: np.ndarray) -> np.ndarray:
    """
    Function: arbitrary_loss_function()

    Purpose: This function is designed to be a naive loss function creation method. It is designed such that
    the arguments in *modifiers are what any minimization/gradient descent algorithm is looking to minimize.

    The callable given in shiftFunc should take modifiers, the shape, and your time vector

    :param modifiers: The modifiers of the loss function that you are looking to solve for. Examples - t0, phase, wavespeed
    :param shiftFunc: The function you are using to shift around the waveform given via shape
    :param timeVec: The time vector of the recorded signal
    :param shape: The shape of your excitation
    :param signal: The signal you are looking to automatically window
    :return: loss: The value of MSE of your function
    """
    exShift = shiftFunc(*modifiers, shape, timeVec)
    loss = np.sum((exShift-signal)**2)
    return loss


########################################################################################################################
# Optimization Methods - Useful for any ML Applications                                                                #
########################################################################################################################
"""
Function: bruteForceMinimizer()
Author: Nate Tenorio
Date: 3/17/2023
Purpose: This code implements a brute force method of searching for the location of 
the minimum value of a scalar input loss function. It is very computationally expensive, but
is guaranteed to converge. If needed, further work will be done with scipy optimization.

This is an exercise in breaking perfectionism :) (I hate this code)
Arguments:
-scalarValues -> An N x 1 array of values you want to check for loss minimization
-func() -> The loss function you want to minimize
-*searchArgs - other arguments you need to pass to your loss function
Returns:
minimizer - the value that minimizes the loss function
minimum - the minimum value of the loss function - useful for 'scoring' how monochromatic a msmt is
"""


def bruteForceMinimizer(variable, func, *searchArgs):
    minimum = func(variable[0])
    minimizer = variable[0]
    for val in variable[1:]:
        L = func(val, *searchArgs)
        if L < minimum:
            minimum = L
            minimizer = val
    return minimizer, minimum


"""
Function: DFP()
Author: Nate Tenorio
Date: 3/17/2023
Purpose: This code implements the Davidon-Fletcher-Powell method for numerical stochastic gradient descent. You must
provide a loss function, and optionally a gradient (otherwise a numerical gradient will be calculated). This method will
effectively match parameters of your choosing that fit your loss function.

Importantly, this requires that you actually *know* your loss function. This is not trivial! 

https://en.wikipedia.org/wiki/Davidon%E2%80%93Fletcher%E2%80%93Powell_formula - It is faster than directly solving
for the Hessian (O(n^3)) to solve for B instead.

Arguments:
- pGuess - your guess for the parameter(s) in question. It just gives the algorithm a place to start - make sure it is
reasonable and small. Must be a vector or single numpy64whatevertypethat'scalled
- func - the loss function you are trying to minimize. First argument is p, then *searchArgs
- grad_func - the gradient of your function with respect to components of p
- *searchArgs - the arguments of your func and grad_func that will be used in our line search
- epsilon (default 0.1) - "machine epsilon", or the acceptable gradient magnitude to ensure convergence
- lineSearchIters (default 10000) - the maximum number of times your line search will iterate before saying 'fuck it'
- plotFunc (default None) - the function you plan to use to plot the results of the line search
- plotFirst (default False) - Boolean of whether or not you wanna plot this shit idk bro
"""


def DFP(pGuess, func, grad_func, *searchArgs,
        epsilon=0.1, lineSearchIters=10000, plotFunc=None, plotFirst=False):
    B = np.identity(np.size(pGuess))  # We don't know anything about B, and I find it does not have convergence issues
    p = pGuess  # Setting p to pGuess while keeping pGuess in memory
    lineSearchArgs = searchArgs  # Again with search arguments (in the case that your arguments are functions)
    continueCon = True  # Setting up our while loop (clumsily)
    if np.size(B) == 1:  # Much less sophisticated search criterion for p that is only a single value
        print("D-F-P is designed for minimizing with multiple inputs.")
        print("scipy.optimize.minimize_scalar struggles to converge with signal windowing applications.")
        print("Try calling utils.bruteForceMinimizer !")
        minimizer, minimum = bruteForceMinimizer(lineSearchArgs[0], func, *searchArgs)
        # res = minimize_scalar(func,
        #                       bracket=(np.min(lineSearchArgs[0]),
        #                                np.floor(np.max(lineSearchArgs[0])/100),
        #                                np.max(lineSearchArgs[0])),
        #                       args=lineSearchArgs, method='brent')
        # pFit = res.x
        return minimizer
    else:
        print("Detected Multivariate P - Using Davidon-Fletcher-Powell")
        while continueCon:
            delF = grad_func(p, *searchArgs)  # Calculating local gradient
            d = -B @ delF  # Defining the direction vector
            lineResult = line_search(func, grad_func, p, d, args=lineSearchArgs,
                                     maxiter=lineSearchIters)  # Performing a line search for P in the direction d
            alpha = lineResult[0]  # Our step size is the first parameter saved from scipy.optimize.line_search
            if alpha is None:  # We break our loop if the line search does not converge (because 99% of the time it is ok)
                break
            pNew = p + alpha * d  # Updating a new p vector
            pNew = np.expand_dims(pNew, axis=1)  # Numpy has 0 iq and has to be told that vectors are matrices :skull:
            deltaP = pNew - p  # Finding the change in P
            delG = np.expand_dims((grad_func(pNew, *searchArgs) - delF), 1)  # Finding the step change in gradient
            B = (B + (deltaP * deltaP.T) / (np.dot(deltaP.T, delG))  # Updating B, our surrogate for the inverse Hessian
                 - ((B @ delG) @ (B @ delG).T) / (delG.T @ B @ delG))
            p = pNew  # Setting P to be equal to the new P!
            p = np.squeeze(p)  # Squeezing p for fun
            gradientMagnitude = np.sqrt(np.sum(delF ** 2))  # We consider our system to have converged if
            if gradientMagnitude < epsilon:  # the magnitude of our gradient is small, thus our
                continueCon = False  # step size is small, and our next line search will prob crap out
        pFit = p  # Renaming the return pFit for clarity
        return pFit
