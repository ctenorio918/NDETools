## Package Imports ##
import datetime
import numpy as np  # Linear Algebra Operator
import pathlib

import tftb.processing.base
from scipy.fft import fft, fft2, fftfreq, fftshift, next_fast_len  # Discrete Fourier Transforms
from scipy import signal as sig  # Other DSP from Scipy
from scipy.signal import windows as win  # Window functions from Scipy
from scipy.optimize import minimize_scalar
import pywt  # PyWavelet
import tftb
from tftb.processing import WignerVilleDistribution
import matplotlib.pyplot as plt  # Standard Plotting Tools
from sklearn.linear_model import LinearRegression  # Linear Regression from sk-learn
import PySimpleGUI as sg  # GUI manipulation for interpretation of wavelets
## Importing My Functions ##
import utils
import lossMinimization as lm


def toneBurst(t0: float,
              timeVec: np.ndarray,
              signal: np.ndarray,
              frequency: float,
              cycles: int,
              plot=False) -> np.ndarray:
    '''
    **Function: toneBurst**

    **Author: Nate Tenorio**

    **Date: 3/22/2023**

    Purpose: Creates a tone burst with offset t0 from given time vector. It takes signal as a required argument because
    I am lazy and want to call this in the DFP code. Not my best work.
    '''
    y = np.sin(2 * np.pi * frequency * timeVec) * (
            np.heaviside(timeVec - t0, 0.5) - np.heaviside(timeVec - t0 - cycles / frequency, 0.5))
    return y


def toneBurstOnly(frequency, cycles, timeRes):
    """
    Function: toneBurstOnly()

    Author: Nate Tenorio

    Date: 3/13/2023

    Purpose: "Sketches" a sinusoidal tone burst with frequency, cycles, and sampling resolution of specifications.
    Same as above, but without heaviside function.

    **Arguments:**

    - frequency - float, the frequency of the tone burst, Hz

    - cycles - the number of cycles in the tone burst

    - timeRes - the time resolution of your signal, seconds
    """
    period = 1 / frequency
    timeVec = np.linspace(start=0, stop=cycles * period, num=int(np.round(cycles * period / timeRes)))
    burst = np.sin(2 * np.pi * frequency * timeVec)
    return burst


########################################################################################################################
# Data Normalization, Zero Padding, and other Preprocessing Methods                                                    #
########################################################################################################################

def normalizeSignal(signalArray: np.ndarray,
                    rowByRow=True) -> np.ndarray:
    """
    **Function: normalizeSignal()**

    **Author: Nate Tenorio**

    **Date: 3/9/2023**

    **Purpose:** This function normalizes each signal in an input signal array such that it has a maximum amplitude of 1, and
    a mean centered at zero. If rowByRow is false, the entire array is normalized to the maximum and offset of the entire
    averaged dataset.

    **Arguments:**

    - signalArray: an N x M array, with M measurements of length N that you wish to normalize
    - rowByRow: whether to normalize each measurement independently or in aggregate

    **Returns:**

    - normalizedSignalArray: an N x M array of signal with zeroed means and maximum amplitudes of 1.
    """
    if rowByRow:  # Method for normalizing each signal independently
        (dataPoints, numSignals) = np.shape(signalArray)
        for M in range(numSignals):
            rawSignal = signalArray[:, M]
            offset = np.mean(rawSignal)  # Offset is the mean of the dataset
            adjSignal = rawSignal - offset  # Subtracting out offset
            amplitude = np.amax(adjSignal)  # Amplitude is the maximum of the dataset
            normSignal = adjSignal / amplitude  # Dividing out the amplitude
            if M == 0:
                normalizedSignalArray = normSignal.reshape(-1, 1)
            else:
                normalizedSignalArray = np.hstack((normalizedSignalArray, normSignal.reshape(-1, 1)))
    else:  # Method for normalizing by entire dataset
        offset = np.mean(signalArray)
        amplitude = np.max(signalArray)
        normalizedSignalArray = signalArray / amplitude - offset
    return normalizedSignalArray


def zeroPadding(timeVec: np.ndarray,
                signal: np.ndarray,
                ) -> [np.ndarray, np.ndarray]:
    """
    **Function:** zeroPadding(timeVec, signalArray)

    **Author:** Nate Tenorio

    **Date:** 3/3/2023 (Most recent edit: 3/7/2023)

    **Purpose:** This function pads the input data with zeros such that its length is equal to the next fast length.

    **Inputs:**
        - timeVec - an Nx1 array of the time recorded in your measurement
        - signal - an Nx1 array of signals, where N is the signal length

    **Returns:**
        - paddedTime - the padded time vector to use in generation of FFT frequencies
        - paddedSignal - the zero-padded signal
    """
    optimalLength = next_fast_len(np.size(timeVec))
    padLength = optimalLength - np.size(timeVec)
    paddedTime = np.pad(timeVec, (0, padLength))
    paddedSignal = np.pad(signal, (0, padLength))
    return paddedTime, paddedSignal


def autoWindowSignalCorr(timeVector, signalArray, frequency, fitForm, *fitArgs, plotFirst=False):
    """
    **Function:** autoWindowSignalCorr()

    **Author:** Nate Tenorio

    **Date:** 3/13/2023

    **Purpose:** This function is designed to automatically window signal that presents as a monochromatic tone burst. This is
    important for consistently processing data sets with a very large amount of signals. It uses a linear regression performed
    on cross-correlated data to predict this endpoint.

    **Arguments:**

    - timeVec: An Nx1 array of the time recorded in your measurement
    - signalArray: An NxM array of signals, where N is the signal length, and M is the number of signals
    - fitForm: A function handle corresponding to expected shape of the waveform. An example of this that exists within the
    signalProcessing folder is 'toneBurstGen'
    - fitArgs: Arguments required for the selected fit form
    (NOTE: This is the code I am far and away the least confident about. A good cross validation is to check FFT amplitude
    statistics throughout the measurement to ensure it is actually capturing what we want.)

    **Returns:**
    - startIndArray: a column vector of start indices of signal to be windowed
    - endIndArray: a column vector of end indices of signal to be windowed
    """
    nPoints, nSignals = np.shape(signalArray)
    # Finding the shape of the waveform that we want to use in cross-correlation
    sigForm = fitForm(*fitArgs)
    # Initializing For Loop:
    for sigNum in range(nSignals):
        #  Using Correlation Instead of Signal for Peak Detection to Minimize Noise Impact
        correlation = sig.correlate(signalArray[:, sigNum], sigForm)

        # Peak Detection (with as much generality as possible built in)
        indices_per_cycle = np.floor((1 / frequency) / (timeVector[1] - timeVector[0]))
        peaks_indices, properties = sig.find_peaks(correlation,
                                                   height=0.001 * max(correlation),
                                                   distance=0.9 * indices_per_cycle)
        prominences = sig.peak_prominences(correlation, peaks_indices)
        prominences = prominences[0]
        peaks_indices, properties = sig.find_peaks(correlation,
                                                   height=0.001 * max(correlation),
                                                   distance=0.9 * indices_per_cycle,
                                                   prominence=abs(np.mean(prominences) - np.std(prominences)))
        dx = np.array([0])

        # Identifying Data Clusters that Survive Peak Detection
        for peakInd in peaks_indices:
            try:
                diff = peakInd - peakIndOld
            except:
                diff = 0
            peakIndOld = peakInd
            dx = np.append(dx, diff)
        chunkInds = np.where(dx > np.mean(dx) + np.std(dx))[0]
        chunkInds = np.append([0], chunkInds)
        i = 0

        # Finding the cluster that has the highest amplitude peak in the cross-correlation
        for chunkStart in chunkInds:
            if i != len(chunkInds) - 1:
                currChunk = peaks_indices[chunkStart:(chunkInds[i + 1] - 1)]
            else:
                currChunk = peaks_indices[chunkStart:]
            if np.round(np.max(correlation[peaks_indices]), 4) in np.round(correlation[currChunk], 4):
                signalChunk = currChunk

        # Using our signal chunk to predict signal window using Linear Regression:
        maxInd = [loc for loc in range(len(signalChunk)) if np.round(correlation[signalChunk[loc]], 4)
                  == np.round(np.max(correlation[signalChunk]),
                              4)]  # Finds index in signalChunk corresponding to global max
        maxInd = maxInd[
            0]  # Python has brain damage and converts our cleverly-obtained index into a list for some reason
        model = LinearRegression(fit_intercept=True)  # Configuring sklearn linear regression
        xFitData = signalChunk[0:maxInd].reshape(-1, 1)  # Our x-axis is indices
        yFitData = correlation[xFitData].reshape(-1, 1)  # Our y-axis is correlation corresponding to those indices
        model.fit(xFitData, yFitData)  # Fitting our model. model.coef_[1] is slope, model.coef_[0] is y-intercept
        xIntercept = np.argmin(np.abs(
            model.coef_[0] * np.linspace(0, nPoints, nPoints - 1) + model.intercept_))  # Function to find x-intercept

        # Calculation of our actual start/end indices
        startInd = np.floor(xIntercept)
        endInd = startInd + len(sigForm)
        try:
            startIndArray = np.append(startIndArray, startInd)
        except:
            startIndArray = startInd
        try:
            endIndArray = np.append(endIndArray, endInd)
        except:
            endIndArray = endInd

        # Plotting the first signal (if enabled)
        if sigNum == 0 and plotFirst:
            xViz = np.linspace(startInd, signalChunk[maxInd], 1000).reshape(-1, 1)  # Creating Dummy Data
            yViz = model.predict(xViz)
            fig, (ax_orig, ax_corr) = plt.subplots(2, 1)
            ax_orig.plot(range(len(timeVector)), signalArray[:, 0], label='Windowed Signal')
            ax_orig.set_xlim(startInd, endInd)
            ax_corr.plot(range(len(correlation)), correlation, label='Correlation')
            ax_corr.plot(peaks_indices,
                         correlation[peaks_indices], 'go', label='Peaks Taken from FindPeaks')
            ax_corr.plot(xViz, yViz, '-r', label='Linear Regression Prediction')
            ax_corr.set_xlim(startInd, endInd)
            ax_corr.set_ylim(1.1 * np.min(correlation), 1.1 * np.max(correlation))
            plt.legend()
            plt.show()

    #  Data Reshaping (Outside for loop)
    startIndArray = startIndArray.reshape(-1, 1)
    endIndArray = endIndArray.reshape(-1, 1)
    return startIndArray, endIndArray


def autoWindowDFP(pGuess: "Parameter Guesses",
                  func: "Loss Function",
                  grad_func: "Gradient of Loss Function",
                  timeVec: np.ndarray,
                  signalArray: np.ndarray,
                  *searchArgs: "Args for line search",
                  epsilon=0.1, lineSearchIters=10000) -> np.ndarray:
    """

    **Function:** AutoWindowDFP

    **Author:** Nate Tenorio

    **Date:** 3/17/2023

    **Purpose:** This code presents a different method of peak detection using the Davidon-Fletcher-Powell formula to minimize
    a provided loss function. This method is less 'naive' in that you are performing a line search for a provided loss
    function and gradient function. The implementation of DFP is a little haphazard as well - but

    Having this second method is useful for several reasons! It trades speed and conventional signal processing for a robust
    method that better emulates how a system would perform after being trained on what the signal should look like. As well,
    having two auto-windowing methods helps us check for agreement - to allow for easy debugging if things are tough.

    **Arguments:**

    - pGuess - your guess for the parameter(s) in question. It just gives the algorithm a place to start - make sure it isreasonable and small. Must be a vector or single numpy64whatevertypethat'scalled
    - func - the loss function you are trying to minimize. First argument is p, then *searchArgs
    - grad_func - the gradient of your function with respect to components of p
    - timeVec - an N x 1 array of the time vector of your measurement
    - signalArray - an N x M array of signals on which you want to perform a D-F-P search
    - *searchArgs - extra arguments of your func and grad_func that will be used in our line search
    Note: This is typically where the time vector and signal vector you are calling go. If your signal
    is an array, I highly recommend considering wrapping your loss function to return an array.
    As it stands, I have *not* done this and call DFP in a loop, which is very inefficient.
    If you want an example of a loss function implemented, see the debugZone.py file

    - epsilon (default 0.1) - "machine epsilon", or the acceptable gradient magnitude to ensure convergence
    - lineSearchIters (default 10000) - the maximum number of times your line search will iterate before saying 'fuck it'

    **Returns:**

    - startIndArray: An array of the starting index for each windowed signal
    - endIndArray: An array of the ending index for each windowed signal
    """
    nPoints, nSignals = np.shape(signalArray)
    for signalNum in range(nSignals):
        t0_current = lm.DFP(pGuess,
                            func,
                            grad_func,
                            timeVec,
                            signalArray[:, signalNum].reshape(-1, 1),
                            *searchArgs, epsilon=epsilon, lineSearchIters=lineSearchIters)
        t0_loc = np.argmin(abs(timeVec) - t0_current)
        try:
            startIndArray = np.append(startIndArray, t0_loc)
        except NameError:
            startIndArray = t0_loc
    return startIndArray


def autoWindowBruteForce(timeVector, signalArray, lossFunc, *searchArgs):
    """
    **Function:** autoWindowBruteForce

    **Author:** Nate Tenorio

    **Date:** 3/24/2023

    **Purpose:** Use a brute force method of minimizing the loss function for your input signal.
    It calculates the loss function at every possible start index. It's terrible... but it works.

    **Arguments:**

    - timeVector - an N x 1 array of the time vector of your measurement
    - signalArray - an N x M array of signals on which you want to perform a D-F-P search
    - lossFunc - the handle loss function you want to minimize (see lossMinimization.py)
    - *searchArgs - additional arguments used in your measurement

    **Returns:**
    - startIndArray - np.ndarray of selected signal start times
    """
    nPoints, nSignals = np.shape(signalArray)
    for signalNum in range(nSignals):
        t0_current, lossMin = lossFunc(timeVector, lm.toneBurstLoss_t0,
                                       timeVector, signalArray[:, signalNum].reshape(-1, 1), *searchArgs)
        t0_loc = np.argmin(abs(timeVector) - t0_current)
        if signalNum == 0:
            startIndArray = t0_loc
        else:
            startIndArray = np.append(startIndArray, t0_loc)
    return startIndArray


########################################################################################################################
# Windowing Functions for DSP                                                                                          #
########################################################################################################################

def ndeWindow(signal, startInd, endInd, *windowArgs, windowMethod=win.hann, plotWindow=False):
    """
    **Function:** ndeWindow()

    **Author:** Nate Tenorio

    **Date:** 3/27/2023

    **Purpose:** This function takes in an array of signal and multiplies it by a Hann Window. This is done to help smooth out
    a discrete Fourier Transform, for example.

    **Arguments:**
        - signal - an 1xN, where N is the signal length
        - startInd - starting location for the window function
        - endInd - ending location for the window function
        - *windowArgs - any additional arguments needed for your window function (i.e. beta for a Kaiser window)
        - windowMethod = scipy.signal.hann() - A function handle for the window function you want to apply
        - printWindow = False - a Boolean of whether you'd like to print your window

    **Returns:**
        - windowedSignal - an NxM array of signals, windowed by
    """
    pointsInWindow = endInd - startInd
    numPoints = len(signal)
    windowScaler = np.zeros(numPoints)
    windowShape = windowMethod(pointsInWindow, *windowArgs)
    windowScaler[startInd:endInd] = windowShape
    signal_out = signal * windowScaler
    print(np.shape(signal_out))
    if plotWindow:
        plt.plot(windowShape)
        plt.title("Window Shape")
        plt.xlabel("Sample # Within Provided Range")
        plt.ylabel("Amplitude")
        plt.show()
    return signal_out


########################################################################################################################
# Discrete Fourier Methods                                                                                             #
########################################################################################################################

def ndeFFT(timeVec: np.ndarray,
           signal: np.ndarray,
           normalize=True,
           unwrapPhase=False,
           zeroPad=False) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    **Function:** ndeFFT

    **Author:** Nate Tenorio

    **Date:** 3/3/2023

    **Purpose:** Receive interpretable one-sided frequency information from the input array of signals, particularly including
    magnitude and phase.

    **Arguments:**
        - timeVec - an Nx1 array of the time recorded in your measurement
        - signal - an Nx1 signal, where N is the signal length
        - unwrapPhase - boolean, whether you would like to unwrap phase data (meaning you aggregate phase beyond 2pi)
        - zeroPad - boolean, whether you would like to zero pad your data

    **Returns:**
        - freqVec - an Nx1 array of the frequencies used in the evaluation
        - fftMagnitude - an Nx1 array of magnitude data from your FFT
        - fftPhase - an Nx1 array of phase data from your FFT
    """
    deltaT = timeVec[1] - timeVec[0]
    samplingFrequency = 1 / deltaT  # Taking sampling frequency from actual data is more consistent
    if zeroPad:
        timeVec, signalArray = zeroPadding(timeVec,
                                           signal)  # Runs above zero padding function utility on your data
    spectrum = fft(signal)  # Calculating the FFT
    fftMagnitude = abs(spectrum)  # Using absolute value to evaluate the Magnitude
    angleData = np.angle(spectrum)  # Taking phase data without unwrapping
    if unwrapPhase:  # If you want to unwrap phase...:
        rebuildPhase = 0  # Creating a base phase of 0
        unwrapData = np.unwrap(angleData)  # Using numpy's unwrap function to unwrap angle
        # Unwrapping angles involves an arctan operation, which is inherently unstable. to overcome this, we choose
        # to rebuild the unwrapped phase using the absolute difference between each point, which keeps the data
        # much more stable!
        phaseDiffs = abs(np.diff(unwrapData))
        for k in range(np.size(phaseDiffs)):
            rebuildPhase = np.append(rebuildPhase, phaseDiffs[k])
        angleData = rebuildPhase
    fftPhase = angleData
    # Here, we only take the positive side of all of our data
    fftMagnitudeRaw = fftMagnitude[:len(signal)//2]
    if normalize:
        maxAmp = np.amax(fftMagnitudeRaw)
        normAmp = fftMagnitudeRaw / maxAmp
        fftMagnitude = normAmp
    else:
        fftMagnitude = fftMagnitudeRaw
    fftPhase = fftPhase[:len(signal)//2]
    freqVec = fftfreq(np.size(timeVec), 1 / samplingFrequency)[
              :len(signal)//2]  # Using scipy's fftfreq to build a frequency vector
    return freqVec, fftMagnitude, fftPhase


def nde_2d_fft(
        time_vector: np.ndarray,
        position_array: np.ndarray,
        signal_array: np.ndarray,
        to_save_flag=False,
        plot_first=False,
        file_name_function=utils.config_filenames_dsp,
        *file_name_args
):
    """
    **Function:** nde_2D_FFT

    **Author:** Nate Tenorio

    **Date:** 4/6/2023

    **Purpose:** The purpose of this code is to perform a 2D-FFT of the the signal provided in signal_array over the
    positions reported in position_array.

    **Arguments:**
     - time_vector: np.ndarray, an N x 1 array representing the time taken in your measurement, N is # of samples
     - position_array: np.ndarray, an N x 2 array representing the X and Y coordinates taken in your measurement, in mm
     - signal_array: np.ndarray, an N x M array of M signals with length N you wish to perform an STFT on
     - to_save_flag: bool, whether you'd like to save the results of the STFT calculations
     - plot_first: bool, whether you'd like to plot the results of the first STFT
     - file_name_function: function, a function that handles the filename of your function
     - *file_name_args: additional arguments to feed the file_name_function

    **Returns:**
     - frequency_mesh: A 2-D array representing the frequency mesh
     - wavenumber_mesh: A 2-D array representing the wavenumber mesh
     - fft_2D_magnitude: An array that, when plotted alongside the mesh, represents the magnitude of the 2D FFT
    """
    nf = 1 / (2 * (time_vector[1] - time_vector[0]))  # Calculating Nyquist Temporal Sampling Frequency
    distance_between_points = (np.sqrt(position_array[1, 0] ** 2 + position_array[1, 1] ** 2) -
                               np.sqrt(position_array[0, 0] ** 2 + position_array[0, 1] ** 2))
    nk = 1 / distance_between_points  # Calculating Nyquist Spatial Sampling Frequency
    numTimes, numPositions = np.shape(signal_array)  # Using shape of signal array to discern sample numbers
    frequency_vector = np.linspace(-nf, nf, num=numTimes)  # Building the frequency vector
    wavenumber_vector = np.linspace(-nk, nk, num=numPositions)  # Building the wavenumber vector
    fft_2d_magnitude = fftshift(fft2(signal_array))  # Calculating the 2D-FFT
    frequency_mesh, wavenumber_mesh = np.meshgrid(frequency_vector, wavenumber_vector)  # Producing meshgrid for plots

    if to_save_flag:
        file_name = file_name_function(*file_name_args)
        save_path = utils.pathMaker(msmtType='2DFFT')
        final_path = save_path / (file_name + ".npz")
        np.savez(final_path, frequency_mesh, wavenumber_mesh, fft_2d_magnitude)

    if plot_first:
        fig = plt.figure(1, dpi=600)  # Plot it!
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.contourf(frequency_mesh,
                    wavenumber_mesh,
                    fft_2d_magnitude)

    return frequency_mesh, wavenumber_mesh, fft_2d_magnitude



###############################################################################################################
# Time/Frequency Representation Package Utility Functions                                                     #
###############################################################################################################

def norm2freq(time_out: np.ndarray,
              norm_freq_out: np.ndarray):
    """
    The way that the time-frequency representation package is written causes it to return normalized frequency
    instead of true representative frequency. Sometimes this can be desirable (machine learning algorithms do
    not care one way or the other!), but for user comprehension, this conversion is often desirable.
    :param time_out: The time vector returned from your TFR
    :param norm_freq_out: The normalized frequency vector returned from your TFR
    :return: tfr_freq: The corrected frequency vector
    """
    fs = 1/(time_out[1]-time_out[0])
    nyquist_fs = fs/2
    tfr_freq_out = norm_freq_out * nyquist_fs
    return tfr_freq_out

def tfr_get_values(tfr,
                   normalize_frequency=False):
    """
    Using the relatively unusual schema to obtain usable information from PyTFTB:


    :param tfr:
    :param normalize_frequency:
    :return:
    """
    tfr_mag, tfr_time, tfr_norm = tfr.run()
    if normalize_frequency:
        tfr_freq = tfr_norm
    else:
        tfr_freq = norm2freq(tfr_time, tfr_norm)
    return tfr_mag, tfr_time, tfr_freq

def plot_TFR(tfr_frequency: np.ndarray,
             tfr_time_or_wavenumber: np.ndarray,
             tfr_mag: np.ndarray,
             tfr: tftb.processing.base.BaseTFRepresentation,
             xBounds: np.ndarray,
             yBounds: np.ndarray,
             t_or_k='t') -> plt.Artist:
    fig = plt.figure()
    plt.pcolormesh(tfr_time_or_wavenumber, tfr_frequency, tfr_mag, shading='gouraud')
    if type(tfr) == type('Michael Wainwirght'):
        tfr_name = tfr
    else:
        tfr_name = tfr.name

    if t_or_k == 'k':
        plt.title(f'{tfr_name} of Frequency vs. Time')
        plt.xlabel('Wavenumber (1/m)')
        plt.ylabel('Frequency, f (Hz)')
        plt.xlim(xBounds[0], xBounds[1])
        plt.ylim(yBounds[0], yBounds[1])
    else:
        plt.title(f'{tfr_name} of Frequency vs. Time')
        plt.xlabel('Time, t (s)')
        plt.ylabel('Frequency, f (Hz)')
        plt.xlim(xBounds[0], xBounds[1])
        plt.ylim(yBounds[0], yBounds[1])
    plt.show()
    return fig

def save_TFR(tfr_frequency: np.ndarray,
             tfr_time: np.ndarray,
             tfr_mag: np.ndarray,
             tfr: tftb.processing.WignerVilleDistribution,
             fileName='Time_Frequency_Representation') -> pathlib.Path:
    """
    Function: save_tfr()

    Purpose: Saves tfr as a numpy .npz file.

    :param tfr_frequency: The frequency from your tfr
    :param tfr_time: The time from your tfr
    :param tfr_mag: The magnitude from your tfr
    :param fileName: The name you would like to give your file
    :return: final_path: The path of the file you've saved
    """
    save_path = utils.pathMaker(msmtType=tfr.name)
    final_path = save_path / (fileName + '.npz')
    np.savez(final_path,
             tfr_frequency,
             tfr_time,
             tfr_mag)
    return final_path

#################################################################################################################
# Short Time Fourier Transform
#################################################################################################################

def save_STFT_properties(segment_length, window_function):
    save_path = utils.pathMaker(msmtType='DSP_Configs')
    data = dict([('-segment_length-', segment_length),
                 ('-window_function-', window_function),
                 ('-method-', 'STFT')])
    currTime = datetime.time
    filename = 'STFT_Properties_' + str(currTime.hour) + '_' + str(currTime.minute)
    utils.saveDictAsJSON(data, save_path, filename)
    full_path = save_path / filename
    return full_path

def save_STFT(stft_frequency,
              stft_time,
              stft_mag,
              fileName) -> pathlib.Path:
    """
    Function: save_STFT()

    Purpose: Saves STFT as a numpy .npz file.

    :param stft_frequency: The frequency from your STFT
    :param stft_time: The time from your STFT
    :param stft_mag: The magnitude from your STFT
    :param fileName: The name you would like to give your file
    :return: final_path: The path of the file you've saved
    """
    save_path = utils.pathMaker(msmtType='STFT')
    final_path = save_path / (fileName + '.npz')
    np.savez(final_path,
             stft_frequency,
             stft_time,
             stft_mag)
    return final_path

def nde_STFT(time_vector: np.ndarray,
             signal: np.ndarray,
             segment_length: int,
             window_function=win.hann,
             remove_trends=False) -> (list, list, list):
    """
    **Function:** nde_STFT

    **Author:** Nate Tenorio

    **Date:** 4/6/2023

    **Purpose:** This method calculates and returns the STFT for each signal denoted in signal_array. The windowing
    method selected via window_function is used in calculation of the STFT. The segment_length parameter refers
    to the width of the window - a wide window gives good frequency resolution but poor time resolution. There
    is always a trade-off, and this is a hyperparameter that must be tuned.

    :param time_vector: np.ndarray, an N x 1 array representing the time taken in your measurement, N is # of samples
    :param signal: np.ndarray, an N x M array of M signals with length N you wish to perform an STFT on
    :param segment_length: int, an integer representing the window width you'd like to use for the STFT
    :param window_function: function, the window function you'd like to use. Can be *custom.*
    :param remove_trends: bool, whether you'd like to use scipy.signal.detrend to remove linear trends

    :return: stft_frequency - np.ndarray of frequencies,
     stft_time - np.ndarray representing time,
     stft_mage - np.ndarray representing STFT magnitude
    """
    timeRes = (time_vector[-1] - time_vector[0]) / len(time_vector)  # Calculating time resolution
    samplingFrequency = 1 / timeRes  # Calculating sampling frequency
    stft_frequency, stft_time, stft_mag = sig.stft(x=signal,  # Calculate the STFT of that signal
                                                   fs=samplingFrequency,  # At the sampling frequency
                                                   window=window_function(segment_length),  # With the given win
                                                   nperseg=segment_length,  # Over the given segment length
                                                   detrend=remove_trends)  # And detrend, if warranted
    return stft_frequency, stft_time, stft_mag

def STFT_plot_labeled(stft_frequency: np.ndarray,
                      stft_time_or_wavenumber: np.ndarray,
                      stft_mag: np.ndarray,
                      xBounds: np.ndarray,
                      yBounds: np.ndarray,
                      t_or_k='t') -> plt.Artist:
    fig = plt.figure()
    plt.pcolormesh(stft_time_or_wavenumber, stft_frequency, stft_mag, shading='gouraud')
    if t_or_k == 'k':
        plt.title('STFT of Frequency vs. Time')
        plt.xlabel('Wavenumber (1/m)')
        plt.ylabel('Frequency, f (Hz)')
        plt.xlim(xBounds[0], xBounds[1])
        plt.ylim(yBounds[0], yBounds[1])
    else:
        plt.title('STFT of Frequency vs. Time')
        plt.xlabel('Time, t (s)')
        plt.ylabel('Frequency, f (Hz)')
        plt.xlim(xBounds[0], xBounds[1])
        plt.ylim(yBounds[0], yBounds[1])
    plt.show()
    return fig

def STFT_plot_unlabeled(stft_frequency: np.ndarray,
                        stft_time_or_wavenumber: np.ndarray,
                        stft_mag: np.ndarray) -> plt.Artist:
    fig = plt.figure(1, dpi=600)  # Plot it!
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    xv, yv = np.meshgrid(stft_frequency, stft_time_or_wavenumber)
    ax.contourf(xv,
                yv,
                stft_mag,
                cmap='Spectral',
                zorder=-40)
    plt.show()
    return fig

def STFT_time2wavenumber(stft_frequency: np.ndarray,
                         stft_time: np.ndarray,
                         propagation_distance: int
                         ) -> list[np.ndarray]:
    """
    **Function: STFT_time2wavenumber()**

    **Author:** Nate Tenorio

    **Date:** 4/12/2023

    **Purpose:** This function uses the following equation to calculate the wavenumber vector that
    corresponds to each input STFT. This is useful to move from a plot of time/frequency to wavenumber/frequency.

    (k = \omega*t/(dx))

    Generally, this allows for parity plots to be made between the output of any given 2DFFT and STFT

    :param stft_frequency_tensor: A list of M 1 x N np.ndarrays representing frequency outputs of STFTs (hz)
    :param stft_time_tensor: A list of M 1 x N np.ndarrays representing time outputs of STFTs (s)
    :param propagation_distance_array: A 1xM np.ndarray of propagation distances for each measurement, (m)
    :return: stft_wavenumber_tensor: A list of M 1 x N np.ndarrays representing the wavenumber (1/m) of the measurement
    """
    deltaX = propagation_distance
    omega = 2 * np.pi * stft_frequency
    t = stft_time
    k = np.dot(t, omega) / deltaX
    stft_wavenumber = k
    return stft_wavenumber


################################################################################################################
# Wagner-Ville Distribution Family                                                                             #
################################################################################################################

def nde_WignerVilleDistribution(time: np.ndarray,
                                signal: np.ndarray,
                                fbins = None,
                                fwindow=None):
    """
    Creating the object representing the WVD of the input signal.
    There are windows (twindow, fwindow) which can be configured manually if so desired. the default is

    :param time: The timestamps you are using in your WVD calculation
    :param signal: The signal you wish to analyze
    :param fbins: The number of frequency bins you wish to use in calculation (please leave this as default)
    :param fwindow: A window function you want to pass to the WVD. Defaults to Hamming window (not sure of length).

    :returns: wvd - the Wagner-Ville Distribution object
    """
    wvd = WignerVilleDistribution(signal, timestamps=time, n_fbins=fbins,fwindow=fwindow)
    return wvd

def nde_pseudoWignerVille(time: np.ndarray,
                          signal: np.ndarray,
                          window=None):
    """
    Calculating the pseudo Wigner-Ville transform of your signal.

    :param time: The timestamps you are using in your WVD calculation
    :param signal: The signal you wish to analyze
    :param window: A window function you want to pass to the WVD. Defaults to Hamming window (not sure of length).
    :return:
    """
    pWFD = tftb.processing.PseudoWignerVilleDistribution(signal=signal, timestamps=time, fwindow=window)
    return pWFD
def nde_SPWVD(time: np.ndarray,
              signal: np.ndarray,
              n_voices=None):
    """
    Computing the smoothed pseudo Wagner-Ville distribution.
    TODO: Investigate other input parameters for this function in specific...
    :param time:
    :param signal:
    :param n_voices:
    :return: SPWVD_mag, SPWVD_time, SPWVD_freq
    """
    SPWVD_mag, SPWVD_time, SPWVD_freq = tftb.processing.affine.smoothed_pseudo_wigner(signal,
                                                                                      timestamps=time,
                                                                                      n_voices=n_voices)
    return SPWVD_mag, SPWVD_time, SPWVD_freq


#################################################################################################################
# Reassigned Methods                                                                                            #
#################################################################################################################

def nde_ReassignedSpectrogram(time: np.ndarray,
                              signal: np.ndarray,
                              window=None):
    """
    Calculates the Reassigned Spectrogram of your signal. See Niethamer et. al.

    :param time: Your time vector, 1xN
    :param signal: signal vector, 1xN
    :param window: a scipy window (taken as an array-like)
    :return: spectrogram, reassigned_spectrogram, reassignment_matrix
    """
    _, re_spec, _ = tftb.processing.reassigned_spectrogram(signal, time_samples=time, window=window)
    spec = tftb.processing.Spectrogram(signal, timestamps=time, window=window)
    return spec, re_spec

################################################################################################################
# Unusual/Uncommon Time-Frequency Representations                                                              #
################################################################################################################

# TODO: Trace https://tftb.readthedocs.io/en/latest/apiref/tftb.processing.html for sources.

def nde_BertrandDistribution(time: np.ndarray,
                             signal: np.ndarray,
                             fmin=None,
                             fmax=None,
                             n_voices=None,
                             window=None):
    """
    ISBN 10709908/96$05.0

    The Bertrand function 'marginalizes to frequency when integrated over t, ... and localizes on hyperbolic
    instantaneous frequencies and group delays.' The original paper is in French and does not appear to have
    an English translation. Still, it is an interesting transform.

    https://tftb.readthedocs.io/en/latest/auto_examples/plot_4_2_2_bertrand_hyperbolic_gd.html#sphx-glr-auto-examples-plot-4-2-2-bertrand-hyperbolic-gd-py

    :param time:
    :param signal:
    :param fmin:
    :param fmax:
    :param n_voices:
    :param window:
    :return:
    """
    bertrandDistribution = tftb.processing.BertrandDistribution(signal,
                                                               timestamps=time,
                                                               fmin=fmin,
                                                               fmax=fmax,
                                                               n_voices=n_voices,
                                                               fwindow=None)
    return bertrandDistribution


def nde_MargenauHill(time: np.ndarray,
                     signal: np.ndarray,
                     window=None):
    """
    The Margenau-Hill algorithm shows up a few times (usually as the pseudo M-H transform) in time-frequency
    representation of signals. I have one paper (Yun et. al) which uses this method.
    :param time: Time vector
    :param signal: Signal vector
    :param window: Windowing function
    :return: mh_tfr: The TFR objected of the Margenau-Hill Distribution
    """
    mh_tfr = tftb.processing.cohen.MargenauHillDistribution(signal,
                                                            timestamps=time,
                                                            fwindow=window)
    return mh_tfr

def nde_pseudoMargenauHill(time: np.ndarray,
                           signal: np.ndarray,
                           window=None):
    """
    Calculates the PMHD (defined clearly in https://www.researchgate.net/publication/238522355_Damage_detection_based-propagation_of_the_longitudinal_guided_wave_in_a_bimetal_composite_pipe)
    This is another form of time-frequency representation that exhibits several distinct features (lagging/leading modes).
    Could be useful. Who knows?
    :param time: Your time vector
    :param signal: Your signal vector
    :param window: A scipy-defined window
    :return:
    """
    pmh_tfr = tftb.processing.cohen.PseudoMargenauHillDistribution(signal,
                                                                   timestamps=time,
                                                                   fwindow=window)
    return pmh_tfr

################################################################################################################
# Continuous Wavelet Transformation Methods                                                                    #
################################################################################################################

# https://dsp.stackexchange.com/questions/76624/continuous-wavelet-transform-vs-discrete-wavelet-transform
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def select_continuous_wavelet(family='mexh',
                              B=1.0,
                              C=1.0,
                              P=1,
                              M=1) -> str:
    """
    **Function:** select_continuous_wavelet(family, order)

    **Author:** Nate Tenorio

    **Date:** 4/13/2023

    **Purpose:** This function is designed to help the user easily select a wavelet shape, and ensures that the user
    has selected a wavelet that exists within the PyWavelet package.

    :param str family: str, the type of wavelet family you wish to expect. List: 'cgau', 'cmor', 'fbsp', 'gaus', 'mexh', 'morl', 'shan'
    :param B: float, the bandwidth B used in Complex Morlet, Shannon, and Frequency B-Spline wavelets
    :param C: float, the center frequency for Complex Morlet, Shannon, and Frequency B-Spline wavelets
    :param P: int, the derivative order used in Gaussian and Complex Gaussian Derivative Wavelets
    :param M: int, the spline order used in Frequency B-Spline Wavelets

    :return: wavelet: str, the wavelet you will use for your CWT
    """

    if family in ['cmor', 'shan']:
        wavelet = family + str(B) + '-' + str(C)
    elif family in ['cgau', 'gaus']:
        wavelet = family + str(P)
    elif family in ['fbsp']:
        wavelet = family + str(M) + '-' + str(B) + '-' + str(C)
    elif family in ['mexh', 'morl']:
        wavelet = family
    else:
        wavelet = None
        print('Selected Wavelet not in "Wavelist" of PyWavelet wavelets.')
    return wavelet


def freq_to_scale_loss(scale: float,
                       wavelet: str,
                       sampling_frequency: float,
                       goal_frequency: float) -> float:
    """
    Short loss function used in numerical calculation of what *scale* corresponds to the frequencies you wish
    to observe.
    :param scale: float, the variable minimized by the optimizer
    :param wavelet: str, the type of wavelet you are using
    :param sampling_frequency: float, the sampling frequency of your measurement
    :param goal_frequency: float, the frequency you are trying to represent via scale
    :return: L: float, the loss value from the input scale and the return
    """
    # TODO: Double check why I multiply by sampling frequency!
    L = abs(pywt.scale2frequency(wavelet, scale) * sampling_frequency - goal_frequency)**2
    return L


def select_wavelet_scale(wavelet: str,
                         sampling_frequency: float,
                         desired_frequencies: np.ndarray) -> np.ndarray:
    """
    **Mini Function:** select_wavelet_scale
    Purpose: Move from measurement specifications to scale interpretable by wavelet transform performance
    :param wavelet: the configured wavelet
    :param sampling_frequency: the sampling frequency of your measurement (1/deltat)
    :param desired_frequencies: a vector of the frequencies you would like to inspect
    :return: scale: the scale array that will be used as an input to your cwt
    """
    scale = np.empty(np.shape(desired_frequencies))
    print(np.shape(scale))
    for i in range(len(desired_frequencies)):
        goal_frequency = desired_frequencies[i]
        result = minimize_scalar(freq_to_scale_loss, bracket=(1, 10), args=(wavelet, sampling_frequency, goal_frequency))
        scale[i] = result.get('x')
        if i == 60:
            print(f'Our resulting value of scale for {goal_frequency} is {result.get("x")}.')
            print(f'This is verified through the PyWavelet package as: {pywt.scale2frequency(wavelet, result.get("x"))}.')
            print(f'The shape of our the result')
    return scale


def save_cwt_properties(wavelet, scale):
    """
    **Function:** save_cwt_properties()

    **Author:** Nate Tenorio

    **Date:** 4/13/2023

    ** Purpose: ** This function documents and saves the properties of your continuous wavelet transform as a JSON file.
    :param wavelet: The wavelet you have configured for the CWT
    :param scale: The scale calculated with select_wavelet_scale
    :return: cwt_properties_path: The path where your properties are saved
    """
    save_path = utils.pathMaker(msmtType='CWT')
    data = dict([('-wavelet-', wavelet),
                 ('-scale-', scale),
                 ('-method-', 'CWT')])
    currTime = datetime.time
    filename = 'CWT_Properties_' + str(currTime.hour) + '_' + str(currTime.minute)
    utils.saveDictAsJSON(data, save_path, filename)
    cwt_properties_path = save_path / filename
    return cwt_properties_path


def cwt_time2wavenumber(
        cwt_freq_vec: np.ndarray,
        cwt_time_vec: np.ndarray,
        propagation_distance: float
) -> np.ndarray:
    """
    Mini Function: cwt_time2wavenumber

    Converts your time vector into a wavenumber vector after the cwt has been performed.

    :param cwt_freq_vec: the frequency vector from your cwt
    :param cwt_time_vec: the time vector from your cwt
    :param propagation_distance: the propagation distance of your measurement
    :return: cwt_wavenumber_vector: the resultant wavenumber vector
    """
    cwt_wavenumber_vector = 2 * np.pi * np.dot(cwt_freq_vec, cwt_time_vec) / propagation_distance
    return cwt_wavenumber_vector


def plot_tfr_cwt(cwt_out: np.ndarray,
                 time_vec: np.ndarray,
                 freq_vec: np.ndarray):
    xv, yv = np.meshgrid(time_vec, freq_vec)
    plt.contourf(xv,
                 yv,
                 cwt_out,
                 cmap='Spectral',
                 zorder=-40)
    plt.title('Time-Frequency Representation of CWT Output')
    plt.xlabel('Time (s)')  # These might be switched
    plt.ylabel('Frequency (MHz)')
    plt.show()


def plot_k_vs_f_cwt(cwt_out: np.ndarray,
                    time_vec: np.ndarray,
                    freq_vec: np.ndarray):
    xv, yv = np.meshgrid(time_vec, freq_vec)
    plt.contourf(xv,
                 yv,
                 cwt_out,
                 cmap='Spectral',
                 zorder=-40)
    plt.title('Time-Frequency Representation of CWT Output')
    plt.xlabel('Wavenumber (1/m)')  # These might be switched
    plt.ylabel('Frequency (MHz)')
    plt.show()


def nde_cwt(signal: np.ndarray,
            wavelet: str,
            scale: np.ndarray,
) -> [np.ndarray]:
    cwt_out, frequency_vec = pywt.cwt(signal, scale, wavelet)
    return cwt_out, frequency_vec

################################################################################################################
# EMD: Empirical Mode Decomposition (AKA Cheating???)                                                          #
################################################################################################################

"""
Gotta think through the process flow on this one.
TODO: Read about sifting

"""