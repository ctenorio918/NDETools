import numpy as np
import pathlib
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import signalProcessing as dsp
import utils
import lossMinimization as lm

"""
Function: loadCalibData
Author: Nate Tenorio
Date: 3/5/2023
Purpose: This function (which is hopefully reusable) loads data used in a line calibration measurement. It will load
all files in a directory in which the path - C:\\... \data\YYYY_MM_DD\CALIB# exists, as is specified in arguments.
Arguments:
- basePath (optional), a pathlib global path object which specifies the directory you wish to look for data in
- calibDate (optional), a string in the form YYYY_MM_DD which specifies the date the calibration was taken
- calibNum (optional), the # of the calibration taken on the provided date which you want to load
Returns:
- material, a string indicating the material used in the measurement
- frequency, a float corresponding to the frequency of excitation of the calibration
- positionList, a numpy array indicating the positions measurements were taken at, in [mm]
Dimensions: [M by 1], where M is the number of measurements, followed by a tuple of x and y positions
           _____________________________________________________
        ^ | [1] [4]                                             |
        | | [2] [5]      ... End                                |
        | | [3] [6]                                             |
        | |_____________________________________________________|
        y x ----------------------> 
        Data will be sorted in accordance with this diagram:
        So the numpy arrays we feed it look like this:
        Index: 0  1  2  3  4  5  6  7  8
            x: x1 x1 x1 x2 x2 x2 x3 x3 x3
            y: y1 y2 y3 y1 y2 y3 y1 y2 y3 
- timeVector, a time vector corresponding to the time sampled after triggering on the oscilloscope
Dimensions: [N x 1], N is number of samples
- signalArray, an array of each signal loaded with this function
Dimensions: [N x M], N is number of samples, M is number of signals
"""


def loadCalibDataset(basePath=pathlib.Path(__file__).parent,
                     calibDate=date.today().strftime('20%y_%m_%d'),
                     calibNum=1):
    calibString = 'CALIB' + str(calibNum)  # The directory of the calibration we want (from autoMeasure)
    searchPath = basePath / 'data' / calibDate / calibString  #
    fileList = utils.retrieveFilenames(searchPath)
    description = utils.readDFI(fileList)
    material = description[0][0]
    frequencyMHz = description[:][1]
    numCycles_text = description[2][0]
    numCycles = float(numCycles_text[:-6])
    frequency = [float((sub.split('M')[0])) for sub in frequencyMHz]
    xPos_with_unit = description[:][4]
    xPos = [float((sub.split('m')[0])) for sub in xPos_with_unit]
    yPos_with_unit = description[:][6]
    yPos = [float((sub.split('m')[0])) for sub in yPos_with_unit]
    first = True
    for file in fileList:
        data, dims = utils.readCSVData(file)
        if first:
            timeVector = data[:, 0].reshape(-1, 1)
            first = False
            signalArray = np.zeros(len(timeVector)).T
        signal = data[:, 1]
        signalArray = np.vstack([signalArray, signal])
    signalArray = signalArray[1:][:]
    ind = np.lexsort((yPos, xPos))
    positionList = [(xPos[i], yPos[i]) for i in ind]
    signalArray = [signalArray[i, :] for i in ind]
    signalArray = np.transpose(signalArray)
    return material, frequency, numCycles, positionList, timeVector, signalArray


"""
Function: createHeatmap()
Author: Nate Tenorio
Date: 3/7/2023
Purpose: This code gives a series of possible endpoints for the line of best propagation for any Rayleigh wave
propagation setup. Your specimen should have dimensions x and y as defined above. At each x position, the FFT of the
signal collected at each y position is calculated. The code determines which of the y positions for each x has the highest
amplitude, and appends that value to an array. The information gleaned from this array is used to predict a range of
viable endpoints, which will be used in final calculation of the line calibration direction.


"""


def createHeatmap(centerFrequency,
                  positionList,
                  numCycles,
                  timeVector,
                  signalArray,
                  numFreqs = 10,
                  normalize=False,
                  padZeros=True,
                  autoWindowMethod=None,
                  plotFirst=False,
                  plotHeatmap=True):
    ## Normalizing the signal of each array, if selected.
    signalLength, signalNum = np.shape(signalArray)
    xPositions = [loc[0] for loc in positionList]
    numX = np.size(np.unique(xPositions))
    yPositions = [loc[1] for loc in positionList]
    numY = np.size(np.unique(yPositions))
    if normalize:
        signalArray = dsp.normalizeSignal(signalArray)
        maxUnique = 0
        for i in range(signalNum):
            val = len(np.unique(signalArray[:, i]))
            if val > maxUnique:
                maxUnique = val
    ## Automatically windowing signal from selected set of many measurements
    ## Davidon-Fletcher-Powell Method
    if autoWindowMethod == 'DFP':
        pGuess = 10 ** -9
        startIndArray = dsp.autoWindowDFP(pGuess,
                                          lm.toneBurstLoss_t0,
                                          lm.toneBurstGrad_t0,
                                          timeVector,
                                          signalArray,
                                          centerFrequency, numCycles,
                                          epsilon=0.01,
                                          lineSearchIters=10000)
        endIndArray = [ind + numCycles / centerFrequency for ind in startIndArray]
    ## Correlation method
    elif autoWindowMethod == 'correlation':
        startIndArray, endIndArray = dsp.autoWindowSignalCorr(timeVector, signalArray, centerFrequency,
                                                              dsp.toneBurstOnly,
                                                              centerFrequency, numCycles, timeVector[1] - timeVector[0],
                                                              plotFirst=plotFirst)
    ## Brute force searching method
    else:
        startIndArray = dsp.autoWindowBruteForce(timeVector, signalArray, lm.toneBurstLoss_t0,
                                                 centerFrequency, numCycles)
        endIndArray = [ind + numCycles / centerFrequency for ind in startIndArray]
    ## Windowing Signal:
    signalArray = dsp.ndeWindow(signalArray, startIndArray, endIndArray, plotWindow=True)
    ## Taking FFT of Each Signal:
    freqVec, fftMagnitude, fftPhase = dsp.ndeFFT(timeVector, signalArray, zeroPad=padZeros)
    ## Plotting the first data set
    if plotFirst:
        plt.plot(freqVec / 10 ** 6, fftMagnitude[:, 0])
        plt.xlabel('Frequency, MHz')
        plt.ylabel('Normalized Amplitude')
        plt.title('FFT of First Signal')
        plt.axis([0.8 * centerFrequency / 10 ** 6, 1.2 * centerFrequency / 10 ** 6, 0, 1])
        plt.show()
    ## Creating Heatmap from Averaged Value of Frequencies in Selected Range:
    freqRes = freqVec[1]-freqVec[0]
    minFreq = centerFrequency - numFreqs*freqRes
    firstPoint = np.argmin(np.abs(freqVec-minFreq))
    lastPoint = firstPoint + numFreqs * 2
    blank = np.zeros((1, signalNum)).reshape(-1, 1)
    for val in range(signalNum):
        currMagnitudes = fftMagnitude[:, val]
        values = currMagnitudes[firstPoint:lastPoint]
        blank[val] = np.mean(values)
    heats = blank
    heatMap = heats.reshape(numX, numY).T
    if plotHeatmap:
        # Creating the plot:
        xDiff = np.unique(xPositions)[1] - np.unique(xPositions)[0]
        yDiff = np.unique(yPositions)[1] - np.unique(yPositions)[0]
        pixel_plot = plt.figure()
        pixel_plot.add_axes()
        plt.imshow(heatMap, interpolation='nearest', cmap='twilight')
        plt.title('Calibration Heatmap for Selected Data Set')
        plt.xticks(np.arange(np.min(xPositions), np.max(xPositions)+0.00001, xDiff))
        plt.yticks(np.arange(np.min(yPositions), np.max(yPositions)+0.00001, yDiff))
        plt.xlabel('X Position [mm]')
        plt.ylabel('Y Position [mm]')
        plt.colorbar(pixel_plot)
        plt.show(pixel_plot)
    return heatMap

def predictEndpoints(heatMap, positionList, numPredicted=6):
    numY, numX = np.shape(heatMap)
    yPos = np.unique([loc[1] for loc in positionList])
    lastColumn = heatMap[:, -1]
    largestAmp = np.argmax(lastColumn)
    yRes = yPos[1]-yPos[0]
    yToSearch = np.linspace(yPos[largestAmp]-yRes, yPos[largestAmp]+yRes, numPredicted)
    return yToSearch

def predictMaxLine(positionList, heatMap, maxY):
    xPositions = [loc[0] for loc in positionList]
    numX = np.size(np.unique(xPositions))
    yPositions = [loc[1] for loc in positionList]
    numY = np.size(np.unique(yPositions))
    startInd = np.argmax(heatMap[:, 0])
    x0_y0 = (xPositions[0], yPositions[startInd])
    xf_yf = (xPositions[numX], maxY)
    return x0_y0, xf_yf


