import numpy as np
import pathlib
import pyvisa as visa
import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import date
import PySimpleGUI as sg
import json


########################################################################################################################
# TDS 5034B Instrument Communication                                                                                   #
########################################################################################################################

def connectToScope(scopeGPIB):
    """
    Author: Nate Tenorio

    Date: 2/10/2023

    Function Name: connectToScope

    Purpose: This function uses the USB/GPIB IEEE 488 Cable to communicate and connect with an oscilloscope

    Arguments:
    - scopeGPIB: The GPIB address of our oscilloscope. Typically of the form INST::02::GPIB for Tektronix products
    Returns:
    - scope: A pyvisa object that can write, read, and query the oscilloscope. Works on TTX5000-series scopes.
    """
    print("Input GPIB: ")
    print(scopeGPIB)
    rm = visa.ResourceManager()
    currInstruments = rm.list_resources()
    currInst = 1
    for instGPIB in currInstruments:
        print("GPIB Computer Sees: ")
        print(instGPIB)
        if instGPIB == scopeGPIB:
            scope = rm.open_resource(instGPIB)
            scope.timeout = None
        elif currInst == len(currInstruments) and instGPIB != scopeGPIB:
            print("Input GPIB Number Does Not Match Any Connected Devices")
            print(currInstruments)
            raise SystemExit()
        currInst = currInst + 1
    print("Instrument Connection Test")
    print(scope.query('*IDN?'))
    return scope


def msmtSetup(scope, timeoutLength=None):
    """
    Author: Nate Tenorio

    Date: 2/10/2023

    Function Name: msmtSetup

    Purpose: Sets up TTX5000-series oscilloscope for automated measurements

    Arguments:
    - scope: a pyvisa object taken from rm.open_resource(#GPIB of scope) related to the scope of interest

    Returns:
    - recordInfo, tuple: 1x6 tuple containing:
        -record: the record length for reconstruction of the time vector
        -tscale: the timescale of the scope, s
        -tstart: the start time of the scope, s
        -vscale: the volts/level of the scope, V
        -voff: the offset from the scope, V
        -vpos: the reference level of the scope, V
    """
    scope.timeout = timeoutLength  # ms

    scope.write('header 0')  # No header
    scope.write('data:encdg ASCI1')  # Setting the way we would like to read our data as ASCII
    scope.write('data:start 1')  # first sample
    record = int(
        scope.query('horizontal:recordlength?'))  # Saving the record length for reconstruction of the time vector
    scope.write('data:stop {}'.format(record))  # last sample
    scope.write('wfmoutpre:byt_n 2')  # 2 bytes per sample
    print("Byte Number:")
    print(scope.query('wfmoutpre:byt_n?'))
    print(scope.query('wfmoutpre:bit_n?'))

    tscale = float(scope.query('wfmoutpre:xincr?'))  # Time Scale
    print("Time Scale")
    print(tscale)
    tstart = float(scope.query('wfmoutpre:xzero?'))  # Start time
    print("Start Time")
    print(tstart)
    vscale = float(scope.query('wfmoutpre:ymult?'))  # volts / level
    print("Voltage Scale")
    print(vscale)
    voff = float(scope.query('wfmoutpre:yzero?'))  # reference voltage
    print("Reference Voltage")
    print(voff)
    vpos = float(scope.query('wfmoutpre:yoff?'))  # reference position (level)
    print("Y-Offset")
    print(vpos)
    recordInfo = (record, tscale, tstart, vscale, voff, vpos)
    print(scope.query('data:encdg?'))
    return recordInfo


def takeMSMT(scope, recordInfo, channelNum=1, triggerNum=4, sampleMode='average', numSamples=256, plotData=False):
    """
    Author: Nate Tenorio

    Date: 2/10/2023

    Function Name: takeMSMT

    Arguments:
    - scope: a pyvisa object taken from rm.open_resource(#GPIB of scope) related to the scope of interest
    - recordInfo: information taken from the scope's current configuration pulled from msmtSetup()
      recordInfo = (record, tscale, tstart, vscale, voff, vpos)
    - channelNum: an integer from 1-4 indicating the channel we are saving data from
    - triggerNum: an integer from 1-4 noting which channel we are triggering off of
    - sampleMode: 'sample' or 'average': references how we want to record our data
    - numSamples: the number of samples you wish to average
    - plotData: boolean for whether you want a simplified plot of your data - good for a sanity check

    One important thing to note - if you are worried about the number of bits of data you are encoding from your device,
    try importing debugZone.countTotalBits or check how many unique numbers are represented within your data set.
    """
    ## Error message generation for improper inputs
    if channelNum > 4 or channelNum < 1:
        print("Sample channel must be integer from 1-4.")
        raise SystemExit()
    if triggerNum > 4 or triggerNum < 1:
        print("Trigger channel must be an integer from 1-4.")
        raise SystemExit()
    if sampleMode not in ('sample', 'average'):
        print("Your sampling mode must be set to either 'sample' or 'average' to record data.")
    ## Performing scope operations
    scope.write('data:source CH' + str(channelNum))  # Setting data source
    scope.write('trigger:a:edge:source CH' + str(triggerNum))  # Setting edge trigger
    scope.write('acquire:mode ' + sampleMode)  # Setting the sample mode
    r = scope.query('*opc?')  # sync
    if sampleMode == 'average':
        scope.write('acquire:numavg ' + str(numSamples))  # Writing # of sample averages
    t1 = time.perf_counter()  # Collecting time data for the sake of recording measurement time
    scope.write('wfmoutpre:bit_n 16')
    scope.write('acquire:state 0')
    scope.write('acquire:stopafter SEQUENCE')
    scope.write('acquire:state 1')
    r = scope.query('*opc?')
    workingState = scope.query('busy?')
    while workingState == 1:
        workingState = scope.query('busy?')
    t2 = time.perf_counter()
    print('Acquisition time {} s'.format(t2 - t1))
    binWave = scope.query_ascii_values('curve?', container=np.array)
    unscaledWave = np.array(binWave, dtype='double')
    scaledWave = (unscaledWave - recordInfo[5]) * recordInfo[3] + recordInfo[4]
    totalTime = recordInfo[0] * recordInfo[1]
    tStop = recordInfo[2] + totalTime
    scaledTime = np.linspace(recordInfo[2], tStop, num=recordInfo[0], endpoint=False)

    if plotData:
        plt.plot(scaledTime, scaledWave)
        plt.title('Channel ' + str(channelNum))  # plot label
        plt.xlabel('time (seconds)')  # x label
        plt.ylabel('Voltage (Volts)')  # y label
        plt.show()
    return scaledTime, scaledWave


########################################################################################################################
# SECTION: FILE I/O - LOADING/READING FILES, CREATING FILENAMES, ORGANIZING MEASUREMENT DATA                           #
########################################################################################################################

def saveMSMTasCSV(fileName, path, data, headers):
    """
    This is a function intended to save data taken from a measurement into a CSV file

    Function: saveMSMTasCSV

    Author: Nate Tenorio

    Date: 2/14/2023

    Arguments:
    -fileName: string - the name of the file you would like to create
    -path: pathlib concrete path - the path of the file directory you are going to write in
    -data: numpy array - an NxM array of data points
    -headers: list of strings = an M-long tuple of column headers for your data
    """
    toWritePath = path / fileName
    toWrite = pd.DataFrame(data, columns=headers)
    toWrite.to_csv(toWritePath, index=False)
    return


def configFilenameDispersion(material: str,
                             xPos: float,
                             yPos: float,
                             freqMHz: float,
                             numCycles: int,
                             fileType='csv',
                             printFilename=False) -> str:
    """
    Function: configFilenameDispersion

    Purpose: Configures filenames for automatic measurement of data taken on the mediator wedge setup.

    :param material: str, the material you are inspecting
    :param xPos: float, the location along the x-axis you are measuring, in mm
    :param yPos: float, '' y-axis ''
    :param freqMHz: float, the center frequency you are inspecting
    :param numCycles: int, the number of cycles in your measurement
    :param fileType: str, the type of file you are saving without the dot
    :param printFilename: bool, whether you want to print the filename to the console
    :return:
    """

    if np.round(xPos, decimals=0) == xPos and type(xPos) != type(1.0):
        xPosString = '_x_{}mm'.format(
            np.round(xPos, decimals=0))  # If distance is an integer, ensures there is decimal point
    else:
        xPosString = '_x_{}mm'.format(xPos)
    if np.round(yPos, decimals=0) == yPos and type(xPos) != type(1.0):
        yPosString = '_y_{}mm'.format(np.round(yPos, decimals=0))
    else:
        yPosString = '_y_{}mm'.format(yPos)
    if np.round(freqMHz, decimals=0) == freqMHz and type(freqMHz) != type(1.0):
        freqString = '_{}MHz'.format(freqMHz)
    else:
        freqString = '_{}MHz'.format(freqMHz)
    cycleString = '_' + str(numCycles) + 'cycles'
    fileName = material + freqString + cycleString + xPosString + yPosString + '.' + fileType
    if printFilename:
        print(fileName)
    return fileName


def config_filenames_dsp(
        location_array: np.ndarray,
        measurement_date: str,
        measurement_specs: pathlib.Path,
        dsp_specs: pathlib.Path
) -> list:
    """
    function: config_filename_dsp

    Author: Nate Tenorio

    Date: 4/6/2023

    Purpose: This function handles file name writing for calculated STFTs

    Arguments:
    - location_array: np.ndarray, an N x 2 array of x and y positions for your provided measurements, mm
    - measurement_date: str, the date of the measurement you are using in your analysis
    - measurement_specs: a JSON file referring to the configs of the measurement
    - file_type: str, the style of file you wish to save, including the decimal (default: .csv)
    """
    nSignals, nDimensions = np.shape(location_array)
    stft_filenames = list()
    measurement_properties_dict = read_JSON_data(measurement_specs)  # Reading the config JSON file
    dsp_properties_dict = read_JSON_data(dsp_specs)
    msmt_string = ""
    dsp_string = ""
    for keys in measurement_properties_dict:
        if keys == '-material-':
            msmt_string = '_' + measurement_properties_dict[keys] + msmt_string
        elif keys == '-excitation-':
            msmt_string = msmt_string + '_' + measurement_properties_dict[keys]
    for dsp_keys in dsp_properties_dict:
        if dsp_keys == '-method-':
            dsp_string = dsp_properties_dict[dsp_keys] + dsp_string
    for loc in range(nSignals):
        curr_position = location_array[loc, :]
        curr_name = (dsp_string + '_' + str(curr_position[0]) + 'mm_x_' + str(curr_position[1]) + 'mm_y_'
                     + measurement_date + msmt_string)
        stft_filenames = stft_filenames.append(curr_name)
    return stft_filenames


def saveDictAsJSON(dictIn: dict, path: pathlib.Path, fileName: str):
    """
    function: saveDictAsJSON

    Author: Nate Tenorio

    Date: 4/3/2023

    Purpose: This function is a simple dictionary to JSON mapping that allows for easy saving/interpretation of, in this
    case, your measurement inputs.

    Arguments:
     - dictIn: the dictionary you would like to save
     - path: the location of the file you would like to save
     - fileName: the name of the file you are looking to save
    """
    fileNameJSON = fileName + '.json'
    dumpPath = path / fileNameJSON
    with open(dumpPath, 'w') as writeFile:
        json.dump(dictIn, writeFile)
        print("Settings Saved Successfully")


def pathMaker(inPath=pathlib.Path(__file__).parent, msmtType='measurement') -> pathlib.Path:
    """
    function: pathMaker

    author: Nate Tenorio

    date: 2/14/2023

    purpose: this function hunts for a folder within the directory passed by the inPath argument named 'data' and creates
    a child path with the day's date. It then creates a child path within the date of the format MSMT#, where the # represents
    the number of measurements made on that specific day.

    arguments:
    - inPath: (optional) the path you would like to begin hunting for 'data' from
    - msmtType: (optional) 'calibration' or 'measurement' - helps organize calibration vs. msmt data. It will use custom
    names if you so choose, though. Sometimes 'config' comes up.
    """
    if type(msmtType) != type('hello'):
        print('Invalid Measurement Type. Strings usable as directories are acceptable.')
        raise SystemExit
    dataPath = inPath / 'data'  # Creating path of inPath/data
    if not dataPath.exists():  # Checks if data already exists
        dataPath.mkdir()  # Makes directory if it does not exist
    todayDate = date.today()  # Pull today's date
    todayPath = dataPath / todayDate.strftime('20%y_%m_%d')  # Creatin path of inPath/data/YYYY_MM_DD
    if not todayPath.exists():
        todayPath.mkdir()
    pathMade = False
    checkMSMT = 1
    while not pathMade:
        if msmtType == 'measurement':
            msmtStrCheck = 'MSMT' + str(checkMSMT)
            finalPath = todayPath / msmtStrCheck
            if not finalPath.exists():
                finalPath.mkdir()
                pathMade = True
            checkMSMT = checkMSMT + 1
        if msmtType == 'calibration':
            calibStrCheck = 'CALIB' + str(checkMSMT)
            finalPath = todayPath / calibStrCheck
            if not finalPath.exists():
                finalPath.mkdir()
                pathMade = True
            checkMSMT = checkMSMT + 1
        if msmtType not in ('measurement', 'calibration'):
            strCheck = msmtType + str(checkMSMT)
            finalPath = todayPath / strCheck
            if not finalPath.exists():
                finalPath.mkdir()
                pathMade = True
            checkMSMT = checkMSMT + 1
    return finalPath


def dataPathHunter(basePath=pathlib.Path(__file__).parent,
                   mDate=date.today().strftime('20%y_%m_%d'),
                   mType='measurement',
                   mNumber=1) -> pathlib.Path:
    """
    Function: dataPathHunter

    Author: Nate Tenorio

    Date: 2/20/2023

    Purpose: This function hunts for the file path of data set that you want to process for the provided measurement.
    The function hunts in the basePath for a folder named 'data', indexes by 'mDate', then 'mType', then 'mNumber'.

    Arguments:
    - basePath (Optional): (type: path) the root of the path you want to search
    - mDate (Optional): (type: string) the date of the measurement you would like to observe, "YYYY_MM_DD"
    - mType (Optional): (type: string) the type of measurement you would like to load. Valid for 'measurement', 'calibration', 'configuration'
    - mNumber (Optional): (type: int) the number of the measurement you would like to take

    Returns:
    - mPath: the path of all of the data taken during a given measurement
    """
    dataPath = basePath / 'data'  # Checking for C://users/.../data | / operator on a path object expands it
    if not dataPath.exists():
        print('Data must be located within a folder named "data" at location specified in basePath.')
        raise SystemExit()
    else:
        todayPath = dataPath / mDate  # Checking for C://users/.../data/YYYY_MM_DD
        if not todayPath.exists():
            print('No data detected at given date. Input should be string "YYYY_MM_DD".')
            raise SystemExit()
        else:
            if mType not in ['measurement', 'calibration', 'configuration']:
                dataPath = todayPath / (mType + str(mNumber))
            elif mType == 'measurement':
                dataPath = todayPath / ('MSMT' + str(mNumber))
            elif mType == 'calibration':
                dataPath = todayPath / ('CALIB' + str(mNumber))
            elif mType == 'configuration':
                dataPath = todayPath / ('config' + str(mNumber))
            if not dataPath.exists():  # Checking for C://users/.../data/YYYY_MM_DD/mTypemNumber
                print(
                    'Data of the listed mType and mNumber does not exist. Check utils.py -> arguments for formatting.')
                raise SystemExit()
    return dataPath  # Returns C://users/.../data/YYYY_MM_DD/MSMT#


def retrieveFilenames(dataPath) -> list:
    '''
    function: retrieveFilenames(dataPath)

    Author: Nate Tenorio

    Date: 2/21/2023

    Purpose: Find the names of all files located within the given path, check if they are .csv files, and return the filenames

    Arguments:
    dataPath - a pathlib path object that you want to explore

    Returns:
    paths - a list of the .csv files in the given path, string
    '''
    fileList = list(dataPath.iterdir())
    return fileList


def readDFI(fileList, delineator='_'):
    """
    Function: readDFI

    Author: Nate Tenorio

    Date: 3/3/2023

    Purpose: DFI stands for Delineated Filename Information. This function is intended to take filenames that are delineated
    by a given character, delete the delineated character, then record the information held in the filename in a row of a
    DataFrame.

    Arguments:
    - fileList - (type: path) a list of file directories you wish to extract information from
    - delineator - (type: string) the character used as a delineator in your filename
    - fileType - (type: string) the file extension of the files you are trying to load
    """
    i = 0
    for currFile in fileList:
        fileName = currFile.stem
        values = np.array(fileName.split(delineator))
        if i == 0:
            data = np.zeros(len(values))
        data = np.vstack((data, values))
        i = i + 1
    data = data[1:, :]
    DFI = pd.DataFrame(data)
    return DFI


def input_to_path(copied_path: str) -> pathlib.Path:
    """
    A quick function for 'hardcoded' file I/O
    :param copied_path: str, a path you copied from Windows, for example
    :return: pathObject: a pathlib.Path of the directory you copied
    """
    copied_path.encode('unicode escape')
    pathObject = pathlib.Path(copied_path)
    return pathObject


def readCSVData(filePath):
    """
    Function: readCSVData(filePath)

    Author: Nate Tenorio

    Date: 3/3/2023

    Purpose: Loads data from a file into a numpy array. Pretty straightforward.

    Arguments:
    - filePath: (type: path), the path of the file you are looking to load.

    Returns:
    - data: (type: numpy array), The data from your file. C'mon y'all, we know this.
    """
    readInFrame = pd.read_csv(filePath)
    data = readInFrame.values
    dims = np.shape(data)
    return data, dims


def read_JSON_data(filePath: pathlib.Path):
    """
    Function: read_JSON_data(filePath)

    Author: Nate Tenorio

    Date: 4/3/2023

    Purpose: Simple file reading from JSON

    Arguments: Path - a pathlib.Path object WITH FILE EXTENSION

    Returns: data - the data from your json file
    """
    with open(filePath) as json_file:
        data = json.load(json_file)
        print("Data Loaded Successfully")
    return data


########################################################################################################################
# SECTION: DATA VISUALIZATION - FUNCTIONS USED IN PLOTTING, FIGURE CREATION, ETC.                                      #
########################################################################################################################

def plot_basic(timeVec: np.ndarray,
               signalVec: np.ndarray,
               title='Signal vs. Time'):
    """
    Mini Function: plot_basic()

    Purpose: Plots the most basic possible plot of signal vs. time. Handy to eliminate repetitive code.

    :param timeVec: np.ndarray, the time vector of your measurement
    :param signalVec: np.ndarray, the signal vector of your measurement
    :param title: str, the title of your plot
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(timeVec, signalVec)
    ax.set(xlabel='Time (s)', ylabel='Voltage (V)',
           title=title)
    plt.show()
    return fig


def plot_FFT(freqVec: np.ndarray,
             magVec: np.ndarray,
             title='FFT Results'
             ) -> plt.Artist:
    fig, ax = plt.subplots()
    ax.plot(freqVec, magVec)
    ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude', title=title)
    plt.show()
    return fig


def save_plot(fig: plt.Figure,
              filename: str,
              location: pathlib.Path):
    """
    Mini Function: save_plot()

    This function saves any plot you produce with default settings attuned to your figure. The file
    extension should be specified in the filename parameter.

    :param fig:
    :param filename:
    :param location:
    :return:
    """
    out_loc = location / filename
    fig.savefig(out_loc)
    print(f'Saved image to {out_loc}')

########################################################################################################################
# SECTION: DATA SORTING - DATA REORGANIZATION, ORDERING, REASSIGNMENT, ETC.                                             #
########################################################################################################################


########################################################################################################################
# SECTION: MISC FUNCTIONS - FUNCTIONS THAT DO NOT FIT ANY OTHER LABEL                                                  #
########################################################################################################################

"""
Function: merge_lists()
Author: Nate Tenorio
Purpose: Merges lists into a list of lists with a length equal to the longer list.
Arguments:
- *lists: The lists you want to merge.
- fillValue: The value you will fill empty vectors with if the lists are of uneven length
returns
- mergedList: The merged list
"""


def merge_lists(*args, fill_value=None):
    maxLength = max([len(nth) for nth in args])  # Iterates through given lists and finds the longest length
    mergedList = []  # Preallocating
    for i in range(maxLength):  # Iterates once for each loop of the max length
        mergedList.append([
            args[k][i] if i < len(args[k]) else fill_value for k in range(len(args))
            # Appends a list made of the kth interval of list1 and the ith interval of list2
        ])
    return mergedList


def cut_quadrants(fg, kg, abs_fft_data) -> tuple:
    """
        cuts the absFftArray and remove 2nd, 3rd and 4th quadrant from
        2D-FFT array (only keep 1st quadrant) before plotting it
        - this helps to get better resolution (maxima on spots which make
          physically no sense or contain phase information are ignored)
          and increases computation speed since less data needs to be
          processed

        args:
            - fg -  your frequency mesh
            - kg - your wavenumber mesh
            - abs_fft_data - absolute value of 2DFFT data
    """
    quarter_y = fg.shape[0] // 2
    quarter_x = fg.shape[1] // 2

    fg = fg[quarter_y:,
         quarter_x:int(1.5 * quarter_x)]
    kg = kg[quarter_y:,
         quarter_x:int(1.5 * quarter_x)]

    abs_fft_data = abs_fft_data[quarter_y:,
                   quarter_x:int(1.5 * quarter_x)]
    return fg, kg, abs_fft_data

def predict_encoding_bits(unique_values: np.ndarray) -> int:
    num_unique = len(unique_values)
    log_out = np.log2(num_unique)
    bitMin = np.ceil(log_out)
    return bitMin

################################## Outdated Functions ##################################################################
def sortDataByType(vals: dict, offset=-2):
    """
    WARNING: THIS FUNCTION IS NO LONGER USED IN GUI OPERATION. SEE cleanGUI_dictionaries().

    function: sortDataByType
    Author: Nate Tenorio
    Date: 2/15/2023
    Purpose: this sorts data taken from a PySimpleGUI input into float values and strings, deletes characters that would
    be unsuitable for reading from a .csv file, then outputs the separated values. Reusable enough to warrant a description.
    Arguments:
    - Vals: (Dictionary), The raw value data taken from an sg.window.read() operation
    Returns:
    - toSaveNums: A list of floats to save, in order of appearance
    - toSaveString: "" strings ""
    """
    data = list(vals.values())  # Converting our dictionary to a list
    data = data[:offset]  # Throwing away the data we don't want to save
    toSave = [None] * (len(data))  # Preallocating an array
    i = 0  # Set iteration constant
    for userInputs in data:  # For each field in the dictionary...
        isNumeric = True  # Assuming at first that the value is numeric
        numericVal = ""  # Creating an empty string for purely numeric values
        for char in userInputs:  # For each character in the string pulled from the dictionary...
            if char.isdigit() or char == '.':  # If the value is a number or decimal point...
                numericVal = numericVal + char  # Add it to a numeric string
            elif char.isalpha():  # If any character in the string is a letter...
                isNumeric = False  # The value cannot be converted to a float
        if isNumeric:  # If the number can be converted into a float...
            toSave[i] = float(numericVal)  # Convert it into a float and add it to our list
        else:
            deleteStr = "(,)'"  # A string of random characters the GUI package likes to throw in
            for c in deleteStr:  # For each of those characters
                userInputs = userInputs.replace(c, "")  # Delete them
            toSave[i] = userInputs  # Add it to our list
        i = i + 1
    toSaveNums = [ele for ele in toSave if
                  isinstance(ele, int) or isinstance(ele, float)]  # Pull all floats or integers
    toSaveString = [ele for ele in toSave if isinstance(ele, str)]  # Pull all strings
    return toSaveNums, toSaveString
