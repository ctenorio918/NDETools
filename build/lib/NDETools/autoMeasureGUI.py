import utils
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import numpy as np

########################################################################################################################
# GUI-BASED UTILITIES                                                                                                  #
########################################################################################################################
"""
Mini Function: nameGUI
Author: Nate Tenorio
Date: 2/15/2023
Purpose: This function creates 'name vectors' for uniformity within GUIs by taking the length of the name
and adding on spaces.
Arguments:
-name: String, the text you would like to configure to label your input
-nameSize: Integer, the length of the combined 'name' vector
"""


def nameGUI(name, nameSize=15):
    dashes = nameSize - len(name)
    nameString = sg.Text(name + '-' * dashes, size=(nameSize))
    return nameString


"""
Function: cleanGUI_Dictionaries()
Purpose: This function is designed to clean the extra punctuation noise that often comes with dictionaries produced
via PySimpleGUI. This includes quotation marks within quotation marks for strings, parenthesis that are unneeded, etc.
Arguments:
- dictGUI: the dictionary created by querying your PySimpleGUI
- exceptions: Any characters that would normally be seen as extraneous that you would like to keep (i.e. colons,
parenthesis, quotation marks)
Returns:
- cleanDictionary - a dictionary with all numeric values saved as floats and all strings without extraneous characters
"""


def cleanGUI_Dictionaries(dictGUI, exceptions=None):
    trash = "'(),"
    if exceptions is not None:  # If there are exceptions:
        toDelete = [bad for bad in trash if bad not in exceptions]  # Ignore them while deleting things
    else:
        toDelete = trash
    for key in dictGUI.keys():  # Iterating through each key in the dictionary
        data = dictGUI[key]  # Looking only at data for the current key
        badKeys = list()
        if data is None or data == "":  # If the value in the field is None or an empty string...
            badKeys.append(key)
        else:  # Otherwise...
            for junk in toDelete:  # For each character in the 'toDelete' string...
                data = data.replace(junk, "")  # ...iterate through data, deleting any instance of the character.
            numIdentifier = [char for char in data if
                             char.isdigit() or char == '.']  # For each character in the data string, see if it is numeric or a decimal
            if len(numIdentifier) == len(data):  # If all characters are numeric or a decimal...
                data = float(data)  # ...convert the value to a float.
            dictGUI[key] = data  # Update the dictionary value with the cleaned-up value
    try:
        for badKey in badKeys:
            del dictGUI[badKey]
    except:
        print('All keys-value pairs are kept.')
    return dictGUI


########################################################################################################################
# GUI AUTO-MEASUREMENT                                                                                                 #
########################################################################################################################





def autoMeasure(guiDict):
    """
    **Function:** autoMeasure()

    **Author:** Nate Tenorio

    **Date:** 2/15/2023 *Revised 4/8/2023*

    Purpose: This script will provide automatic file I/O from a TTX5000-series oscilloscope, create directories to save the
    data in conveniently, and save the data. It will update with the position you should move to between each measurement,
    and plots the readout from the previous measurement as a sanity check.

    <A stretch goal for this project would be the complete automation of this measurement via the use of stepper motors. This
    would be a great project for an undergrad who is interested in mechatronics.>

    Data is saved in a manner which allows processing code in mediatorProcessing.py or calibration.py

    :param

    """

    sg.theme('DarkGrey3')
    dataExists = False  # Setting ourselves up for a break where you try to plot data that has not been sampled
    ## Parameters that represent physics of the system broken out of numInputList for clarity
    freqMHz = guiDict['-freq-']
    xStart_mm = guiDict['-xStart-']
    xStop_mm = guiDict['-xStop-']
    xSamples = int(guiDict['-xSteps-'])
    yStart_mm = guiDict['-yStart-']
    yStop_mm = guiDict['-yStop-']
    ySamples = int(guiDict['-ySteps-'])
    numCycles = int(guiDict['-numCycles-'])

    ## Parameters that affect how we communicate with the scope
    GPIB = guiDict['-GPIB-']
    sampleMethod = guiDict['-mode-']
    numAvg = guiDict['-averages-']

    ## Parameters that affect strictly File I/O
    msmtType = guiDict['-type-']
    material = guiDict['-material-']

    ## Creating GUI for setting scope channels
    layoutScope = [[sg.Text("Set Scope to Desired Position. Double check that signal fits on current window.")],
                   [sg.Text("Record Channel #:"), sg.Push(),
                    sg.Combo((1, 2, 3, 4), size=8, key='rec', default_value=1)],
                   [sg.Text("Trigger Channel #:"), sg.Push(),
                    sg.Combo((1, 2, 3, 4), size=8, key='trig', default_value=4)],
                   [sg.Button('Continue with Selected Channels')]]

    windowScope = sg.Window("Setup Scope Now", layoutScope)

    ## Event loop for channel selecting GUI
    while True:
        eventScope, vals = windowScope.read()
        if eventScope == sg.WIN_CLOSED:
            recChannel = '1'
            trigChannel = '4'
        if eventScope == 'Continue with Selected Channels':
            channels = (list(vals.values()))
            recChannel = channels[0]
            trigChannel = channels[1]
            windowScope.close()
            break

    ## Calculating what positions the receiver must be set in based on current configurations
    xPositions = np.linspace(xStart_mm, xStop_mm, xSamples)
    xPositions = np.round(xPositions, decimals=2)
    yPositions = np.linspace(yStart_mm, yStop_mm, ySamples)
    yPositions = np.round(yPositions, decimals=2)
    ## Exception handling for improper measurement setup
    ## If xPositions is not greater than 0, and you have a difference in x-distances, return error
    if not np.size(xPositions) > 0:
        if not xStart_mm == xStop_mm:
            windowError = sg.Window("Error", [[sg.Text('Error, cannot navigate from xStart to xStop with 0 steps.')],
                                              [sg.Button('Oops, my bad!')]])
            while True:
                errorEvent, errorVals = windowError.read()
                if errorEvent == sg.WIN_CLOSED or errorEvent == 'Oops, my bad!':
                    windowError.close()
                    raise SystemExit
        elif np.size(yPositions) > 0:
            ## If xPositions is 0 and yPositions is not empty, create an x Vector of xStart_mm that is the length of yPos
            xPositions = np.linspace(xStart_mm, xStop_mm, np.size(yPositions))
        else:
            print("You set the system to not take any measurements.")
            raise SystemExit
    ## Identical to above code, but for y
    elif not np.size(yPositions) > 0:
        if not yStart_mm == yStop_mm:
            windowError = sg.Window("Error", [[sg.Text('Error, cannot navigate from xStart to xStop with 0 steps.')],
                                              [sg.Button('Oops, my bad!')]])
            while True:
                errorEvent, errorVals = windowError.read()
                if errorEvent == sg.WIN_CLOSED or errorEvent == 'Oops, my bad!':
                    windowError.close()
                    raise SystemExit
        elif msmtType == 'calibration':
            yPositions = [yStart_mm]
        else:
            yPositions = np.linspace(yStart_mm, yStop_mm, np.size(xPositions))
    elif xSamples != ySamples and msmtType == 'measurement':
        totalSamples = max(xSamples, ySamples)
        xPositions = np.linspace(xStart_mm, xStop_mm, totalSamples)
        yPositions = np.linspace(yStart_mm, yStop_mm, totalSamples)

    ## Creating a list of data points we will visit, and in what order.
    if msmtType == 'calibration':
        '''
        For the calibration scheme, the measurements will look kind of like this:
          _____________________________________________________
        ^ | [1] [4] [7]                                         |
        | | [2] [5] [8] ... End                                 |
        | | [3] [6] [9]                                         |
        | |_____________________________________________________|
        y x ----------------------> 
        
        As we will perform a raster scan across the range we have selected for calibration. This process
        will probably change somewhat as we discover good optimizations/approximations to make, but as-is, this
        is the best we've got.
        
        So the numpy arrays we feed it look like this:
        Index: 0  1  2  3  4  5  6  7  8
            x: x1 x1 x1 x2 x2 x2 x3 x3 x3
            y: y1 y2 y3 y1 y2 y3 y1 y2 y3 
        So the number of measurements in the calibration = xSteps*ySteps
        '''
        xFinal = np.array([])
        yFinal = np.array([])
        for xVal in xPositions:
            for _ in yPositions:
                xFinal = np.append(xFinal, xVal)
            yFinal = np.append(yFinal, yPositions)
        remaining = (np.size(xPositions) * np.size(yPositions))
    elif msmtType == 'measurement':
        """
        Taking a measurement implies you have already performed a calibration and know the axis of highest amplitude
           _____________________________________________________
        ^ | [1]                                                 |
        | |     [2]       ... End                               |
        | |         [3]                                         |
        | |_____________________________________________________|
        y x ----------------------> 
        
        Such that our vectors are simply:
        [x1, x2, x3, ..., xn]
        [y1, y2, y3, ..., yn]
        """
        xFinal = xPositions
        yFinal = yPositions
        remaining = np.size(xFinal)
    else:
        print("Invalid Measurement Type")
        raise SystemExit
    ## Connecting to the oscilloscope. For additional information, check documentation in utils.py
    scope = utils.connectToScope(GPIB)  # Creating scope object
    recordInfo = utils.msmtSetup(scope, timeoutLength=None)  # Configuring measurement
    path = utils.pathMaker(msmtType=msmtType)  # Creating file path to save all data to

    ## Creating GUI for measurement taking:

    layoutFinal = [[sg.Text('X-axis location should be:'), sg.Text("", key='-x-'), sg.Text("mm"), sg.Push()],
                   [sg.Text('Y-Axis location should be:'), sg.Text("", key='-y-'), sg.Text("mm"), sg.Push()],
                   [sg.Text('Current measurement progress:'), sg.Push(),
                    sg.ProgressBar(numAvg, orientation='h', key='prog', size=(20, 20))],
                   [sg.Text("", key='complete?')],
                   [sg.Button('Record Data'), sg.Button('Next Position'),
                    sg.Button('Show Plot of Previous Measurement'), sg.Text("", key='remaining?')]]

    ## Creating event loop for final GUI:

    windowFinal = sg.Window('AutoMeasurement User Interface', layoutFinal, finalize=True)

    currMeasurement = 0
    windowFinal['-x-'].update(xFinal[currMeasurement])
    windowFinal['-y-'].update(yFinal[currMeasurement])
    windowFinal['remaining?'].update(remaining)

    ## Final event loop:

    while True:
        eventFinal, valsFinal = windowFinal.read()
        if eventFinal == sg.WIN_CLOSED or (remaining - currMeasurement) == 0:
            print("Complete!")
            break
        if eventFinal == 'Record Data':
            scaledTime, scaledWave = utils.takeMSMT(scope, recordInfo, channelNum=recChannel, triggerNum=trigChannel,
                                                    sampleMode=sampleMethod, numSamples=numAvg, plotData=False)
            windowFinal['prog'].update(numAvg)
            fileName = utils.configFilenameDispersion(material, xFinal[currMeasurement],
                                                      yFinal[currMeasurement], freqMHz, numCycles, '.csv')
            utils.saveMSMTasCSV(fileName, path, np.vstack((scaledTime, scaledWave)).T, ('Time (s)', 'Signal (V)'))
            windowFinal['complete?'].update('Current Measurement Saved!')
            dataExists = True
        if eventFinal == 'Next Position':
            currMeasurement = currMeasurement + 1
            if remaining - currMeasurement != 0:
                windowFinal['-x-'].update(xFinal[currMeasurement])
                windowFinal['-y-'].update(yFinal[currMeasurement])
                windowFinal['prog'].update(0)
                windowFinal['complete?'].update("")
                windowFinal['remaining?'].update(str(remaining - currMeasurement))
        if eventFinal == 'Show Plot of Previous Measurement':
            if not dataExists:
                print("You have not saved any data yet.")
            else:
                try:
                    plt.plot(scaledTime, scaledWave)
                    plt.title('Recorded Signal from Oscilloscope')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Voltage (V)')
                    plt.show(block=False)
                except:
                    print("You have not saved any data yet.")


## END FUNCTION

########################################################################################################################
# GUI CONFIGURATION                                                                                                    #
########################################################################################################################

"""
Function: configAcquisition()
Author: Nate Tenorio
Date: 2/15/2023
Purpose: This scope opens a GUI that allows the user to easily configure measurement settings. This code is generally 
repurposable for any measurement method that records data that you would like to organize spatially and by frequency.

NOTES ON LINEAR SINE SWEEP - MSMT taken on 4/5/2023 has the following specs:
1-8 MHz (listed), not 1:1 with reality
300 mVpp
1 cycle burst
1ms trigger
Set to 100 KHz -> Period of 10 microseconds
AFG waveform has 10kS at 1GS/s

GUI Arguments:
 - Frequency, MHz
 - xStart, mm - the place along the scan path you plan to start. The x-direction is typically lengthwise.
 - xStop, mm - the stopping place along the x-axis.
 - xSteps - the number of measurements (steps) you want to take along the propagation path in the x-direction
 - yStart/Stop are the same
 - ySteps - typically equal to either xSteps or 0 unless you are calibrating and need to scan across Y to find the ray path
 - Material - what material are you working on?
 - GPIB: The GPIB address of the scope you are saving data from. Typically of the form GPIB::##::INSTR
 - Sample Mode: 'sample' or 'average'
 - Num Averages: How many averages do you want the scope to take, if averaging? (DEBUG Note: It is good to save *something*)
 
 Actions:
 - Load settings from file: see defaultSettings.CSV for formatting. 
 - Save current settings as .CSV file <- surprisingly hard to implement
    This function always saves the inputs in the format that the system wants to read, which is handy.
    A good guess of params FOR A MEASUREMENT is:
    freq = 5 MHz
    xStart = 11 mm, xStop = 21 mm, xStep = 20
    yStart = 6 mm, yStop = 6 mm, yStep = 0
    Sample Mode: 'average', numAverages = 256, GPIB = GPIB::04::INSTR
    
    For a CALIBRATION - process is still being developed. For now, we are implementing a raster scan acquisition.
    freq = 5 MHz 
    xStart = 11 mm, xStop = 21 mm, xStep = 10
    yStart = 5 mm, yStop = 7 mm, yStep = 3  <- Num MSMTs to calibrate = xStep * yStep, this would be a coarse calibration
    Sample Mode: 'average', numAverages = 64 (depends on record length), GPIB = GPIB::04::INSTR
- Begin Automated Measurement
    Runs autoMeasure() and closes configuration GUI
Returns: (None)
- Can return .csv of saved settings, as well as an organized directory for those settings.  
"""


def configAcquisition():
    ## Setting GUI Theme
    sg.theme('DarkBlue4')

    """
    To sum up the little commands I had to use...
    sg.Text() creates text in the GUI
    sg.Push() justifies all commands on either side of it to the edge
    utils.GUIname() is a cheeky function I wrote to make sure things line up nicely
    sg.InputText() allows the user to input a string. The key= optional argument helps you index this data later.
    sg.Combo() works like above, but as a drop down list
    sg.FileBrowse() opens the file browser
    sg.Button() creates clickable entities. Clicking on these entities creates an event that is equal to the string
    assigned to the button
    """

    layoutL = [[sg.Text('Configure Measurement:', font='14'), sg.Push()],

               [nameGUI('Min Frequency:'), sg.InputText(s=10, key='-freq-'), sg.Text('MHz'),
                sg.Push(), sg.Text('Measurement Type:'), sg.Combo(('measurement', 'calibration'), s=20, key='-type-')],

               [nameGUI('Max Frequency:'), sg.InputText(s=10, key='-freqMax-'),
                sg.Push(), sg.Text('Excitation Scheme'),
                sg.Combo(('sin', 'linear sin sweep', 'ramp', 'gaussian'), s=20, key='-excitation-')],

               [nameGUI('x Start:'), sg.InputText(s=10, key='-xStart-'), sg.Text('mm'),
                sg.Push(), sg.Text('Scope Sample Mode:'), sg.Combo(('average', 'sample'), s=20, key='-mode-')],

               [nameGUI('x Stop:'), sg.InputText(s=10, key='-xStop-'), sg.Text('mm'),
                sg.Push(), sg.Text('Number of Averages:'), sg.InputText(s=22, key='-averages-')],

               [nameGUI('x Steps:'), sg.InputText(s=10, key='-xSteps-'),
                sg.Push(), sg.Text('Material:'), sg.InputText(s=22, key='-material-')],

               [nameGUI('y Start:'), sg.InputText(s=10, key='-yStart-'), sg.Text('mm'),
                sg.Push(), sg.Text('GPIB Port:'), sg.InputText(s=22, key='-GPIB-')],

               [nameGUI('y Stop:'), sg.InputText(s=10, key='-yStop-'), sg.Text('mm')],

               [nameGUI('y Steps:'), sg.InputText(s=10, key='-ySteps-')],

               [nameGUI('Burst Cycles:'), sg.InputText(s=10, key='-numCycles-')],

               [sg.Text('Load from File:'), sg.Input(key='-enteredLocation-'), sg.FileBrowse(key='-fileLoc-'),
                sg.Button('Submit')],

               [sg.Button('Begin Measurement with Current Settings'), sg.Button('Save Current Settings')]]

    ## Creating the Window Object
    window = sg.Window('Mediator Wedge Measurement', layoutL)

    while True:  # Event Loop
        event, vals = window.read()  # Read the inputs of the window at the start of the event
        if event == sg.WIN_CLOSED:  # If you press the X button in the top right, close the GUI
            window.close()
            break
        if event == 'Submit':  # If you want to submit a CSV file of the default format, load those values into the MSMT fields
            if vals['-fileLoc-']:
                configs = utils.read_JSON_data(vals['-fileLoc-'])
            try:
                for setting in configs:
                    if configs[setting] != '-fileLoc':
                        window[setting].update(configs[setting])
            except:
                print("Currently selected file does not match typical form.\n"
                      "A JSON file with dictionary fields matching GUI keys is expected.")
        if event == 'Save Current Settings':  # If you want to save a .csv file of the current inputs for faster loading
            cleanVals = cleanGUI_Dictionaries(vals)
            fileName = 'autoMeasurementConfigs'
            toSavePath = utils.pathMaker(msmtType='config')
            utils.saveDictAsJSON(cleanVals, toSavePath, fileName)
        if event == 'Begin Measurement with Current Settings':
            guiDict = cleanGUI_Dictionaries(vals)
            window.close()
            autoMeasure(guiDict)
            break


if __name__ == '__main__':
    configAcquisition()
