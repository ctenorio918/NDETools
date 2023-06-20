import numpy as np
import autoMeasureGUI
import lineCalibration
import utils

"""
Before you start:

- Decide on how many points you want to scan.

- Double check the code below, and comment out steps you don't want to perform (DO NOT PUSH THESE TO GITHUB)

- Double check the inputs to the code below (typically measurement dates for File I/O)
"""
########################################################################################################################
# Step 1: Press Big Green Play Button in Top Right Corner!                                                             #
########################################################################################################################

"""
Follow the steps in the auto measure GUI documentation. Use (or save) a config file.
"""
autoMeasureGUI.configAcquisition()

########################################################################################################################
# Step 2: Take your measurement as shown by the above GUI                                                              #
########################################################################################################################

########################################################################################################################
# Step 3: Load the data set. Make sure before you start that the date and number are correct.                          #
########################################################################################################################
material, frequency, numCycles, positionList, timeVector, signalArray = \
    lineCalibration.loadCalibDataset(calibDate='2023_03_10', calibNum=1)

########################################################################################################################
# Step 4: Create the heatmap. Double-check the loss minimization method and loss function you are using match.         #
########################################################################################################################

heatmap = lineCalibration.createHeatmap(frequency[0] * 10 ** 6, positionList, numCycles, timeVector, signalArray,
                                        plotFirst=True)

########################################################################################################################
# Step 5: Predict Endpoints. This function calculates some points to create greater endpoint resolution for our msmt.  #
########################################################################################################################

yToSearch = lineCalibration.predictEndpoints(heatmap, positionList, numPredicted=6)

########################################################################################################################
# Step 6: Take Data at Endpoints.                                                                                      #
########################################################################################################################

autoMeasureGUI.configAcquisition()

########################################################################################################################
# Step 7: Compare FFT Data at Endpoints as per Below. Make sure this is set to your 2nd measurement                    #
########################################################################################################################

material, frequency, numCycles, yPositions, timeVector_endpoints, signalArray_endpoints = \
    lineCalibration.loadCalibDataset(calibDate='2023_03_10', calibNum=1)
heatmap_endpoints = lineCalibration.createHeatmap(frequency[0] * 10 ** 6, yPositions, numCycles, timeVector_endpoints,
                                                  signalArray_endpoints, plotFirst=True)
maxY = yPositions[np.argmin(heatmap_endpoints), 1]

x0_y0, xf_yf = lineCalibration.predictMaxLine(yPositions, heatmap, maxY)

print("Starting Position of Final Measurement [mm]")
print(x0_y0)
print("Ending Position of Final Measurement [mm]")
print(xf_yf)
print("Create a Config File by running autoMeasureGUI.configAcquisition() in your terminal or in PyCharm!")