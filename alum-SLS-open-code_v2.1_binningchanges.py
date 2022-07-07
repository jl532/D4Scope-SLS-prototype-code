## final control software for SLS aluminum detector rig for open format imaging
## this set of conditions was used to image the cov-spot assay on 5-3-2022

## establish global variables like exposure conditions to run

alignConfig = {"gain" : 12,
               "expo" : 10e3,
               "digshift" : 2,
               "pixelform" : 'Mono8',
               'binval' : 2}

#this is a carry over from previous detector software, but useful for final implementation
singleConfig = {'gain': 12,  # change this for gain if you need-  0-24
                'expo': 1e6,  # 2e6 is a 2 second exposure 10s
                'digshift': 2,   # dont change
                'pixelform':'Mono12p', #dont change
                'binval': 2}   #dont change

# final conditions to be run in single capture
expoVars = [1]
gainVars = [5]
laserPowVars = [50,150,200]

fileName = "_SLSp_LT"


# import json configuration for array into array_setup, which holds critical array info
import json
stringConfig = ""
with open("array_setup.json", 'r') as jsonConfigs:
    stringConfig = jsonConfigs.read()
array_setup = json.loads(stringConfig)

## import packages
# general purpose packages
import os, sys, time
from time import sleep

# IO packages
import json
import csv
from pypylon import pylon
import easygui
import serial

# math and image processing packages
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

#helper functions added here. might be useful to put in utils.py later.
def cvWindow(name, image, keypressBool, delay):
    """
    cvWindow - debugging tool used to display an image just to see what's in a specific image
    """
    
    print("---Displaying: "
          +  str(name)
          + "  ---")
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    pressedKey = cv2.waitKey(delay)
    cv2.destroyAllWindows()
    if keypressBool:
        return pressedKey
        sys.exit()

def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data

# set up buffer converters for image acq from the camera
vidConverter = pylon.ImageFormatConverter()
vidConverter.OutputPixelFormat = pylon.PixelType_Mono8
vidConverter.OutputBitalignment = pylon.OutputBitAlignment_MsbAligned

# set up single capture buffer to image converter for 12 bit

singleConverter = pylon.ImageFormatConverter()
singleConverter.OutputPixelFormat = pylon.PixelType_Mono16
singleConverter.OutputBitalignment = pylon.OutputBitAlignment_MsbAligned
#connect and establish camera

## inquire user for what they want to do - quick imaging and optimization?
# or auto run through and gather all data


# run align step with aligning assist overlay of crosshairs
#align step
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.BinningVertical.SetValue(alignConfig['binval'])
camera.BinningHorizontal.SetValue(alignConfig['binval'])
camera.Width = 1000 
camera.Height = 800
camera.Gain = alignConfig['gain']
camera.ExposureTime = alignConfig['expo']
camera.DigitalShift = alignConfig['digshift']
camera.PixelFormat = alignConfig['pixelform']
camera.OffsetX = 52 # mult of 4, but also, this is horizontal
camera.OffsetY = 200# default for 350 for centered, 260 for vertical chip
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

#connect to ESP:
arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

print("press q to exit from align script")
# grab images/video from camera, display on cvwindow
# waitkey and end if user wants, end program and reset if want to change settings
# proceed to single capture if prepared

# adjust brightness of one red laser from 0-254
# adjust brightness of second red Laser from -254 to 0

brightnessVal = "-250"
write_read(brightnessVal)

brightnessVal = "250"
write_read(brightnessVal)

camera.Gain = 24
camera.ExposureTime = 10e4
camera.DigitalShift = 4

# mark up the image with circles to verify ROI alignment
fids3 = [0,0], [0,34.5],[0,172.5],[0,207]


fidPatternRows = int(np.amax([x[0] for x in fids3]) - np.amin([x[0] for x in fids3]) + 2 *  round(array_setup["row_pitch"]))
fidPatternCols = int(np.amax([x[1] for x in fids3]) - np.amin([x[1] for x in fids3]) + 2 *  round(array_setup["col_pitch"]))

fidPatternImg = np.zeros((fidPatternRows,fidPatternCols))
fidPrimeRowShift = fids3[0][0] + array_setup["row_pitch"]
fidPrimeColShift = fids3[0][1] + array_setup["col_pitch"]
for eachFiducial in fids3:
    # shift fiducials to center them between the pitch padding
    cv2.circle(fidPatternImg, 
                (round(eachFiducial[1] + fidPrimeColShift), 
                 round(eachFiducial[0] + fidPrimeRowShift)),
                array_setup["radii"], 
                255, 
                thickness = -1)
fidPatternImg = cv2.normalize(fidPatternImg.copy(),
                        np.zeros(shape=(fidPatternRows, fidPatternCols)),
                        0, 255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)

while camera.IsGrabbing():
    buffer = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if buffer.GrabSucceeded():
        frame = vidConverter.Convert(buffer).GetArray()
    
    res = cv2.matchTemplate(frame, fidPatternImg, cv2.TM_CCORR_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
                  
    #marking up the image with aligning helpers
    verImg = cv2.cvtColor((frame.copy()).astype('uint8'), cv2.COLOR_GRAY2BGR)
    # draw where the fiducials are found
    cv2.rectangle(verImg,
                  max_loc,
                  (max_loc[0] + fidPatternCols,
                   max_loc[1] + fidPatternRows),
                  (0, 105, 255),
                  3)
                  
    cv2.putText(verImg,
                "align array within cyan box please",
                (225,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
                cv2.LINE_AA)
    
    cv2.rectangle(verImg,
                  (250,175),
                  (750,625),
                  (255, 255, 0),
                  2)
    cv2.putText(verImg,
                "my code can only compensate for so much",
                (200,700),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
                cv2.LINE_AA)
    cv2.putText(verImg,
                "press q when aligned",
                (300,775),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
                cv2.LINE_AA)
                  
    # show image frame by frame
    cv2.imshow('Frame',verImg)
    # Press Q on keyboard to  exit
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        # turn off camera and arduino light and connection for next time.
        cv2.destroyAllWindows()
        write_read("0")
        break
buffer.Release()
camera.Close()

# =============================================================================

# turn off camera and arduino light and connection for next time.
# run image capture step of just one image, or multiple
brightnessVal = "250"
write_read(brightnessVal)
time.sleep(0.5)
brightnessVal = "-250"
write_read(brightnessVal)

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
#camera.BinningVertical.SetValue(singleConfig['binval'])
#camera.BinningHorizontal.SetValue(singleConfig['binval'])
camera.BinningVertical.SetValue(2)
camera.BinningHorizontal.SetValue(2)
camera.Width = 1000
camera.Height = 800
camera.Gain = singleConfig['gain']
camera.ExposureTime = singleConfig['expo']
camera.DigitalShift = singleConfig['digshift']
camera.PixelFormat = singleConfig['pixelform']
#camera.OffsetX = 52 # mult of 4, but also, this is horizontal
#camera.OffsetY = 200# default for 350 for centered, 260 for vertical chip

buffer = camera.GrabOne(int(singleConfig['expo']*1.1))
if not buffer:
    raise RuntimeError("Camera failed to capture single image")
image = singleConverter.Convert(buffer).GetArray()
image = (image * 16)

brightnessVal = "0"
write_read(brightnessVal)

cv2.imshow('captured image',image)
cv2.waitKey(-1)
cv2.destroyAllWindows()
 
buffer.Release()
arduino.close()
camera.Close()
 
 
## save image for analysis later
 
output = easygui.enterbox("enter unique file name here", "enter the details", "test")
fileOut = str(output) + fileName + ".tiff"
cv2.imwrite(fileOut ,image)
print("image saved as: " + fileOut)
#=============================================================================
# TIME TO ANALYZE. LETS ASSUME THEY"RE NOT TOO SKEWED

# fidPatternImg generated previously
# image was saved from previously 

#generate verImg from image
# imageCols, imageRows = image.shape[::-1]
# image8b = cv2.normalize(image.copy(),
#                          np.zeros(shape=(imageRows, imageCols)),
#                          0, 255,
#                          norm_type=cv2.NORM_MINMAX,
#                          dtype=cv2.CV_8U)
# verImg = cv2.cvtColor(image8b .copy(), cv2.COLOR_GRAY2BGR)

# res = cv2.matchTemplate(image8b, fidPatternImg, cv2.TM_CCORR_NORMED)
# _, _, _, max_loc = cv2.minMaxLoc(res)

# #mark up location of fiducials in cyan
# cv2.rectangle(verImg,(max_loc[0], max_loc[1]),
#                               (max_loc[0]+fidPatternCols, max_loc[1]+fidPatternRows),
#                               (250,250,0),2)

# # address from bottom left fiducial to topleft array point
# relPosRows = -274
# relPosCols = 0

# ###### generate spot masks for ndimage analysis for means
# imgMaskFOREG = np.zeros(image.shape)
# imgMaskBACKList = []
# foreground_negative_OBO = np.zeros(image.shape)

# imgMaskFIDUC = np.zeros(image.shape)

# #now, iterate through all circles and mark mask, also recoding their row/col positioncircle_centerRow = int(array_setup["top_left_coords"][0])
# radii = int(array_setup["radii"])
# circlePositions = []
# rowPos = 0
# colPos = 0
# circle_centerRow = max_loc[1] + relPosRows + array_setup["row_pitch"]
# for eachRow in range(int(array_setup["rows"])):
#     rowPos = rowPos + 1
#     circle_centerCol = max_loc[0] + relPosCols + array_setup["col_pitch"]
#     colPos = 0
#     for eachCol in range(int(array_setup["cols"])):
#         colPos = colPos + 1
#         bg_image_mask = np.zeros(image.shape)
#         circlePositions.append([rowPos,colPos])
#         cv2.circle(verImg, 
#                    (round(circle_centerCol), round(circle_centerRow)),
#                    radii, 
#                    (0,0,255), 
#                    thickness = 1)
#         cv2.circle(imgMaskFOREG, 
#                    (round(circle_centerCol), round(circle_centerRow)),
#                    radii, 
#                    255, 
#                    thickness = -1)
#         cv2.circle(bg_image_mask,
#                    (round(circle_centerCol), round(circle_centerRow)),
#                    round(radii*2)+2, 
#                    255, 
#                    thickness = -1)
#         cv2.circle(foreground_negative_OBO, 
#                    (round(circle_centerCol), round(circle_centerRow)),
#                     radii+2, 
#                     1, 
#                     thickness = -1)
#         imgMaskBACKList.append(bg_image_mask)
#         circle_centerCol = circle_centerCol + int(array_setup["col_pitch"])
#     circle_centerRow = circle_centerRow + int(array_setup["row_pitch"])

# finalBGmasks = []
# for eachBGMask in imgMaskBACKList:
#     finalBGmasks.append(np.multiply(eachBGMask,foreground_negative_OBO))

# label_im, nb_labels = ndimage.label(imgMaskFOREG)
# spot_vals = ndimage.measurements.mean(image, label_im,
#                                           range(1, nb_labels+1))

# bgCalculated = []
# for eachBgMask in finalBGmasks:
#     # maskedSubImg = np.multiply(eachBgMask,subImage)
#     # cvWindow("mult result", maskedSubImg, False, 0)
#     label_bgEa, _ = ndimage.label(eachBgMask)
#     mean_bgEa = ndimage.measurements.mean(image, label_bgEa)
#     bgCalculated.append(mean_bgEa)
    
# csvRowOut = []
# csvRowOut.append(fileOut)
# sequence = array_setup["spot_index"]

# print(len(spot_vals))
# print(len(sequence))
# zipped = zip(sequence,spot_vals,bgCalculated,circlePositions)
# zipped = list(zipped)
# res = sorted(zipped, key = lambda x: x[0])

# print(res)
# with open('d4DataOutput-COV.csv', 'a', newline='') as csvfile:
#     csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for each in res:
#         csvRowOut.append("Row" + str(each[3][0]) + " Col" + str(each[3][1]) + " Group" + str(each[0]))
#         csvRowOut.append(each[1])
#         csvRowOut.append(each[2])
#         csvRowOut.append(each[1]-each[2])
#     csvWriter.writerow(csvRowOut)
#     print("csv written")

# fig, ax = plt.subplots() 
# ax.set_axis_off() 
# table = ax.table( 
#     cellText = np.round(np.reshape(spot_vals,(7,7))),  
#     cellLoc ='center',  
#     loc ='upper left')         
# table.set_fontsize(20)
   
# ax.set_title('Auto Analysis Data report (beta)', 
#              fontweight ="bold") 
# plt.show() 


# cv2.putText(verImg,
#                 "press q when done",
#                 (350,605),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (255, 255, 0),
#                 2,
#                 cv2.LINE_AA)

# cv2.imshow("marked image", verImg)
# cancelEarly = cv2.waitKey(0)
cv2.destroyAllWindows()


# save data as json and csv for data export and analysis
# use two csvs - foregrounds and backgrounds, and report with row and column numbers
# let user do whatever formatting they want later, we can do that in pos