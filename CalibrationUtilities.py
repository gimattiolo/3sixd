import os
import re
import json
import numpy
import cv2
import sys
sys.path.append(r'.')
#import pymf

# def GetAvailableCameras():
#     return pymf.get_MF_devices()

# def GetAvailableCameraIndices():
#     nameList = pymf.get_MF_devices()    
#     indexList = []
#     index = 0
#     for name in nameList:
#         indexList.append(index)
#         index += 1
#     return indexList

# the filename is {captureIndex}_{cameraIndex}.ext
def GetCameraIndex(filename) :
    index = -1
 
    regex  = '_[0-9]+.'
    match = re.search(regex, filename)

    if match is None :
        return index

    matchSpan = match.span()
    matchStr = filename[ matchSpan[0] : matchSpan[1] ]
    if matchStr is None :
        return index
    matchStr = matchStr.replace('_', '')
    matchStr = matchStr.replace('.', '')
    
    try :
        index = int(matchStr)
    except :
        index = -1
    return index

# the filename is {captureIndex}_{cameraIndex}.ext
def GetCaptureIndex(filename) :
    index = -1
 
    regex  = '[0-9]+_'
    match = re.search(regex, filename)

    if match is None :
        return index

    matchSpan = match.span()
    matchStr = filename[ matchSpan[0] : matchSpan[1] ]
    if matchStr is None :
        return index
    matchStr = matchStr.replace('_', '')
    
    try :
        index = int(matchStr)
    except :
        index = -1
    return index
    
def GetCaptureName(captureIndex, cameraIndex, ext) :
    # camera index is simply related to the order
    # camera identifier is the opencv camera identifier
    return str(captureIndex) + '_' + str(cameraIndex) + ext

def ArrayToList(array) :
    shape = array.shape
    arrayRowMajor = array.copy()
    arrayRowMajor.shape = array.size, 1

    vector = []
    for i in range(0, array.size) :
        vector.append(arrayRowMajor[i][0])
        
    return vector

def ListToArray(vector, shape) :
    array = numpy.asarray(vector)
    array.shape = shape
    return array

def CameraCalibrationToJson(IntrinsicMatrix, Distortion, ReprojectionError, ImageSize) :
    IntrinsicMatrixVector = ArrayToList(IntrinsicMatrix)
    DistortionVector = ArrayToList(Distortion)
    
    obj = {
        'IntrinsicMatrix': IntrinsicMatrixVector, 
        'IntrinsicMatrixShape': IntrinsicMatrix.shape, 
        'Distortion': DistortionVector, 
        'DistortionShape': Distortion.shape, 
        'ImageSize' : ImageSize,
        'ReprojectionError' : ReprojectionError,
    }
    
    return json.dumps(obj, sort_keys = False, indent = 4)

def JsonToCameraCalibration(jsonContent) :
    obj = json.loads(jsonContent)

    IntrinsicMatrix = ListToArray(obj['IntrinsicMatrix'], obj['IntrinsicMatrixShape'])
    Distortion = ListToArray(obj['Distortion'], obj['DistortionShape'])
    ImageSize = obj['ImageSize']
    ReprojectionError = obj['ReprojectionError']
    
    return IntrinsicMatrix, Distortion, ReprojectionError, ImageSize

# R - Rotation Matrix between first and second camera coordinate systems.
# T - Translation vector between the coordinate systems of the cameras.
# E - Essential matrix.
# F - Fundamental matrix.
# S - Scale factor
def StereoCalibrationToJson(IntrinsicMatrix1, Distortion1, IntrinsicMatrix2, Distortion2, R, T, E, F, S) :
    IntrinsicMatrix1Vector = ArrayToList(IntrinsicMatrix1)
    Distortion1Vector = ArrayToList(Distortion1)
    IntrinsicMatrix2Vector = ArrayToList(IntrinsicMatrix2)
    Distortion2Vector = ArrayToList(Distortion2)
    RVector = ArrayToList(R)
    TVector = ArrayToList(T)
    EVector = ArrayToList(E)
    FVector = ArrayToList(F)
    SVector = ArrayToList(S)
    obj = {
        'IntrinsicMatrix1': IntrinsicMatrix1Vector, 
        'IntrinsicMatrix1Shape': IntrinsicMatrix1.shape, 
        'Distortion1': Distortion1Vector, 
        'Distortion1Shape': Distortion1.shape, 
        'IntrinsicMatrix2': IntrinsicMatrix2Vector, 
        'IntrinsicMatrix2Shape': IntrinsicMatrix2.shape, 
        'Distortion2': Distortion2Vector, 
        'Distortion2Shape': Distortion2.shape, 
        'R': RVector, 
        'RShape': R.shape, 
        'T': TVector, 
        'TShape': T.shape, 
        'E': EVector, 
        'EShape': E.shape, 
        'F': FVector, 
        'FShape': F.shape, 
        'S' : SVector,
        'SShape': S.shape, 
    }
    
    return json.dumps(obj, sort_keys = False, indent = 4)

def JsonToStereoCalibration(jsonContent) :
    obj = json.loads(jsonContent)

    IntrinsicMatrix1 = ListToArray(obj['IntrinsicMatrix1'], obj['IntrinsicMatrix1Shape'])
    Distortion1 = ListToArray(obj['Distortion1'], obj['Distortion1Shape'])
    IntrinsicMatrix2 = ListToArray(obj['IntrinsicMatrix2'], obj['IntrinsicMatrix2Shape'])
    Distortion2 = ListToArray(obj['Distortion2'], obj['Distortion2Shape'])

    R = ListToArray(obj['R'], obj['RShape'])
    T = ListToArray(obj['T'], obj['TShape'])
    E = ListToArray(obj['E'], obj['EShape'])
    F = ListToArray(obj['F'], obj['FShape'])
    # for retro-compatibility
    if 'S' in obj and 'SShape' in obj :
        S = ListToArray(obj['S'], obj['SShape'])
    else :
        S = numpy.ones((1, 1))

    return IntrinsicMatrix1, Distortion1, IntrinsicMatrix2, Distortion2, R, T, E, F, S

# R - Rotation Matrix between first and second camera coordinate systems.
# T - Translation vector between the coordinate systems of the cameras.
# S - Scale factor
def WorldSpaceCalibrationToJson(IntrinsicMatrix1, Distortion1, R, T, S) :
    IntrinsicMatrix1Vector = ArrayToList(IntrinsicMatrix1)
    Distortion1Vector = ArrayToList(Distortion1)
    RVector = ArrayToList(R)
    TVector = ArrayToList(T)
    SVector = ArrayToList(S)
    obj = {
        'IntrinsicMatrix1': IntrinsicMatrix1Vector, 
        'IntrinsicMatrix1Shape': IntrinsicMatrix1.shape, 
        'Distortion1': Distortion1Vector, 
        'Distortion1Shape': Distortion1.shape, 
        'R': RVector, 
        'RShape': R.shape, 
        'T': TVector, 
        'TShape': T.shape, 
        'S' : SVector,
        'SShape': S.shape, 
    }
    
    return json.dumps(obj, sort_keys = False, indent = 4)

def JsonToWorldSpaceCalibration(jsonContent) :
    obj = json.loads(jsonContent)

    IntrinsicMatrix1 = ListToArray(obj['IntrinsicMatrix1'], obj['IntrinsicMatrix1Shape'])
    Distortion1 = ListToArray(obj['Distortion1'], obj['Distortion1Shape'])

    R = ListToArray(obj['R'], obj['RShape'])
    T = ListToArray(obj['T'], obj['TShape'])
    # for retro-compatibility
    if 'S' in obj and 'SShape' in obj :
        S = ListToArray(obj['S'], obj['SShape'])
    else :
        S = numpy.ones((1, 1))

    return IntrinsicMatrix1, Distortion1, R, T, S

def CameraUndistort(image, IntrinsicMatrix, distortion) :
    h, w = image.shape[:2]
    IntrinsicMatrix1, roi = cv2.getOptimalNewCameraMatrix(IntrinsicMatrix, distortion, (w,h), 0 ,(w,h))
    # undistort
    undistorted = cv2.undistort(image, IntrinsicMatrix, distortion, None, IntrinsicMatrix1)
    x, y, w, h = roi
    undistorted = undistorted[y : y + h, x : x + w]
    
    return undistorted

def CameraRemap(image, intrinsicMatrix, distortion) :
    h,  w = image.shape[:2]
    IntrinsicMatrix1, roi = cv2.getOptimalNewCameraMatrix(intrinsicMatrix, distortion, (w,h), 1 ,(w,h))

    mapx, mapy = cv2.initUndistortRectifyMap(intrinsicMatrix, distortion, None, IntrinsicMatrix1, (w,h), 5)
    undistorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    undistorted = undistorted[y : y + h, x : x + w]
    
    return undistorted

def GetCameras() :
    #left camera eye, right camera eye
    cameras = [0, 1]
    return cameras    
    
def GetPatternSize() :
    patternSize = (6, 4)
    return patternSize

def GetSearchSize() :
    searchSize = (11, 11)
    return searchSize

def GetZeroZoneSize() :
    zeroZoneSize = (-1, -1)
    return zeroZoneSize
    
def GetCalibrationPath() :
    path = 'C:/Users/gmattiolo/gits/pythonscripts/calibration17'
    return path
    
def GetCalibrationPatternSquareSideLengthInMeters() :
    return 1.0
        
def make_gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
