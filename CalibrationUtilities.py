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

def MakePairs(input_pairs, pin_ids) :
    pairs = []
    
    num_pairs = len(input_pairs)

    for i in range(num_pairs-1) :
        j = (i+1)
        pairs.append((input_pairs[i], input_pairs[j]))

    # check pairs
    for c0, c1 in pairs :
        if not(c0 in pin_ids and c1 in pin_ids) :
            print(f'Incorrect pair {c0},{c1}')
            return None

    if True :
        pair_dict = {}
        # check pairs
        for pair in pairs :
            if pair in pair_dict :
                print(f'Incorrect pair {pair}')
                return None

            # this catches also the case when the pair has the same ids
            pair_inv = (pair[1], pair[0])
            if pair_inv in pair_dict :
                print(f'Repeated pair {pair}')
                return None

            pair_dict[pair] = None

    return pairs

def invertExtrisics(M) :
    
    invM = numpy.eye(4, dtype=numpy.float32)

    R = M[0:3,0:3]
    t = M[0:3,3:]
    invR = numpy.transpose(R)
    invT = numpy.dot(invR, -t) 
    invM[0:3,0:3] = invR
    invM[0:3,3:] = invT

    return invM

class Focuser:
    bus = None
    CHIP_I2C_ADDR = 0x0C

    def __init__(self, bus):
        self.focus_value = 0
        self.bus = bus
        pass
        
    def read(self):
        return self.focus_value

    def write(self, chip_addr, value):
        if value < 0:
            value = 0
        self.focus_value = value

        value = (value << 4) & 0x3ff0
        data1 = (value >> 8) & 0x3f
        data2 = value & 0xf0
        os.system("i2cset -y {} 0x{:02X} {} {}".format(self.bus, chip_addr, data1, data2))

    OPT_BASE    = 0x1000
    OPT_FOCUS   = OPT_BASE | 0x01
    OPT_ZOOM    = OPT_BASE | 0x02
    OPT_MOTOR_X = OPT_BASE | 0x03
    OPT_MOTOR_Y = OPT_BASE | 0x04
    OPT_IRCUT   = OPT_BASE | 0x05
    opts = {
        OPT_FOCUS : {
            "MIN_VALUE": 0,
            "MAX_VALUE": 1000,
            "DEF_VALUE": 0,
        },
    }
    def reset(self,opt,flag = 1):
        info = self.opts[opt]
        if info == None or info["DEF_VALUE"] == None:
            return
        self.set(opt,info["DEF_VALUE"])

    def get(self,opt,flag = 0):
        info = self.opts[opt]
        return self.read()

    def set(self,opt,value,flag = 1):
        info = self.opts[opt]
        if value > info["MAX_VALUE"]:
            value = info["MAX_VALUE"]
        elif value < info["MIN_VALUE"]:
            value = info["MIN_VALUE"]
        self.write(self.CHIP_I2C_ADDR, value)
        print("write: {}".format(value))

# image_size=(w,h)
def ComputeUndistortRectifyMap(fisheye, image_size,  calibration_image_size, camera_matrix, distortion_coefficients, balance=1.0, image_size2=None, image_size3=None) :
    R = None
    #m1type Type of the first output map, e.g., cv2.CV_16SC2 or cv2.CV_32F    
    ml_type=cv2.CV_16SC2
    if fisheye :
        #image_size is the dimension of input image to un-distort    
        assert image_size[0]/image_size[1] == calibration_image_size[0]/calibration_image_size[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"    
        if not image_size2:
            image_size2 = image_size    
        if not image_size3:
            image_size3 = image_size    
        scaled_camera_matrix = camera_matrix * image_size[0] / calibration_image_size[0]  # The values of Kcamera_matrix is to scale with image dimension.
        scaled_camera_matrix[2][2] = 1.0  # Except that camera_matrix[2][2] is always 1.0    # This is how scaled_camera_matrix, dim2 and balance are used to determine the final camera_matrix used to un-distort image. OpenCV document failed to make this clear!
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_camera_matrix, distortion_coefficients, image_size2, R, balance)
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(scaled_camera_matrix, distortion_coefficients, R, new_camera_matrix, image_size3, cv2.CV_16SC2)
        roi = (0, 0, image_size[0], image_size[1])
    else :
        # Refine the camera matrix (optional, as above)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, imageSize=image_size, alpha=balance, newImageSize=image_size, centerPrincipalPoint=False)
        # Compute the undistortion and rectification transformation maps once
        mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, R, new_camera_matrix, image_size, m1type=ml_type)
    return new_camera_matrix, roi, mapx, mapy

def UndistortImage(img, mapx, mapy) :
    return cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def CropUndistortedImage(img, roi) :
    x, y, w, h = roi
    return img[y:y+h, x:x+w]
