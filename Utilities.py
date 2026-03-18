import os
import CalibrationUtilities
import numpy
import cv2
import csv
import traceback
import time


MAX_NUM_FACES = 1

#NUM_FACE_LANDMARKS = 468 # no iris
NUM_FACE_LANDMARKS = 478 # for refined mesh, with iris

FACE_TOP_VERTEX = 10
FACE_LEFT_VERTEX = 454
FACE_RIGHT_VERTEX = 234
FACE_BOTTOM_VERTEX = 152
NOSE_TIP_VERTEX = 275
NOSE_FRONT_FLAT = 19
NOSE_REAR_FLAT = 94

NUM_BLENDSHAPES = 51

ls = list(range(0, NUM_FACE_LANDMARKS))
# ls = [FACE_TOP_VERTEX, FACE_BOTTOM_VERTEX, FACE_LEFT_VERTEX, FACE_RIGHT_VERTEX]

def UnprojectPoint(px, py, pz, aspectRatio, f, h, flipX) :
    # from the docs
    # x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
    # z represents the landmark depth with the depth at the head/wrist/hips being the origin, 
    # and the smaller the value the closer the landmark is to the camera. 
    # The magnitude of z uses roughly the same scale as x.    
    # Note that handedness is determined assuming the input image is mirrored, 
    #i.e., taken with a front-facing/selfie camera with images flipped horizontally. 
    # If it is not the case, please swap the handedness output in the application.

    # Observations
    # image reference framework start at the top left corner of the frame
    # both x and y are normalized: the bottom right corner in the frame is [1, 1]
    # we assume depth zero is at the focal length of the camera. Wrist landmark is always at depth 0
    # we also assume depth has same scale as x
    # x goes from left to right
    # y from top to bottom
    # camera forward is + z

    if flipX :
        # flip the x because the image is supposed to be mirrored, like a selfie camera 
        px = 1.0 - px
    # flip the y because mediapipe y points downwards 
    py = 1.0 - py

    xNormalization = aspectRatio * h
    yNormalization = h

    # denormalize 3d coordinates
    # from [0, 1] -> [-w/2, +w/2]
    px = xNormalization * (px - 0.5)
    # from [0, 1] -> [-h/2, +h/2]
    py = yNormalization * (py - 0.5)
    pz = xNormalization * pz
    
    return px, py, pz
    
def SendHandData(results, socket, ip, port, separator, f, h, aspectRatio, scale):
    if not results.multi_handedness or not results.multi_hand_landmarks:
        return

    headers = []
    for h_idx, hand in enumerate(results.multi_handedness):
        for c_idx, hand in enumerate(hand.classification):
            headers.append(hand.label + separator + str(hand.score) + separator)

    message = ''

    h_idx = -1
    for landmark_list in results.multi_hand_landmarks :
        h_idx = h_idx + 1
        message += headers[h_idx]
        if landmark_list:
            for l_idx, landmark in enumerate(landmark_list.landmark) :
                if ((landmark.HasField('visibility') and
                    landmark.visibility < VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                    landmark.presence < PRESENCE_THRESHOLD)):
                    continue

                px, py, pz = UnprojectPoint(aspectRatio, scale, landmark, f, h,flipX = True)

                message += str(l_idx) + separator + str(px) + separator + str(py) + separator + str(pz) + separator
                
    # print(message)
    bytes = message.encode('utf-8')    
    socket.sendto(bytes, (ip, port))
    
def SendBodyData(results, socket, ip, port, separator, f, h, aspectRatio, scale, flipX):
    landmark_list = results.pose_landmarks

    if not landmark_list:
        return

    # headers = []
    # for l_idx, landmark in enumerate(landmark_list.landmark):
        # headers.append(landmark.label + separator + str(landmark.score) + separator)
        # print(landmark)

    message = ''
    for l_idx, landmark in enumerate(landmark_list.landmark):
        # if ((landmark.HasField('visibility') and
            # landmark.visibility < VISIBILITY_THRESHOLD) or
            # (landmark.HasField('presence') and
            # landmark.presence < PRESENCE_THRESHOLD)):
            # continue

        px, py, pz = UnprojectPoint(aspectRatio, scale, landmark, f, h, flipX)

        message += str(l_idx) + separator + str(px) + separator + str(py) + separator + str(pz) + separator
                
    # print(message)
    bytes = message.encode('utf-8')    
    socket.sendto(bytes, (ip, port))

def SendBodyData_load(message, points3d, visibilities, socket, ip, port, separator):
    n = points3d.shape[1]

    for i in range(0, n):
        m = str(i) + separator + str(visibilities[i]) + separator + str(points3d[0, i]) + separator + str(points3d[1, i]) + separator + str(points3d[2, i]) + separator
        message += m
                
    bytes = message.encode('utf-8')    
    socket.sendto(bytes, (ip, port))

def SendBodyData(points3d, visibilities, socket, ip, port, separator):
    SendBodyData_load('', points3d, visibilities, socket, ip, port, separator)
    
def SendBodyData_timestamped(timestamp, points3d, visibilities, socket, ip, port, separator):
    message = str(timestamp) + separator
    SendBodyData_load(message, points3d, visibilities, socket, ip, port, separator)

def SendFaceData(weights, socket, ip, port, separator):
    message = ''

    n = len(weights)

    for i in range(0, n):
        message += str(i) + separator + str(weights[i])+ separator 
                
    bytes = message.encode('utf-8')    
    socket.sendto(bytes, (ip, port))

def LoadCameraCalibration(path) :
    IntrinsicMatrix = None
    Distortion = None 
    ReprojectionError = None 
    ImageSize = None

    print(f'Using calibration file {path}')
    ableToLoad = False
    if os.path.exists(path) and os.path.isfile(path) :
        fileObj = open(path, 'r')
        if not fileObj is None :
            jsonContent = fileObj.read()
            IntrinsicMatrix, Distortion, ReprojectionError, ImageSize = CalibrationUtilities.JsonToCameraCalibration(jsonContent)
            ableToLoad = True
    if not ableToLoad :
        print(f'Unable to open calibration file {path}')

    return ableToLoad, IntrinsicMatrix, Distortion, ReprojectionError, ImageSize 
    
def LoadStereoCalibration(path) :
    R = numpy.identity(3)
    T = numpy.zeros((3, 1))
    E = numpy.identity(3)
    F = numpy.identity(3)
    S = numpy.ones((1,1))
    
    print(f'Using stereo calibration file {path}')
    ableToLoad = False
    if os.path.exists(path) and os.path.isfile(path) :
        fileObj = open(path, 'r')
        if not fileObj is None :
            jsonContent = fileObj.read()
            IntrinsicMatrix1, Distortion1, IntrinsicMatrix2, Distortion2, R, T, E, F, S = CalibrationUtilities.JsonToStereoCalibration(jsonContent)
            ableToLoad = True
    if not ableToLoad :
        print(f'Unable to open stereo calibration file {path}')

    return ableToLoad, R, T, E, F, S

def LoadWorldSpaceCalibration(path) :
    R = numpy.identity(3)
    T = numpy.zeros((3, 1))
    S = numpy.ones((1,1))
    
    print(f'Using worldspace calibration file {path}')
    ableToLoad = False
    if os.path.exists(path) and os.path.isfile(path) :
        fileObj = open(path, 'r')
        if not fileObj is None :
            jsonContent = fileObj.read()
            IntrinsicMatrix1, Distortion1, R, T, S = CalibrationUtilities.JsonToWorldSpaceCalibration(jsonContent)
            ableToLoad = True
    if not ableToLoad :
        print(f'Unable to open worldspace calibration file {path}')

    return ableToLoad, R, T, S

def GetCalibrationParameters(ableToLoad, IntrinsicMatrix, Distortion, ImageSize, pixelSize_m) :
    # in m
    # see https://en.wikipedia.org/wiki/35_mm_format
    w_35mm = 0.036
    h_35mm = 0.024
    f_35mm = 0.035

    f = f_35mm
    w = w_35mm
    h = h_35mm

    if ableToLoad :
        # average between horizontal and vertical focal length
        f = pixelSize_m * 0.5 * (IntrinsicMatrix[0][0] + IntrinsicMatrix[1][1])
        # we store channels, width and height
        w = pixelSize_m * ImageSize[1]
        h = pixelSize_m * ImageSize[2]

    ar = w / h

    return f, h, ar

# see https://amytabb.com/ts/2019_06_28/
# OCV is right handed 
# camera reference framework x goes to left, y is up, z goes away from the camera
# image reference framework, origin bottom left , x left to right, y bottom up. Camera center / pricipal point is not at the origin
# def OCVTOUnity(R, T) :
    # return R, T

def ComputeFrameScales(frameSizes, desiredSize) :
    frameScales = [1.0] * len(frameSizes)

    for i in range(0, len(frameSizes)) :
        if frameSizes[i][0] >= frameSizes[i][1] :
            frameScales[i] = desiredSize[0] / frameSizes[i][0]
        else :
            frameScales[i] = desiredSize[1] / frameSizes[i][1]

    return frameScales
    
def ApplyFrameScaling(frames, frameSizes, frameScales, concatFrames, desiredOffset) :
    for i in range(0, len(frames)) :
        scaledSize = ( (int)(frameScales[i] * frameSizes[i][0]), (int)(frameScales[i] * frameSizes[i][1]))
        scaledFrame = cv2.resize(frames[i], (scaledSize[0], scaledSize[1]))
        concatFrames[i][ desiredOffset[1] : desiredOffset[1] + scaledSize[1], desiredOffset[0] : desiredOffset[0] + scaledSize[0] ] = scaledFrame

def SmartNormalize(v, epsilon, default) :
    n = numpy.linalg.norm(v)
    if n < epsilon :
        print(f'Trying to normalize nearly zero length vector {v}')
        #for line in traceback.format_stack():
        #    print(line.strip())        
        return default
    return v / n 

#see https://cristal.univ-lille.fr/~casiez/1euro/
class OneEuroFilter :
    #params
    
    # data update rate in Hz
    rate = 0.0
    # cut off frequency for dx in Hz
    dCutOff = 0.0
    # minimum cut off frequency for x in Hz
    minCutOff = 0.0
    # multiplier to the modulus of dx used to update the cut off for x
    beta = 0.0

    #internal
    firstTime = True
    dxFilter = None
    xFilter = None
    
    def __init__(self, rate, minCutOff, dCutOff, beta) :
        self.firstTime = True
        self.rate = rate
        self.minCutOff = minCutOff
        self.dCutOff = dCutOff
        self.beta = beta
        
        self.dxFilter = OneEuroFilter.LowPassFilter()
        self.xFilter = OneEuroFilter.LowPassFilter()
        
    # rate: data update rate in Hz
    # cutOff: cut off frequency in Hz
    def Alpha(rate, cutOff) :
        tau = 1.0 / (2.0 * numpy.pi * cutOff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)    

    def Filter(self, x) :
        if self.firstTime :
            self.firstTime = False
            dx = 0.0
        else :
            dx = (x - self.xFilter.hatxprev) * self.rate
        edx = self.dxFilter.Filter(dx, OneEuroFilter.Alpha(self.rate, self.dCutOff))
        #TODO replace norm with something faster?
        cutOff = self.minCutOff + self.beta * numpy.linalg.norm(edx)
        x1 = self.xFilter.Filter(x, OneEuroFilter.Alpha(self.rate, cutOff))
        return x1
    
    class LowPassFilter :
        firstTime = True
        hatxprev = 0.0

        def __init__(self) :
            firstTime = True

        def Filter(self, x, alpha) :
            if self.firstTime :
                self.firstTime = False
                self.hatxprev = x
            hatx = alpha * x + (1.0 - alpha) * self.hatxprev
            self.hatxprev = hatx
            return hatx 

class MovingAverageFilter :
    #params

    #window size
    size = 1
    samples = [] 
    average = 0.0
    oldestIndex = 0
    invSize = 1
    
    def __init__(self, size) :
        self.size = max(size, 1)
        self.invSize = 1.0 / self.size
        self.samples = [0.0] * self.size
        self.average = 0.0
        self.oldestIndex = 0

    def Filter(self, x) :
        self.average += self.invSize * (x - self.samples[self.oldestIndex])

        self.samples[self.oldestIndex] = x        
        self.oldestIndex = (self.oldestIndex + 1) % self.size
        
        return self.average
        
def TransformLandmarks_expressionInvariant(points, epsilon, scaleEstimation) :
    top = points[:, FACE_TOP_VERTEX]
    bottom = points[:, FACE_BOTTOM_VERTEX]
    left = points[:, FACE_LEFT_VERTEX]
    right = points[:, FACE_RIGHT_VERTEX]

    n = len(ls)

    center = numpy.zeros(3)
    for i in range(0, n) :
        center += points[:, ls[i]]
    center /= n

    h = 0.0
    for i in range(0, n) :
        temp = points[:, ls[i]] - center 
        h += numpy.linalg.norm(temp)
    h /= n

    scale = 1.0
    if scaleEstimation :
        if h > epsilon :
            scale = 1.0 / h
    y1 = top - center
    y1 = SmartNormalize(y1, epsilon, numpy.array([0.0, 1.0, 0.0]))

    x1 = left - right 
    x1 = x1 - numpy.dot(x1, y1) * y1
    x1 = SmartNormalize(x1, epsilon, numpy.array([1.0, 0.0, 0.0]))

    z1 = numpy.cross(x1, y1)

    # x1 -> (1, 0, 0) 
    # y1 -> (0, 1, 0) 
    # z1 -> (0, 0, 1) 
    R = numpy.identity(3)
    R[0, :] = x1
    R[1, :] = y1
    R[2, :] = z1
    
    points1 = numpy.copy(points)
    
    #print(f'x:{x1} y:{y1} z:{z1}')

    for i in range(0, NUM_FACE_LANDMARKS) :
        points1[:, i] = scale * R.dot(points1[:, i] - center)
        
    return points1

def TransformLandmarks(points, epsilon) :

    top = points[:, FACE_TOP_VERTEX]
    bottom = points[:, FACE_BOTTOM_VERTEX]
    left = points[:, FACE_LEFT_VERTEX]
    right = points[:, FACE_RIGHT_VERTEX]

    # n is number of samples and m is number of features/coordinates
    # m, n = points.shape

    # sum all the columns
    center = points.sum(axis=1) / NUM_FACE_LANDMARKS

    h = numpy.linalg.norm(top - bottom)
    scale = 1.0
    if h > epsilon :
        scale = 1.0 / h

    # we use the flat part under the nose between the nostrels to find the z of the face
    z1 = points[:, NOSE_REAR_FLAT] - points[:, NOSE_FRONT_FLAT];
    z1 = SmartNormalize(z1, epsilon, numpy.array([0.0, 0.0, 1.0]))
    x1 = left - right;
    x1 = SmartNormalize(x1, epsilon, numpy.array([1.0, 0.0, 0.0]))

    y1 = numpy.cross(z1, x1)

    # x1 -> (1, 0, 0) 
    # y1 -> (0, 1, 0) 
    # z1 -> (0, 0, 1) 
    R = numpy.identity(3)
    R[0, :] = x1
    R[1, :] = y1
    R[2, :] = z1

    for i in range(0, NUM_FACE_LANDMARKS) :
        points[:, i] = scale * R.dot(points[:, i] - center)
    return points

def LoadCSV(filePath) :
    lines = []
    fileHandle = None
    success = False
    
    try :
        fileHandle = open(filePath, encoding='utf-8', newline='')
    except Exception as e :
        print(f'Invalid path {filePath}')
        return success, lines
    
    lines = list(csv.reader(fileHandle))

    m = len(lines)
    if m > 0 :
        print(f'{m} line(s) loaded from {filePath}')
        success = True
    else :
        print(f'Invalid number of lines {m}')

    fileHandle.close()

    return success, lines

def SaveBlendshapes(serializedBlendshapes, filePath) :
    fileHandle = None
    try :
        fileHandle = open(filePath, mode='w', encoding='utf-8', newline='')
    except Exception as e :
        print(f'Unable to open {filePath}: {e}')
        return False

    csvWriter = csv.writer(fileHandle, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    m = len(serializedBlendshapes)

    for i in range(0, m) :
        row = []
        for j in range(0, len(serializedBlendshapes[i])) :
            row.append('%.6f' % serializedBlendshapes[i][j])
        #print(f'Saving row {i} with {len(row)} entries')
        csvWriter.writerow(row)

    fileHandle.close()

    print(f'{m} blendshape(s) saved to {filePath}')

    return True

def GetFaceLandmarks(face_mesh_results, image_cols, image_rows, landmarks, visibility_threshold, presence_threshold) :
    faceIndex = 0
    
    if face_mesh_results.multi_face_landmarks:
        for landmark_list in face_mesh_results.multi_face_landmarks:

            if landmark_list:

                #if ((landmark.HasField('visibility') and landmark.visibility < visibility_threshold) 
                #    or (landmark.HasField('presence') and landmark.presence < presence_threshold)):
                #    continue

                landmarksEnumerate = enumerate(landmark_list.landmark)
                
                for i, landmark in landmarksEnumerate:
                    idx = faceIndex * NUM_FACE_LANDMARKS + i
                    landmarks[0, idx] = landmark.x
                    landmarks[1, idx] = landmark.y
                    landmarks[2, idx] = landmark.z

        faceIndex = faceIndex + 1

def SerializeBlendshape(blendshape) :
    return blendshape.flatten()
    
def DeserializeBlendshape(blendshape) :
    return numpy.reshape(blendshape, (3, NUM_FACE_LANDMARKS), 'F')    
    