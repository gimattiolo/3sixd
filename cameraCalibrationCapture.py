import numpy as np
import sys
import cv2
import argparse
import glob
import os
import CalibrationUtilities
import subprocess
import re
import time

class CameraObject :

    def reset(self) :
        self.sensor_id = 0
        self.pin_index = 0
        self.identifier = ''
        self.file = ''
        self.capture = None
        self.frame = None
        self.decoratedFrames = None
        self.foundGrid = False


    def __init__(self) :
        self.reset()

    def release(self) :
        
        if self.capture :
            self.capture.release()
        self.reset()

    def __repr__(self) :
        return f'{self.pin_index}|{self.identifier}||{self.sensor_id}|{self.file}'

def ScanCameras(filter=None) :
    subprocess_out = subprocess.check_output(["v4l2-ctl", "--list-devices"]) 
    subprocess_out_str = str(subprocess_out)

    print(subprocess_out_str)
    identifiers = re.findall(pattern='(platform:tegra-capture-vi:[0-9]+)', string=subprocess_out_str)
    files = re.findall(pattern='/dev/video[0-9]+', string=subprocess_out_str)

    # if filter : 

    #     identifiers = filter

    num_cameras = len(identifiers)

    if num_cameras <= 0 :
        print('Found no camera')
        sys.exit(0)

    assert num_cameras == len(files)

    # print(identifiers)

    cameraObjects = {}
    for i in range(num_cameras) :
        cameraObject = CameraObject()
        match = re.search(pattern='[0-9]+', string=identifiers[i])
        assert match is not None
        cameraObject.pin_index = int(match.group())
        cameraObject.identifier = identifiers[i]
        cameraObject.file = files[i]
        match = re.search(pattern='[0-9]+', string=files[i])
        assert match is not None
        cameraObject.sensor_id = int(match.group())        
        cameraObjects[cameraObject.pin_index] = cameraObject

    # sort the entries by pin
    sorted_items = sorted(cameraObjects.items())
    cameraObjects = dict(sorted_items)

    return cameraObjects

def main():
    parser = argparse.ArgumentParser('Capture calibration images')
    parser.add_argument('--listcameras', dest='list_cameras', action="store_true", help='list available cameras and their indices, and exit')
    parser.add_argument('--cameraindices', dest='camera_indices', type=int, nargs='*', help='indices of cameras to capture from')
    #parser.add_argument('--fileindices', dest='file_indices', type=int, nargs='*', help='corresponding indices for each camera when writing file names, must be same size as --cameraindices')
    parser.add_argument('--path', type=str, help='set the capture destination folder')
    parser.add_argument('--append', type=bool, default=True, help='append images into capture destination folder')
    parser.add_argument('--auto', metavar='MS', default=-1, type=int, help='autocapture images, delayed by MS milliseconds')
    parser.add_argument('--patternsize', dest='pattern_size', type=int, nargs=2, help='2D size of checkerboard pattern to detect')
    args = parser.parse_args()

    if args.list_cameras:
        # nameList = CalibrationUtilities.GetAvailableCameras()
        nameList = []    
        index = 0
        for name in nameList:
            print ('%d: %s' % (index, name))
            index += 1
        sys.exit(0)

    FORCE_SAVE = False

    AUTO_SAVE = False
    #in msec
    waitKeyPeriod = 1
    if args.auto >= 0:
        AUTO_SAVE = True
        waitKeyPeriod = args.auto
        
    AppendMode = args.append

    # requestedCameras = CalibrationUtilities.GetCameras()
    # if args.camera_indices:
    #     requestedCameras = args.camera_indices
    # availableCameras = CalibrationUtilities.GetAvailableCameraIndices()
    # cameras = []
    # for r in requestedCameras:
    #     if r in availableCameras:
    #         cameras.append(r)

    
    # cameras = [0,1,2,3,4]

    cameraObjects = ScanCameras()

    num_cameras = len(cameraObjects)

    pins = list(cameraObjects.keys())


    #file_indices = []
    # if args.file_indices:
    #     file_indices = args.file_indices
    # else:
    # for pin_index in pins:
    #     file_indices.append(pin_index)

    print(f'Using cameras:{cameraObjects}')
    #print(f'Using indices:{file_indices}')

    # if len(file_indices) != len(cameraObjects):
    #     print('File indices count != camera indice count, exiting.')
    #     sys.exit(1)

    calibrationPath = CalibrationUtilities.GetCalibrationPath()
    if args.path:
        calibrationPath = args.path        
    print(f'Using calibration path {calibrationPath}')      

    patternSize = CalibrationUtilities.GetPatternSize()
    if args.pattern_size:
        patternSize = tuple(args.pattern_size)
    print(f'Using pattern size {patternSize}')

    ext = '.png'

    # width, height
    size_default = (480, 480)

    if not os.path.exists(calibrationPath) :
        os.mkdir(calibrationPath)

    if not os.path.exists(calibrationPath) :
        print("Invalid path " + calibrationPath)
        sys.exit(1)

    print("Creating capture objects...")
    concatFrames = []
    # without this images are upside down
    flip_method = 2
    api_preference=cv2.CAP_GSTREAMER
    for pin, camera in cameraObjects.items() :
        pipeline=CalibrationUtilities.make_gstreamer_pipeline(sensor_id=camera.sensor_id, flip_method=flip_method)
        camera.capture = cv2.VideoCapture(pipeline, api_preference)
        print(f'sensor:{camera.sensor_id},pin:{pin},open:{camera.capture.isOpened()}')
        concatFrames.append(np.zeros((size_default[1], size_default[0], 3), np.uint8))

    
    print("Analyzing previous captures...")
    running = True
    captureIndex = -1
    if AppendMode :
        fileList = os.listdir(calibrationPath)
        for i in range(0, len(fileList)):

            filename = fileList[i]
        
            se = os.path.splitext(filename)
        
            if len(se) < 2 or se[1] != ext :
                continue
        
            thisCameraIndex = CalibrationUtilities.GetCameraIndex(filename)
            if thisCameraIndex in pins:
                thisCaptureIndex = CalibrationUtilities.GetCaptureIndex(filename)
                if thisCaptureIndex > captureIndex :
                    captureIndex = thisCaptureIndex
    captureIndex += 1
    
    print(f'Appending captures starting with index {captureIndex}')

    #get size of concat
    frameSizes = [(0, 0)] * num_cameras
    frameScales = [1.0] * num_cameras
    for i in range(num_cameras) :
        
        pin = pins[i]
        camera = cameraObjects[pin]
        
        if camera.capture.isOpened() :
            counter = 0
            while counter < 3 :
                counter += 1
                ret, camera.frame = camera.capture.read()
                if not ret :
                    time.sleep(1)
                    continue
                # shape returns height, width, channels
                frameShape = camera.frame.shape
                
                frameSizes[i] = (frameShape[1], frameShape[0])
                if frameSizes[i][0] > frameSizes[i][1] :
                    frameScales[i] = size_default[0] / frameSizes[i][0]
                else :
                    frameScales[i] = size_default[1] / frameSizes[i][1]
                break            
            if counter >= 3 :
                print(f'{pin} not reading frames')
                exit(1)
    print("Running...")

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    origin = (0,150)
    fontScale              = 5
    fontColor              = (0,0,255) # red in BGR
    thickness              = 10
    lineType               = cv2.LINE_8

    while running :

        camerasOK = True
        for pin, camera in cameraObjects.items() :
            camera.foundGrid = False
            if camera.capture.isOpened() :
                # Capture frame-by-frame
                ret, camera.frame = camera.capture.read()

                if not ret :
                    print(f'{pin} not reading frames')
                    continue

                camera.decoratedFrame = camera.frame.copy()
                # gray = cv2.cvtColor(decoratedFrames[i], cv2.COLOR_BGR2GRAY)
                # ret, corners = cv2.findChessboardCorners(gray, patternSize, None)
                # if ret :
                #     foundGridPerCamera[i] = True
                #     decoratedFrames[i] = cv2.drawChessboardCorners(decoratedFrames[i], patternSize, corners, ret)                    
                # if ret :
                    # cornersSubPix = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)
        
                text = f'ID#{camera.sensor_id}|PIN#{pin}'
                cv2.putText(camera.decoratedFrame, text, 
                    origin, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType,
                    bottomLeftOrigin=False)
            
            else :
                camerasOK = False
        # Display the resulting frame

        if not camerasOK :
            print('Unable to open all the required cameras')
            continue

        key = cv2.waitKey(waitKeyPeriod)
        # if cv2.waitKey(waitKeyPeriod) & 0xFF == ord('q') :
        if AUTO_SAVE or key == ord('s'):
        
            camerasSeeingGrid = []
            for pin, camera in cameraObjects.items() :
                if camera.foundGrid :
                    camerasSeeingGrid.append(pin)

            if FORCE_SAVE or len(camerasSeeingGrid) > 1 :
                for pin in camerasSeeingGrid :
                    camera = cameraObjects[pin]
                    
                    # the filename is {captureIndex}_{fileCameraIndex}.ext
                    filename = os.path.join(calibrationPath, CalibrationUtilities.GetCaptureName(captureIndex, pin, ext))
                    print("Saving image " + filename)
                    cv2.imwrite(filename, camera.frame)
                captureIndex += 1
            else :
                pass
                #print("Checkerboard not visible in enough images! Skipping save")
        
        if key == ord('q'):
            running = False
            break

        # y, x
        offset = (0, 0)

        for i in range(num_cameras) :
            pin = pins[i]
            camera = cameraObjects[pin]
            scaledSize = ( (int)(frameScales[i] * frameSizes[i][0]), (int)(frameScales[i] * frameSizes[i][1]))
            scaledFrame = cv2.resize(camera.decoratedFrame, (scaledSize[0], scaledSize[1]))
            concatFrames[i][ offset[1] : offset[1] + scaledSize[1], offset[0] : offset[0] + scaledSize[0] ] = scaledFrame

        windowFrame = cv2.hconcat(concatFrames)
        
        cv2.imshow('CameraCalibrationCapture', windowFrame)

    # When everything done, release the capture
    for pin, camera in cameraObjects.items() :
        camera.release()

    time.sleep(5)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()