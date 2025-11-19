import numpy as np
import sys
import cv2
import argparse
import glob
import os
import CalibrationUtilities


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    # SENSOR_ID=1
    # FRAMERATE=21
    # width=4032
    # height=3040
    #gst-launch-1.0 nvarguscamerasrc sensor-id=$SENSOR_ID ! "video/x-raw(memory:NVMM),width=$width,height=$height,framerate=$FRAMERATE/1" ! nvvidconv ! nvegltransform ! nveglglessink -e

    # return (
    #     "nvarguscamerasrc sensor-id=%d ! "
    #     "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
    #     "nvvidconv flip-method=%d ! "
    #     "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
    #     "videoconvert ! "
    #     "video/x-raw, format=(string)BGR ! appsink"
    #     % (
    #         sensor_id,
    #         capture_width,
    #         capture_height,
    #         framerate,
    #         flip_method,
    #         display_width,
    #         display_height,
    #     )

    return (
        'nvarguscamerasrc sensor-id=%d ! '
        '"video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=%d/1" ! '
        'nvvidconv flip-method=%d ! nvegltransform ! nveglglessink -e'
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            
        )


    )

def main():
    parser = argparse.ArgumentParser('Capture calibration images')
    parser.add_argument('--listcameras', dest='list_cameras', action="store_true", help='list available cameras and their indices, and exit')
    parser.add_argument('--cameraindices', dest='camera_indices', type=int, nargs='*', help='indices of cameras to capture from')
    parser.add_argument('--fileindices', dest='file_indices', type=int, nargs='*', help='corresponding indices for each camera when writing file names, must be same size as --cameraindices')
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

    AUTO_SAVE = True
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

    cameras = [0]

    #############################
    name = '/dev/video1'


    pipeline=gstreamer_pipeline(sensor_id=1)
    print(pipeline)
    a = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    print(f'opened:{a.isOpened()}')

    exit(1)
    #############################





    file_indices = []
    if args.file_indices:
        file_indices = args.file_indices
    else:
        index = 0
        for cam in cameras:
            file_indices.append(index)
            index += 1

    print(f'Using cameras {cameras}')
    print(f'Using indices {file_indices}')

    if len(file_indices) != len(cameras):
        print('File indices count != camera indice count, exiting.')
        sys.exit(1)

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
    captures = []
    frames = []
    decoratedFrames = []
    foundGridPerCamera = []
    concatFrames = []
    for i in range(0, len(cameras)) :

        #cv::CAP_FFMPEG or cv::CAP_IMAGES or cv::CAP_DSHOW.
        a = cv2.VideoCapture('/dev/video1')



        #captures.append( cv2.VideoCapture(cameras[i], cv2.CAP_DSHOW) )
        captures.append(a)

        print(f'{captures[i].isOpened()=}')

        exit(1)



        frames.append(None)
        decoratedFrames.append(None)
        foundGridPerCamera.append(False)
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
            if thisCameraIndex in file_indices:
                thisCaptureIndex = CalibrationUtilities.GetCaptureIndex(filename)
                if thisCaptureIndex > captureIndex :
                    captureIndex = thisCaptureIndex
    captureIndex += 1
    
    print(f'Appending captures starting with index {captureIndex}')

    #get size of concat
    frameSizes = [(0, 0)] * len(cameras)
    frameScales = [1.0] * len(cameras)
    for i in range(0, len(cameras)) :
        if captures[i].isOpened() :
            ret, frames[i] = captures[i].read()
            # shape returns height, width, channels
            frameShape = frames[i].shape
            
            frameSizes[i] = (frameShape[1], frameShape[0])
            if frameSizes[i][0] > frameSizes[i][1] :
                frameScales[i] = size_default[0] / frameSizes[i][0]
            else :
                frameScales[i] = size_default[1] / frameSizes[i][1]
            
    print("Running...")

    while running :

        camerasOK = True
        for i in range(0, len(cameras)) :
            foundGridPerCamera[i] = False
            if captures[i].isOpened() :
                # Capture frame-by-frame
                ret, frames[i] = captures[i].read()

                decoratedFrames[i] = frames[i].copy()

                if ret :
                    gray = cv2.cvtColor(decoratedFrames[i], cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, patternSize, None)
                    if ret :
                        foundGridPerCamera[i] = True
                        decoratedFrames[i] = cv2.drawChessboardCorners(decoratedFrames[i], patternSize, corners, ret)                    
                    # if ret :
                        # cornersSubPix = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)
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
            for c in range(0, len(cameras)) :
                if foundGridPerCamera[c] :
                    camerasSeeingGrid.append(c)

            if FORCE_SAVE or len(camerasSeeingGrid) > 1 :
                for cameraIndex in camerasSeeingGrid :
                    fileCameraIndex = file_indices[cameraIndex]
                    # the filename is {captureIndex}_{fileCameraIndex}.ext
                    filename = os.path.join(calibrationPath, CalibrationUtilities.GetCaptureName(captureIndex, fileCameraIndex, ext))
                    print("Saving image " + filename)
                    cv2.imwrite(filename, frames[cameraIndex])
                captureIndex += 1
            else :
                print("Checkerboard not visible in enough images! Skipping save")
        
        if key == ord('q'):
            running = False
            break

        # y, x
        offset = (0, 0)

        for i in range(0, len(cameras)) :
            scaledSize = ( (int)(frameScales[i] * frameSizes[i][0]), (int)(frameScales[i] * frameSizes[i][1]))
            scaledFrame = cv2.resize(decoratedFrames[i], (scaledSize[0], scaledSize[1]))
            concatFrames[i][ offset[1] : offset[1] + scaledSize[1], offset[0] : offset[0] + scaledSize[0] ] = scaledFrame

        windowFrame = cv2.hconcat(concatFrames)
        
        cv2.imshow('CameraCalibrationCapture', windowFrame)

    # When everything done, release the capture
    for i in range(0, len(cameras)) :
        captures[i].release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()