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
import shutil

class CameraDatum :

    def reset(self) :
        self.sensor_id = -1
        self.pin_id = -1
        self.identifier = ''
        self.file = ''
        self.capture = None
        self.frame = None
        self.decoratedFrames = None
        self.foundGrid = False
        self.size = (0,0)
        self.scale = 0

    def __init__(self) :
        self.reset()

    def release(self) :
        
        if self.capture :
            self.capture.release()
        self.reset()

    def __repr__(self) :
        return f'{self.pin_id}|{self.identifier}||{self.sensor_id}|{self.file}'

def ScanCameras(allowed_pins) :
    subprocess_out = subprocess.check_output(["v4l2-ctl", "--list-devices"]) 
    subprocess_out_str = str(subprocess_out)

    print(subprocess_out_str)
    identifiers = re.findall(pattern='(platform:tegra-capture-vi:[0-9]+)', string=subprocess_out_str)
    files = re.findall(pattern='/dev/video[0-9]+', string=subprocess_out_str)

    num_cameras = len(identifiers)

    if num_cameras <= 0 :
        print('Found no camera')
        sys.exit(0)

    assert num_cameras == len(files)

    # print(identifiers)

    camerData = {}
    for i in range(num_cameras) :
        cameraDatum = CameraDatum()
        match = re.search(pattern='[0-9]+', string=identifiers[i])
        assert match is not None
        cameraDatum.pin_id = int(match.group())
        cameraDatum.identifier = identifiers[i]
        cameraDatum.file = files[i]
        match = re.search(pattern='[0-9]+', string=files[i])
        assert match is not None
        cameraDatum.sensor_id = int(match.group())        
        camerData[cameraDatum.pin_id] = cameraDatum

    # sort the entries by pin
    sorted_items = sorted(camerData.items())
    camerData = dict(sorted_items)

    if allowed_pins :
        pins = list(camerData.keys())
        for pin in pins :
            if pin not in allowed_pins :
                del camerData[pin]

    return camerData

class CaptureDatum :

    def reset(self) :
        self.tuple = []
        self.counter = 0

    def __init__(self) :
        self.reset()

def main():
    parser = argparse.ArgumentParser('Capture calibration images')
    #parser.add_argument('--fileindices', dest='file_indices', type=int, nargs='*', help='corresponding indices for each camera when writing file names, must be same size as --cameraindices')
    parser.add_argument('--path', type=str, help='set the capture destination folder')
    parser.add_argument('--save_mode', type=int, default=0, help='0:append images into capture destination folder,1: delete content before starting')
    parser.add_argument('--auto', metavar='MS', default=-1, type=int, help='autocapture images, delayed by MS milliseconds')
    parser.add_argument('--patternsize', dest='pattern_size', type=int, nargs=2, help='2D size of checkerboard pattern to detect')
    parser.add_argument('--num_shots_per_capture', type=int, default=10, help='number of images per capture')
    parser.add_argument('--capture_delta_time_sec', type=int, default=5, help='capture delta time in seconds')
    parser.add_argument('--intrinsics_captures', type=int, nargs='+', help='capture sequence for intrisics')
    parser.add_argument('--extrinsics_captures', type=int, nargs='+', help='capture sequence for extrinsics')
    parser.add_argument('--allowed_pins', type=int, nargs='+', help='allowed pins')
    parser.add_argument('--flip_methods', type=int, nargs='+', help='flip methods')
    args = parser.parse_args()

    # if args.allowed
    # args.allowed_pins.sort()
    args.intrinsics_captures.sort()

    # if args.list_cameras:
    #     # nameList = CalibrationUtilities.GetAvailableCameras()
    #     nameList = []    
    #     index = 0
    #     for name in nameList:
    #         print ('%d: %s' % (index, name))
    #         index += 1
    #     sys.exit(0)

    AUTO_SAVE = False
    #in msec
    waitKeyPeriod = 1
    if args.auto >= 0:
        AUTO_SAVE = True
        #waitKeyPeriod = args.auto
        
    SaveMode = args.save_mode

    # requestedCameras = CalibrationUtilities.GetCameras()
    # if args.camera_indices:
    #     requestedCameras = args.camera_indices
    # availableCameras = CalibrationUtilities.GetAvailableCameraIndices()
    # cameras = []
    # for r in requestedCameras:
    #     if r in availableCameras:
    #         cameras.append(r)

    
    # allowed_pins = [1,2,3,4,5]
    #allowed_pins = [1, 3, 5]

    cameraData = ScanCameras(args.allowed_pins)

    num_cameras = len(cameraData)

    assert num_cameras >= 0

    assert(len(args.flip_methods) == num_cameras)

    pin_ids = list(cameraData.keys())


    #file_indices = []
    # if args.file_indices:
    #     file_indices = args.file_indices
    # else:
    # for pin_index in pins:
    #     file_indices.append(pin_index)

    print(f'Using cameras:{cameraData}')
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
    size_default = (1920, 1080)
    #size_default = (400, 400)

    if not os.path.exists(calibrationPath) :
        os.mkdir(calibrationPath)

    if not os.path.exists(calibrationPath) :
        print("Invalid path " + calibrationPath)
        sys.exit(1)

    print("Creating capture objects...")

    NUM_VIEWS = 2

    concatFrames = [None] * NUM_VIEWS
    # (0): none             - Identity (no rotation)
    # (1): counterclockwise - Rotate counter-clockwise 90 degrees
    # (2): rotate-180       - Rotate 180 degrees
    # (3): clockwise        - Rotate clockwise 90 degrees
    # (4): horizontal-flip  - Flip horizontally
    # (5): upper-right-diagonal - Flip across upper right/lower left diagonal
    # (6): vertical-flip    - Flip vertically
    # (7): upper-left-diagonal - Flip across upper left/low
    # without this images are upside down
    api_preference=cv2.CAP_GSTREAMER

    for k in range(len(pin_ids)) :
        pin_id = pin_ids[k]
        cameraDatum = cameraData[pin_id]
        flip_method = args.flip_methods[k]
        pipeline=CalibrationUtilities.make_gstreamer_pipeline(sensor_id=cameraDatum.sensor_id, flip_method=flip_method)
        cameraDatum.capture = cv2.VideoCapture(pipeline, api_preference)
        print(f'sensor:{cameraDatum.sensor_id},pin:{pin_id},open:{cameraDatum.capture.isOpened()}')
    # create views in the window
    for i in range(NUM_VIEWS) :
        concatFrames[i] = np.zeros((size_default[1], size_default[0], 3), np.uint8)

    
    print("Analyzing previous captures...")
    running = True
    nextFileCaptureIndex = -1
    if SaveMode == 0 :
        # append
        fileList = os.listdir(calibrationPath)
        for i in range(0, len(fileList)):

            filename = fileList[i]
        
            se = os.path.splitext(filename)
        
            if len(se) < 2 or se[1] != ext :
                continue
        
            pin_id = CalibrationUtilities.GetCameraIndex(filename)
            if pin_id in pin_ids:
                thisCaptureIndex = CalibrationUtilities.GetCaptureIndex(filename)
                if thisCaptureIndex > nextFileCaptureIndex :
                    nextFileCaptureIndex = thisCaptureIndex
    elif SaveMode == 1 :
        # delete
        shutil.rmtree(calibrationPath, ignore_errors=False, onerror=None)
        os.mkdir(calibrationPath)
    else :
        print(f'Unsupported save mode:{SaveMode}')
        exit(1)

    nextFileCaptureIndex += 1
    firstCaptureIndex = nextFileCaptureIndex
    
    print(f'Appending captures starting with index {nextFileCaptureIndex}')



    #get size of concat
    for pin_id, cameraDatum in cameraData.items() :
        if cameraDatum.capture.isOpened() :
            counter = 0
            while counter < 3 :
                counter += 1
                ret, cameraDatum.frame = cameraDatum.capture.read()
                if not ret :
                    time.sleep(1)
                    continue
                # shape returns height, width, channels
                frameShape = cameraDatum.frame.shape
                
                cameraDatum.size = (frameShape[1], frameShape[0])
                if cameraDatum.size[0] > cameraDatum.size[1] :
                    cameraDatum.scale = size_default[0] / cameraDatum.size[0]
                else :
                    cameraDatum.scale = size_default[1] / cameraDatum.size[1]
                break            
            if counter >= 3 :
                print(f'{pin_id} not reading frames')
                exit(1)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    origin = (0,150)
    fontScale              = 5
    fontColor              = (0,0,255) # red in BGR
    thickness              = 10
    lineType               = cv2.LINE_8

    num_simultanous = 2

    captureData = []

    # cycle = False
    # for i in range(num_cameras) :
    #     # single camera for intrisics
    #     captureDatum = CaptureDatum()
    #     captureDatum.tuple.append(pin_ids[i])
    #     captureData.append(captureDatum)

    #     # camera pairs for extrisics
    #     captureDatum = CaptureDatum()

    #     if not cycle and i == num_cameras-1 :
    #         break 
    #     for j in range(num_simultanous) :
    #         captureDatum.tuple.append(pin_ids[(i + j) % num_cameras])
    #     captureData.append(captureDatum)

    for pin_id in args.intrinsics_captures :
        # single camera for intrisics
        captureDatum = CaptureDatum()
        captureDatum.tuple.append(pin_id)
        captureData.append(captureDatum)

    pairs = CalibrationUtilities.MakePairs(args.extrinsics_captures, pin_ids)

    if not pairs :
        return

    for pair in pairs :
        # single camera for intrisics
        captureDatum = CaptureDatum()
        captureDatum.tuple.append(pair[0])
        captureDatum.tuple.append(pair[1])
        captureData.append(captureDatum)

    num_captures_expected = args.num_shots_per_capture * len(captureData)

    current_capture_id = 0

    now = time.time()
    lastGridTime = 0

    forceDetection = False

    startCaptureTime = now

    window_name = 'CameraCalibrationCapture'

    print("Running...")

    window_visible = True

    captureCompleted = False
    while running :
        now = time.time()
        deltaTime = now - lastGridTime

        if not captureCompleted and captureData[current_capture_id].counter == args.num_shots_per_capture :
            current_capture_id += 1

        wasCaptureCompleted = captureCompleted
        captureCompleted = current_capture_id >= len(captureData)

        if captureCompleted and not wasCaptureCompleted :
            endCaptureTime = now
            num_captures_actual = nextFileCaptureIndex - firstCaptureIndex
            assert num_captures_actual == num_captures_expected, f'Captures|Actual:{num_captures_actual}|Expected:{num_captures_expected}'
            text_info = f'{num_captures_expected} captures in {endCaptureTime - startCaptureTime:,.3f}sec'
            print(f'{text_info}')

        #print(deltaTime)
        detectGrid = (deltaTime > args.capture_delta_time_sec)
        if detectGrid :
            lastGridTime = now

        for pin_id, cameraDatum in cameraData.items() :
            cameraDatum.foundGrid = False

        camerasOK = True
        foundGridInAllViews = detectGrid and not captureCompleted
        for pin_id, cameraDatum in cameraData.items() :
            if cameraDatum.capture.isOpened() :
                # Capture frame-by-frame
                ret, cameraDatum.frame = cameraDatum.capture.read()

                if not ret :
                    print(f'{pin_id} not reading frames')
                    continue

                cameraDatum.decoratedFrame = cameraDatum.frame.copy()

                text_pin = f'Pin{pin_id}'

                if not captureCompleted :
                    pin_in_process = pin_id in captureData[current_capture_id].tuple
                    if pin_in_process :
                        text_pin += f'#'
                    if detectGrid and pin_in_process :
                    
                        gray = cv2.cvtColor(cameraDatum.decoratedFrame, cv2.COLOR_BGR2GRAY)
                        cameraDatum.foundGrid, corners = cv2.findChessboardCorners(gray, patternSize, None)

                        cameraDatum.foundGrid = cameraDatum.foundGrid | forceDetection

                        foundGridInAllViews = foundGridInAllViews and cameraDatum.foundGrid

                        if cameraDatum.foundGrid :
                            cameraDatum.decoratedFrame = cv2.drawChessboardCorners(cameraDatum.decoratedFrame, patternSize, corners, ret)                    

                        # if ret :
                            # cornersSubPix = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

                cv2.putText(cameraDatum.decoratedFrame, 
                    text_pin, 
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
        
            if foundGridInAllViews :
                msg = f'Saved '
                for pin_id in captureData[current_capture_id].tuple :
                    cameraDatum = cameraData[pin_id]
                    
                    # the filename is {nextFileCaptureIndex}_{fileCameraIndex}.ext
                    filename = CalibrationUtilities.GetCaptureName(nextFileCaptureIndex, pin_id, ext)
                    filepath = os.path.join(calibrationPath, filename)
                    msg += f'{filename},'

                    cv2.imwrite(filepath, cameraDatum.frame)
                print(f'{msg} | {captureData[current_capture_id].tuple} | {current_capture_id=} | {captureData[current_capture_id].counter}')

                nextFileCaptureIndex += 1

                captureData[current_capture_id].counter += 1

            else :
                pass
                #print("Checkerboard not visible in enough images! Skipping save")

        if key == ord('q') :#or not window_visible:
            running = False
            break

        # if cv2.getWindowProperty("foo", cv2.WND_PROP_VISIBLE):
        #     window_visible = True
        # else :
        #     window_visible = False

        # print(window_visible)

        # y, x

        if NUM_VIEWS != num_cameras :
            for i in range(len(concatFrames)) :
                concatFrames[i].fill(0.0)

        capture_list = pin_ids 
        if not captureCompleted and NUM_VIEWS != num_cameras :
            capture_list = captureData[current_capture_id].tuple

        offset = (0, 0)
        frame_id = 0        
        for pin_id in capture_list :
            cameraDatum = cameraData[pin_id]
            scaledSize = ( (int)(cameraDatum.scale * cameraDatum.size[0]), (int)(cameraDatum.scale * cameraDatum.size[1]))
            scaledFrame = cv2.resize(cameraDatum.decoratedFrame, (scaledSize[0], scaledSize[1]))
            concatFrames[frame_id][ offset[1] : offset[1] + scaledSize[1], offset[0] : offset[0] + scaledSize[0] ] = scaledFrame
            frame_id += 1
            if frame_id >= NUM_VIEWS :
                break

        windowFrame = cv2.hconcat(concatFrames)
        
        if not captureCompleted :
            text_info = f'{args.capture_delta_time_sec - deltaTime:,.3f}/{args.capture_delta_time_sec}|{captureData[current_capture_id].counter}/{args.num_shots_per_capture}|{current_capture_id}/{len(captureData)}'

        # if not captureCompleted :
        cv2.putText(windowFrame, 
            text_info, 
            org=(0,400), 
            fontFace=font, 
            fontScale=fontScale,
            color=(255,0,0), # blue in BGR
            thickness=thickness,
            lineType=lineType,
            bottomLeftOrigin=False)

        cv2.imshow(window_name, windowFrame)

    # When everything done, release the captures
    for pin_id, cameraDatum in cameraData.items() :
        cameraDatum.release()

    time.sleep(5)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
