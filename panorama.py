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

    cameraData = {}
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
        cameraData[cameraDatum.pin_id] = cameraDatum

    # sort the entries by pin
    sorted_items = sorted(cameraData.items())
    cameraData = dict(sorted_items)

    if allowed_pins :
        pins = list(cameraData.keys())
        for pin in pins :
            if pin not in allowed_pins :
                del cameraData[pin]

    return cameraData

class CaptureDatum :

    def reset(self) :
        self.tuple = []
        self.counter = 0

    def __init__(self) :
        self.reset()

def main():
    parser = argparse.ArgumentParser('Panorama')
    parser.add_argument('--path', type=str, help='set the capture destination folder')
    parser.add_argument('--save_mode', type=int, default=0, help='0:append images into capture destination folder,1: delete content before starting')
    args = parser.parse_args()

    #in msec
    waitKeyPeriod = 1
        
    SaveMode = args.save_mode
    
    # allowed_pins = [1,2,3,4,5]
    allowed_pins = [1, 3, 5]

    cameraData = ScanCameras(allowed_pins)

    num_cameras = len(cameraData)

    assert num_cameras >= 0

    pin_ids = list(cameraData.keys())

    print(f'Using cameras:{cameraData}')

    print(f'Using path {args.path}')      

    ext = '.png'

    # width, height
    size_default = (1920, 1080)
    #size_default = (400, 400)

    if not os.path.exists(args.path) :
        os.mkdir(args.path)

    if not os.path.exists(args.path) :
        print("Invalid path " + args.path)
        sys.exit(1)

    print("Creating capture objects...")

    # (0): none             - Identity (no rotation)
    # (1): counterclockwise - Rotate counter-clockwise 90 degrees
    # (2): rotate-180       - Rotate 180 degrees
    # (3): clockwise        - Rotate clockwise 90 degrees
    # (4): horizontal-flip  - Flip horizontally
    # (5): upper-right-diagonal - Flip across upper right/lower left diagonal
    # (6): vertical-flip    - Flip vertically
    # (7): upper-left-diagonal - Flip across upper left/low
    # without this images are upside down
    flip_method = 2
    api_preference=cv2.CAP_GSTREAMER

    black_view = np.zeros((size_default[1], size_default[0], 3), np.uint8)

    for pin_id, cameraDatum in cameraData.items() :
        pipeline=CalibrationUtilities.make_gstreamer_pipeline(sensor_id=cameraDatum.sensor_id, flip_method=flip_method)
        cameraDatum.capture = cv2.VideoCapture(pipeline, api_preference)
        print(f'sensor:{cameraDatum.sensor_id},pin:{pin_id},open:{cameraDatum.capture.isOpened()}')
    # create views in the window

    panorama = np.zeros((size_default[1], size_default[0], 3), np.uint8)
    
    running = True
    if SaveMode == 0 :
        pass
    elif SaveMode == 1 :
        # delete
        shutil.rmtree(args.path, ignore_errors=False, onerror=None)
        os.mkdir(args.path)
    else :
        print(f'Unsupported save mode:{SaveMode}')
        exit(1)

    captureIndex = 0
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    origin = (0,150)
    fontScale              = 5
    fontColor              = (0,0,255) # red in BGR
    thickness              = 10
    lineType               = cv2.LINE_8

    num_simultanous = 2

    now = time.time()

    startCaptureTime = now

    window_name = 'Panorama'

    print("Running...")

    window_visible = True

    while running :
        now = time.time()

        camerasOK = True
        for pin_id, cameraDatum in cameraData.items() :
            if cameraDatum.capture.isOpened() :
                # Capture frame-by-frame
                ret, cameraDatum.frame = cameraDatum.capture.read()

                if not ret :
                    print(f'{pin_id} not reading frames')
                    continue

            else :
                camerasOK = False
        # Display the resulting frame

        if not camerasOK :
            print('Unable to open all the required cameras')
            continue

        key = cv2.waitKey(waitKeyPeriod)
        # if cv2.waitKey(waitKeyPeriod) & 0xFF == ord('q') :

        if key == ord('q') :#or not window_visible:
            running = False
            break

        # if cv2.getWindowProperty("foo", cv2.WND_PROP_VISIBLE):
        #     window_visible = True
        # else :
        #     window_visible = False

        # print(window_visible)

        cv2.imshow(window_name, panorama)

    # When everything done, release the captures
    for pin_id, cameraDatum in cameraData.items() :
        cameraDatum.release()

    time.sleep(5)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()