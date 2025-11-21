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
import WaveUtilities

class CameraDatum :

    def reset(self) :
        self.sensor_id = -1
        self.pin_id = -1
        self.identifier = ''
        self.file = ''
        self.capture = None
        self.frame = None
        self.IntrinsicMatrix = None
        self.Distortion = None
        self.ReprojectionError = None
        self.ImageSize = None

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
    parser.add_argument('--intrinsic_path', type=str, help='set the intrinsic image capture source folder/data export folder')
    parser.add_argument('--extrinsic_path', type=str, help='set the extrinsic image capture source folder/data export folder')
    parser.add_argument('--world_space_path', type=str, default='', help='set the world space capture source folder/data export folder')
    parser.add_argument('--pairs', type=int, nargs='+', help='pairs of cameras for stereo calibration')

    
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

    pairs = CalibrationUtilities.MakePairs(args.pairs, pin_ids)        

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

    # SD (Standard Definition)	640 x 480	4:3	480p
    # HD (High Definition	1280 x 720	16:9	720p
    # Full HD	1920 x 1080	16:9	1080p
    # 2K	2048 x 1152	1:1.77	N/A
    # UHD	3840 x 2160	16:9	Sometimes called “2160p” and often mistakenly referred to as “4k”
    # DCI 4K	4096 x 2160	1:1.9	Just 4K (the “DCI” part is sometimes dropped)
    

    # tipically [1.7, 2.2, 3.5, 4.6, 6.5, 7.0, 10.0, 14.0] micrometers
    # see https://www.vision-doctor.com/en/camera-technology-basics/sensor-and-pixel-sizes.html#:~:text=Industrial%20cameras%20usually%20use%201,with%20the%20same%20pixel%20size.
    pixelSize_m = 1.0
    scale = 0.1    


    cameraCalibrationOK = True

    for k0 in range(0, num_cameras) :
        c0 = pin_ids[k0]

        cameraDatum = cameraData[c0]

        cameraFilename = os.path.join(args.intrinsic_path, f'calibration{c0}.json')
        
        camaraCalibrationLoaded, cameraDatum.IntrinsicMatrix, cameraDatum.Distortion, cameraDatum.ReprojectionError, cameraDatum.ImageSize = WaveUtilities.LoadCameraCalibration(cameraFilename)

        cameraCalibrationOK = cameraCalibrationOK and camaraCalibrationLoaded

        if not cameraCalibrationOK :
            break

        cameraDatum.f, cameraDatum.h, cameraDatum.ar = WaveUtilities.GetCalibrationParameters(camaraCalibrationLoaded, cameraDatum.IntrinsicMatrix, cameraDatum.Distortion, cameraDatum.ImageSize, pixelSize_m)
        
        print(f'Camera {c0} using focal length {cameraDatum.f}[m], image height {cameraDatum.h}[m], aspect ratio {cameraDatum.ar}, pixelSize {pixelSize_m}[m], scale {scale}')

    if not cameraCalibrationOK :
        print('Unable to load camera calibrations')
        sys.exit(1)

    stereoCalibrationOK = True

    ProjectionMatrices = {}
    ExtrinsicMatrices = {}
    CamToWorldMatrices = {}
    WorldToCamMatrices = {}

    first_pin_id = pin_ids[0]

    ExtrinsicMatrices[(first_pin_id, first_pin_id)] = np.identity(4)
    ProjectionMatrices[(first_pin_id, first_pin_id)] = np.dot(cameraData[c0].IntrinsicMatrix, np.block([ [ np.identity(3), np.zeros((3, 1)) ] ]))

    # store extrinsic
    for c0, c1 in pairs :
        key = (c0, c1)
        invKey = (c1, c0)

        stereoFilename = os.path.join(args.extrinsic_path, f'stereoCalibration{c0}_{c1}.json')
        stereoCalibrationLoaded, R, T, E, F, S = WaveUtilities.LoadStereoCalibration(stereoFilename)

        stereoCalibrationOK = stereoCalibrationOK and stereoCalibrationLoaded
        
        if not stereoCalibrationOK :
            break

        # print(f'Stereo pair {c0}->{c1} using\nR=\n{R}\nT=\n{T}\nS={S}')

        
        # from c0 to c1
        E3x4 = np.block( [
            [ R, T ],
        ] )

        invR = np.linalg.inv(R)
        #invT = -inverse(R) * T
        invT = -np.dot(invR, T) 

        # from c1 to c0
        invE3x4 = np.block( [
            [ invR, invT ],
        ] )

        E4x4 = np.block( [
            [ E3x4 ],
            [ np.array([ 0.0, 0.0, 0.0, 1.0 ]) ]
        ] )

        invE4x4 = np.block( [
            [ invE3x4 ],
            [ np.array([ 0.0, 0.0, 0.0, 1.0 ]) ]
        ] )

        ExtrinsicMatrices[key] = E4x4
        ExtrinsicMatrices[invKey] = invE4x4

        # load worldspace info
        worldSpaceFilename = os.path.join(args.world_space_path, f'worldSpaceCalibration{c0}.json')
        worldSpaceCalibrationLoaded, R, T, S = WaveUtilities.LoadWorldSpaceCalibration(worldSpaceFilename)
        if worldSpaceCalibrationLoaded:
            # from world to c0
            WC3x4 = np.block( [
                [ R, T ],
            ] )

            WC4x4 = np.block( [
                [ WC3x4 ],
                [ np.array([ 0.0, 0.0, 0.0, 1.0 ]) ]
            ])

            invR = np.linalg.inv(R)
            invT = -np.dot(invR, T) 

            # from c0 to world
            invWC3x4 = np.block( [
                [ invR, invT ],
            ] )            

            invWC4x4 = np.block( [
                [ invWC3x4 ],
                [ np.array([ 0.0, 0.0, 0.0, 1.0 ]) ]
            ] )

            # store the matrix to transform from each camera's space to world space
            CamToWorldMatrices[c0] = invWC4x4
            WorldToCamMatrices[c0] = WC4x4
        else:
            # identity
            CamToWorldMatrices[c0] = np.identity(4)
            WorldToCamMatrices[c0] = np.identity(4)

    if not stereoCalibrationOK :
        print('Unable to load stereo calibrations')
        sys.exit(1)

    # store projection matrices from c0 -> c
    for k0 in range(num_cameras) :
        c = pin_ids[k0]

        cameraDatum = cameraData[c]

        c0 = pin_ids[0]

        # from c0 to c
        key = (c0, c)
        
        # we express everything in the camera c0 reference framework, i.e. camera c0 reference framework is the world reference framework
        
        # from c0 -> c
        E_0_c_4x4 = np.identity(4)
        
        for i in range(0, k0) :
            E_0_c_4x4 = np.dot(ExtrinsicMatrices[(pin_ids[i], pin_ids[i + 1])], E_0_c_4x4)
        
        E_0_c_3x4 = E_0_c_4x4[0:3, :]
        ProjectionMatrices[key] = np.dot(cameraDatum.IntrinsicMatrix, E_0_c_3x4)

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


        # make panorama





        cv2.imshow(window_name, panorama)

    # When everything done, release the captures
    for pin_id, cameraDatum in cameraData.items() :
        cameraDatum.release()

    time.sleep(5)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()