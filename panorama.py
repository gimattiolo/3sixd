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
import math

#import torch

two_pi = 2 * math.pi

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
        self.f_pixels = 0
        self.h_pixels = 0
        self.ar = 0.0

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

def Angle2Dir(gamma_theta) :
    gamma = gamma_theta[0]
    theta = gamma_theta[1]
    return np.array([math.sin(theta) * math.sin(gamma), math.cos(theta), math.sin(theta) * math.cos(gamma)])

# returns 3, H, W
def Angle2Dir_vectorized(gamma_theta) :
    gamma = gamma_theta[:,:,0]
    theta = gamma_theta[:,:,1]
    X = np.sin(theta) * np.sin(gamma)
    Y = np.cos(theta)
    Z = np.sin(theta) * np.cos(gamma)
    return np.stack((X,Y,Z), axis=0)

def UV2Angle(uv) :
    gamma_theta = np.zeros((2,1), dtype=np.float32)
    #u=0 -> gamma=-pi, u=1 -> gamma=pi
    gamma_theta.x = math.pi * (2.0 * uv.x - 1.0)
    #v=0 -> theta=pi, v=1 -> theta=0
    gamma_theta.y = math.pi * (1.0 - uv.y) 
    return gamma_theta

def UV2Angle_vectorized(uv) :
    gamma_theta = np.zeros(uv.shape, dtype=np.float32)
    #u=0 -> gamma=-pi, u=1 -> gamma=pi
    gamma_theta[:, :, 0] = math.pi * (2.0 * uv[:, :, 0] - 1.0)     
    #v=0 -> theta=pi, v=1 -> theta=0
    gamma_theta[:, :, 1] = math.pi * (1.0 - uv[:, :, 1]) 
    return gamma_theta

def Lerp(a0, a1, x) :
    return a0 + (a1 - a0) * x

# a0 is a scalar
# a1 is H,W,3, 
# x is H,W
def Lerp_vectorized(a0, a1, x, out) :
    for i in range(3) :
        out[:,:,i] += a0 + (a1[:,:,i] - a0) * x
    
def Normalize(x) :
    return x / np.linalg.norm(x)

# x is assumed to be 3, H, W
def Normalize_vectorized(x) :
    norm = np.linalg.norm(x, axis=0)
    x[0,:,:] /= norm
    x[1,:,:] /= norm
    x[2,:,:] /= norm
    return x 

# def Angle2UV(gammaTheta, offset_rad) :
#     uv = np.zeros((2,1))

#     gammaTheta.x += offset_rad

#     two_pi = 2 * math.pi
#     #if (gammaTheta.x > two_pi) :
#     #  gammaTheta.x = gammaTheta.x - two_pi
    
#     gammaTheta.x = Lerp(gammaTheta.x, gammaTheta.x - math.pi, gammaTheta.x > two_pi)
				
#     uv.x = gammaTheta.x / two_pi

#     #ifdef PXR_FLIP_PORTAL_U
#     uv.x = 1.0 - uv.x
#     #endif
#     uv.y = gammaTheta.y / two_pi
#     #ifdef PXR_FLIP_PORTAL_V
#         #uv.y = 1.0 - uv.y
#     #endif
# 	#if (_Flip > 1.0) : 
# 	#	uv.y = 1.0 - uv.y
# 	#
#     #uv.y = lerp(uv.y, 1.0 - uv.y, _Flip > 1.0)
           
#     return uv

# def Dir2Angle(dir) :
#     # x: [0, UNITY_TWO_PI]
#     # y: [0, UNITY_PI] 
#     gammaTheta = np.zeros((2,1))
#     gammaTheta.y = math.acos(dir.y)

#     dir.y = 0.0
#     dir = Normalize(dir)

#     gammaTheta.x = math.acos(dir.x)
# 	#if(dir.z < 0.0)
#     #  gammaTheta.x = UNITY_TWO_PI - gammaTheta.x;
#     gammaTheta.x = Lerp(gammaTheta.x, two_pi - gammaTheta.x, dir.z < 0.0)

    return gammaTheta

def MakeUV(shape) :
    H,W = shape

    # np arrays j is column, along x
    def create_array_element_u(i, j):
        return j / (W-1) 

    # np arrays i is row, along y
    def create_array_element_v(i, j):
        return i / (H-1) 

    i_u = np.fromfunction(create_array_element_u, shape, dtype=np.float32)
    i_v = np.fromfunction(create_array_element_v, shape, dtype=np.float32)

    print(f'{i_u.min()}|{i_u.max()}')
    print(f'{i_v.min()}|{i_v.max()}')

    return np.stack((i_u, i_v), axis=2)

#each Ms[ci] is the transform from ci to c0
def ComputeWorldToC0(Ms_ci_c0) :

    num_cameras = len(Ms_ci_c0)

    Ps = [None] * num_cameras
    Vs = [None] * num_cameras

    eye = np.eye(4, dtype=np.float32)
    for i in range(num_cameras) :
        Vs[i] = np.dot(Ms_ci_c0[i], eye[:, 2:3])[0:3]
        Ps[i] = np.dot(Ms_ci_c0[i], eye[:, 3:])[0:3]

    A = np.zeros((3 * num_cameras, num_cameras), dtype=np.float32)
    b = np.zeros((3 * num_cameras, 1), dtype=np.float32)

    for i in range(num_cameras) :
        j = (i + 1) % num_cameras
        r = 3*i
        A[r:r+3, i:i+1] = Vs[i]; 
        A[r:r+3, j:j+1] = -Vs[j]

        b[r:r+3, :] = Ps[j] - Ps[i]  

    # num_cameras,1
    # in c0 reference framework
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    x0 = np.zeros((3, 1), dtype=np.float32)

    for i in range(num_cameras) :
        p = Ps[i] + x[i] * Vs[i] 
        x0 += p
    x0 /= num_cameras

    # return w -> c0
    eye[0:3, 3:] = x0
    return eye

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
    H, W = (1080, 1920)
    #H, W = (400, 400)

    size_default = (H,W) 

    empty_frame = np.zeros((H, W, 3), dtype=np.float32)

    empty_frame[:, :, 2] = 255.0 

    if not os.path.exists(args.path) :
        os.mkdir(args.path)

    if not os.path.exists(args.path) :
        print("Invalid path " + args.path)
        sys.exit(1)

    pairs = CalibrationUtilities.MakePairs(args.pairs, pin_ids)        

    if not pairs :
        return

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

    for pin_id, cameraDatum in cameraData.items() :
        pipeline=CalibrationUtilities.make_gstreamer_pipeline(sensor_id=cameraDatum.sensor_id, flip_method=flip_method)
        cameraDatum.capture = cv2.VideoCapture(pipeline, api_preference)
        print(f'sensor:{cameraDatum.sensor_id},pin:{pin_id},open:{cameraDatum.capture.isOpened()}')
    # create views in the window

    panorama = np.zeros((H, W, 3), np.float32)

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

        cameraDatum.f_pixels, cameraDatum.h_pixels, cameraDatum.ar = WaveUtilities.GetCalibrationParameters(camaraCalibrationLoaded, cameraDatum.IntrinsicMatrix, cameraDatum.Distortion, cameraDatum.ImageSize, pixelSize_m)
        
        print(f'Camera {c0} using focal length {cameraDatum.f_pixels}[pixels], image height {cameraDatum.h_pixels}[pixels], aspect ratio {cameraDatum.ar}, pixelSize {pixelSize_m}[m], scale {scale}')

    if not cameraCalibrationOK :
        print('Unable to load camera calibrations')
        sys.exit(1)

    stereoCalibrationOK = True

    ProjectionMatrices = {}
    ExtrinsicMatrices = {}

    first_pin_id = pin_ids[0]

    ExtrinsicMatrices[(first_pin_id, first_pin_id)] = np.identity(4, dtype=np.float32)

    # load extrinsics
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

        invR = np.transpose(R)
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
        # worldSpaceFilename = os.path.join(args.world_space_path, f'worldSpaceCalibration{c0}.json')
        # worldSpaceCalibrationLoaded, R, T, S = WaveUtilities.LoadWorldSpaceCalibration(worldSpaceFilename)
        # if worldSpaceCalibrationLoaded:
        #     # from world to c0
        #     WC3x4 = np.block( [
        #         [ R, T ],
        #     ] )

        #     WC4x4 = np.block( [
        #         [ WC3x4 ],
        #         [ np.array([ 0.0, 0.0, 0.0, 1.0 ]) ]
        #     ])

        #     invR = np.linalg.inv(R)
        #     invT = -np.dot(invR, T) 

        #     # from c0 to world
        #     invWC3x4 = np.block( [
        #         [ invR, invT ],
        #     ] )            

        #     invWC4x4 = np.block( [
        #         [ invWC3x4 ],
        #         [ np.array([ 0.0, 0.0, 0.0, 1.0 ]) ]
        #     ] )

        #     # store the matrix to transform from each camera's space to world space
        #     CamToWorldMatrices[c0] = invWC4x4
        #     WorldToCamMatrices[c0] = WC4x4
        # else:
        #     # identity
        #     CamToWorldMatrices[c0] = np.identity(4)
        #     WorldToCamMatrices[c0] = np.identity(4)

    if not stereoCalibrationOK :
        print('Unable to load stereo calibrations')
        sys.exit(1)

    # store matrices from c0 -> ci
    Ms_c0_ci = [None] * num_cameras
    Ms_ci_c0 = [None] * num_cameras
    for k0 in range(num_cameras) :
        pin_id = pin_ids[k0]

        cameraDatum = cameraData[pin_id]

        # from c0 -> ci
        Ms_c0_ci[k0] = np.identity(4)
        
        for i in range(0, k0) :
            key = (pin_ids[i], pin_ids[i + 1])
            Ms_c0_ci[k0] = np.dot(ExtrinsicMatrices[key], Ms_c0_ci[k0])
        
        Ms_ci_c0[k0] = CalibrationUtilities.invertExtrisics(Ms_c0_ci[k0])

    M_w_c0 = ComputeWorldToC0(Ms_ci_c0)

    for k0 in range(num_cameras) :
        pin_id = pin_ids[k0]

        cameraDatum = cameraData[pin_id]

        # we express everything in the camera c0 reference framework, i.e. camera c0 reference framework is the world reference framework
        
        #world - > ci
        E_w_ci_4x4 = np.dot(Ms_c0_ci[k0], M_w_c0)
        ProjectionMatrices[pin_id] = np.dot(cameraDatum.IntrinsicMatrix, E_w_ci_4x4[0:3, :])

    print("Running...")


    window_visible = True

    i_uv = MakeUV(size_default)
    gammaTheta = UV2Angle_vectorized(i_uv)
    ray_inW = Angle2Dir_vectorized(gammaTheta)
    ray_inW = ray_inW.reshape((3,-1))    

    pixel_coords= {}
    conditions = {}

    num_acculations = np.zeros((H,W), dtype=np.float32)

    for pin_id, cameraDatum in cameraData.items() :

        # in the following we assume the focal quad has size 1 x 1
        # and is at distance f along z relative to the camera
        # hfovAngle_rad = math.atan(0.5 * cameraDatum.h_pixels / cameraDatum.f_pixels)
        # hfovAngle_deg = math.degrees(hfovAngle_rad)
        
        # hh_meters = 0.5
        # hw_meters = hh_meters * cameraDatum.ar
        # half_size = np.array([hw_meters, hh_meters])
        # f_meters = hh_meters / math.tan(hfovAngle_rad)

        # for each pixel fo the 360m texture gets the ray in world space

        # _M_W2V = np.ones((3,3))

        # ray_inV = np.matmul(_M_W2V, ray_inW)
        # ray_inV = np.reshape(ray_inV, (3, H, W))
        # #3,H,W -> H,W,3
        # ray_inV = np.transpose(ray_inV, (1, 2, 0))

        # factor = f_meters / np.maximum(ray_inV[:, :, 2], 0.001)

        # ray_inV[:, :,0] *= factor
        # ray_inV[:, :,1] *= factor
        # ray_inV[:, :,2] *= factor

        # uvs = 0.5 * (1.0 + ray_inV[:, :, 0:2] / half_size)
        
        # image origin in top left 
        # image x is from left to right
        # image y is from top to bottom
        # z is forward
        # right handed
        ps = np.dot(ProjectionMatrices[pin_id][:, 0:3], ray_inW)
        ps[0:2, :] /= np.maximum(0.001, ps[2, :]) 
        ps = np.reshape(ps[0:2, :], (2, H, W))
        #2,H,W -> H,W,2
        ps = np.transpose(ps, (1, 2, 0)).astype(np.int32)

        pixel_x = ps[:, :, 0]
        pixel_y = ps[:, :, 1]

        mask_x = (0 <= pixel_x) & (pixel_x < W)
        mask_y = (0 <= pixel_y) & (pixel_y < H)
        condition = mask_x & mask_y

        num_acculations += condition

        # Combine into a single array of pixel coordinates
        # y is row, x is column
        pixel = np.stack((pixel_y, pixel_x), axis=2)

        condition_int = condition.astype(np.int32)

        # we do this so when sampling we don't have invalid pixel coordinates
        pixel[:,:,0] *= condition_int
        pixel[:,:,1] *= condition_int

        conditions[pin_id] = condition.astype(np.float32)
        pixel_coords[pin_id] = pixel

    print(f'{num_acculations.min()}|{num_acculations.max()}')

    accumulation_normalization = 1.0 / np.maximum(1.0, num_acculations)

    ray_inW = np.reshape(ray_inW, (3, H, W))
    ray_inW = np.transpose(ray_inW, (1, 2, 0))

    output_id = 0

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
        panorama.fill(0.0)
        for pin_id, cameraDatum in cameraData.items() :

            pixel = pixel_coords[pin_id]

            if cameraDatum.frame is None :
                # display empty frame
                color = empty_frame
            else :
                color = cameraDatum.frame

            color = color[pixel[:, :, 0], pixel[:, :, 1], :]

            ### debug ###

            #panorama[:, :, 2] = 255.0 * pixel[:, :, 1].astype(np.float32) / 1920.0
            
            #color.fill(0.0)
            #color[:,:, 2] = 255.0 * 0.5 * (1.0 + ray_inW[:, :, 0].astype(np.float32))
            #color[:,:, 2] = 255.0 * ray_inW[:, :, 0].astype(np.float32)

            #############

            Lerp_vectorized(0.0, color, conditions[pin_id], panorama)

        panorama[:,:,0] *= accumulation_normalization
        panorama[:,:,1] *= accumulation_normalization
        panorama[:,:,2] *= accumulation_normalization


        if key == ord('s') :
            filename = os.path.join(args.path, f'panorama_{output_id}.png')
            if cv2.imwrite(filename=filename, img=panorama) :
                output_id += 1
            else : 
                print(f'Unable to save into {filename}')
 
        cv2.imshow(window_name, panorama.astype(np.uint8))

    # When everything done, release the captures
    for pin_id, cameraDatum in cameraData.items() :
        cameraDatum.release()

    time.sleep(5)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
