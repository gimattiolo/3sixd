import sys
import argparse
import os
import subprocess
import re
import time
import shutil
import math
import threading
import subprocess

import multiprocessing

# os.environ['LD_PRELOAD'] = '/home/gimattiolo/gits/3sixd/.venv/lib/python3.8/site-packages/torch.libs/libgomp-d22c30c5.so.1.0.0'

#import torch

import cupy as cp
import numpy as np
import cv2
import ffmpeg

import Utilities
import CalibrationUtilities
from VulkanCompute import VulkanCompute 

two_pi = 2 * math.pi

# 0:cpu,1:cuda,2:vulkan
COMPUTE_MODE = 2

class CameraDatum :

    def reset(self) :
        self.sensor_id = -1
        self.pin_id = -1
        self.identifier = ''
        self.file = ''
        self.capture = None
        self.frame_bgr = None # always normalized to [0,1]
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
        self.frame_bgr = None
        if self.capture :
            self.capture.release()
        self.reset()

    def __repr__(self) :
        return f'{self.pin_id}|{self.identifier}||{self.sensor_id}|{self.file}'

def ScanCameras(pin_data) :

    cameraData = {}
    if Script.args.benchmark :
        num_cameras = 6

        for i in range(num_cameras) :
            cameraDatum = CameraDatum()
            cameraDatum.pin_id = i
            cameraDatum.identifier = i
            cameraDatum.file = i
            cameraDatum.sensor_id = i        
            cameraData[cameraDatum.pin_id] = cameraDatum

    else :

        subprocess_out = subprocess.check_output(['v4l2-ctl', '--list-devices']) 
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

    if pin_data :
        pins = list(cameraData.keys())
        for pin in pins :
            if pin not in pin_data :
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

def UV2Angle(uv, alpha) :
    gamma_theta = np.zeros((2,1), dtype=np.float32)
    #u=0 -> gamma=-pi, u=1 -> gamma=pi
    gamma_theta.x = math.pi * (2.0 * uv.x - 1.0)
    #v=0 -> theta=pi, v=1 -> theta=0
    gamma_theta.y = (2.0 * alpha - math.pi) * uv.y + (math.pi - alpha)
    return gamma_theta

def UV2Angle_vectorized(uv, alpha) :
    gamma_theta = np.zeros(uv.shape, dtype=np.float32)
    #u=0 -> gamma=-pi, u=1 -> gamma=pi
    gamma_theta[:, :, 0] = math.pi * (2.0 * uv[:, :, 0] - 1.0)     
    #v=0 -> theta=pi, v=1 -> theta=0
    gamma_theta[:, :, 1] = (2.0 * alpha - math.pi) * uv[:, :, 1] + (math.pi - alpha) 
    return gamma_theta

def Angle2UV(gamma_theta_input, offset_rad) :
    uv = np.zeros((2,1), dtype=np.float32)

    gamma_theta = np.copy(gamma_theta_input)

    gamma_theta.x += offset_rad

    gamma_theta.x = Lerp(gamma_theta.x, gamma_theta.x - two_pi, gamma_theta.x > two_pi)
				
    uv.x = gamma_theta.x / two_pi

#ifdef PXR_FLIP_PORTAL_U
    uv.x = 1.0 - uv.x
#endif
    uv.y = gamma_theta.y / math.pi
           
    return uv

def Angle2UV_vectorized(gamma_theta_input, offset_rad) :
    uv = np.zeros(gamma_theta_input.shape, dtype=np.float32)

    gamma_theta = np.copy(gamma_theta_input)

    gamma_theta[:, :, 0] += offset_rad

    condition = gamma_theta[:, :, 0] > two_pi
    condition = condition.astype(np.float32)
    gamma_theta[:, :, 0] = Lerp(gamma_theta[:, :, 0], gamma_theta[:, :, 0] - two_pi, condition)
				
    uv[:, :, 0] = gamma_theta[:, :, 0] / two_pi

#ifdef PXR_FLIP_PORTAL_U
    uv[:, :, 0] = 1.0 - uv[:, :, 0]
#endif
    uv[:, :, 1] = gamma_theta[:, :, 1] / math.pi
           
    return uv

def Dir2Angle(dir_input) :
    dir = np.copy(dir_input)
    # x: [0, UNITY_TWO_PI]
    # y: [0, UNITY_PI] 
    gamma_theta = np.zeros((2,1), dtype=np.float32)
    gamma_theta.y = math.acos(dir.y)

    dir.y = 0.0
    _Normalize(dir)

    gamma_theta.x = math.acos(dir.x)
    gamma_theta.x = Lerp(gamma_theta.x, two_pi - gamma_theta.x, dir.z < 0.0)

    return 

def Dir2Angle_vectorized(dir_input) :

    dir = np.copy(dir_input)

    # x: [0, UNITY_TWO_PI]
    # y: [0, UNITY_PI] 
    shape = dir.shape

    gamma_theta = np.zeros((shape[0], shape[1], 2), dtype=np.float32)
    gamma_theta[:, :, 1] = math.acos(dir[:, :, 1])

    dir[:, :, 1] = 0.0
    _Normalize_vectorized(dir)

    gamma_theta[:, :, 0] = np.acos(dir[:, :, 0])

    condition = dir[:, :, 2] < 0.0
    condition = condition.astype(np.float32)

    gamma_theta[:, :, 0] = Lerp(gamma_theta[:, :, 0], two_pi - gamma_theta[:, :, 0], condition)

    return gamma_theta

def _Normalize_vectorized(v) :
    norm = np.linalg.norm(v, ord=None, axis=2, keepdims=False)
    v[:,:,0] /= norm
    v[:,:,1] /= norm
    v[:,:,2] /= norm

def _Normalize(v) :
    norm = np.linalg.norm(v, ord=None, axis=2, keepdims=False)
    v.x /= norm
    v.y /= norm
    v.z /= norm

def Lerp(a0, a1, x) :
    return a0 + (a1 - a0) * x

# a0 is a scalar
# a1 is H,W,3, 
# x is H,W
def Lerp_vectorized(x, a0, a1, out) :
    #out += a0 + (a1 - a0) * x
    out += np.where(x, a1, a0) # faster on cpu

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

    # print(f'{i_u.min()}|{i_u.max()}')
    # print(f'{i_v.min()}|{i_v.max()}')

    return np.stack((i_u, i_v), axis=2)

#each Ms[ci] is the transform from ci to c0
def ComputeWorldToC0(Ms_ci_c0, pin_ids) :

    num_cameras = len(Ms_ci_c0)

    Ps = [None] * num_cameras
    Vs = [None] * num_cameras

    eye = np.eye(4, dtype=np.float32)
    for i in range(num_cameras) :
        pin_id = pin_ids[i]

        Vs[i] = np.dot(Ms_ci_c0[pin_id], eye[:, 2:3])[0:3]
        Ps[i] = np.dot(Ms_ci_c0[pin_id], eye[:, 3:])[0:3]

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

class PinDatum :
    def reset(self) :
            self.pin_id = -1
            self.flip = 2

    def __init__(self) :
        self.reset()    

# this method finds out the pins and the related info

def GetPinsData(pairs_list, flip_methods_list) :
    # allowed_pins = [1,2,3,4,5]
    pin_data = {}
    for i in range(len(pairs_list)) :
        e = pairs_list[i]
        f = flip_methods_list[i]
        if e not in pin_data :
            datum = PinDatum()
            datum.pin_id = e
            datum.flip = f
            pin_data[e] = datum
            continue

    # sort the entries by pin
    sorted_items = sorted(pin_data.items())
    pin_data = dict(sorted_items)

    return pin_data

def encoding_main(daemon, process_args):

    args, bytes, delay_sec, event = process_args

    # UDP destination address and port
    url=f'udp://{args.udp_address}:{args.udp_port}?pkt_size={args.udp_packet_size}'
    print(f'{url=}')
    # on console run ffplay udp://@127.0.0.1:5000?pkt_size=1316

    input_file='/home/gimattiolo/gits/3sixd/AdobeStock_197174490_Video_4K_Preview.mp4'
    output_file='/home/gimattiolo/gits/3sixd/output.mp4'

    # original streaming working
    # process = (
    #     ffmpeg
    #     # .input(
    #     #'/home/gimattiolo/gits/3sixd/AdobeStock_197174490_Video_4K_Preview.mp4',
    #     #        stream_loop=-1
    #     #     )
    #     .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{_args.W}x{_args.H}')
    #     #.output(rtmp_url, format='flv', vcodec='libx264', acodec='aac', preset='veryfast')         
    #     .output(
    #         f'{url}',                 
    #         vcodec='libx264', # Or 'copy' if input is already H.264
    #         format='mpegts',  # Or 'h264' if streaming raw H.264
    #         preset='ultrafast', 
    #         tune='zerolatency',
    #         #sdp_file='/home/gimattiolo/gits/3sixd/my_rtp.sdp'
    #         #'x264opts': 'bframes=0:weightp=0'
    #         #keyint='30', 
    #         #scenecut='0',
    #         # format='rawvideo', 
    #         # pix_fmt='rgb24',
    #         )
    #     .run_async(pipe_stdin=True)
    # )

    stream = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{args.W}x{args.H}')

    if os.path.isfile(args.video_path) :
        split_input = stream.split()
        output_udp = split_input[0].output(
            f'{url}', 
            vcodec='libx264', 
            format='mpegts', 
            preset='ultrafast', 
            tune='zerolatency'
        )

        output_file = split_input[1].output(stream,
            args.video_path, 
            format='mp4',
            #vcodec='copy'  # Copy codecs without re-encoding
        ).overwrite_output()

        process = ffmpeg.merge_outputs(output_udp, output_file).run_async(pipe_stdin=True)
    else :
        stream = stream.output(
            f'{url}', 
            vcodec='libx264', 
            format='mpegts', 
            preset='ultrafast', 
            tune='zerolatency'
        )
        process = ffmpeg.run_async(stream, pipe_stdin=True)


    # ffmpeg_command = [
    #     'ffmpeg', '-y', 
    #     '-i', f'{input_file}',
    #     '-c:v','libx264', 
    #     '-b:v', '2M', 
    #     '-r', '30',
    #     '-c:a', 'aac', 
    #     '-b:a', '128k', f'{output_file}',
    # ]

    # ffmpeg_command = [
    #     'ffmpeg', '-y', '-i', '-', 'f=rawvideo', 'pix_fmt=bgr24', f's={Script.W}x{Script.H}', '-map', '0', '-c:v', 'copy', '-c:a', 'copy', '-f', 'tee', f'[f=mpegts]{url}|[f=mp4]{output_file}',
    # ]

    # process = subprocess.Popen(
    #         ffmpeg_command,
    #         stdin=subprocess.PIPE,
    #         # stdout=subprocess.PIPE,
    #         # stderr=subprocess.PIPE # Optional: capture stderr for error handling
        # )

    while not event.is_set() :

        try :
            #print(f'queue_size={Script.panoramas.qsize()}')
            bytes = bytes.get(block=False)
            bytes.task_done()
            process.stdin.write(bytes)
        except Exception :
            pass

        time.sleep(delay_sec)

    process.stdin.close()
    process.wait()

    daemon.Exit()

def panorama_main(daemon, process_args):
    args, cameraData_shared, panoramas, bytes, delay_sec, event = process_args

    colors_bgr_numpy = np.zeros((len(cameraData_shared), args.H, args.W, 4), dtype=np.float32)

    # zeros = np.zeros(Script.H*Script.W*3, dtype=np.float32)
    # ones = np.ones(Script.H*Script.W*3, dtype=np.float32)

    pixel_coords_numpy = np.zeros((len(cameraData_shared), args.H, args.W, 2), dtype=np.int32)
    conditions_numpy = np.zeros((len(cameraData_shared), args.H, args.W, 4), dtype=np.float32)

    for pin_id in cameraData_shared.keys() :

        pixel_coords_numpy[pin_id,:,:,0:3] = np.array(args.pixel_coords[pin_id])
        conditions_numpy[pin_id,:,:,0:3] = np.array(args.conditions[pin_id])

    accumulation_normalization_numpy = np.array(args.accumulation_normalization)

    # zeros = np.zeros(Script.H*Script.W*3, dtype=np.float32)
    # ones = np.ones(Script.H*Script.W*3, dtype=np.float32)

    panorama_bgr_numpy = np.zeros((args.H, args.W, 4), np.float32)

    if COMPUTE_MODE == 0 :

        pass

    elif COMPUTE_MODE == 1 :

        pixel_coords_cuda = cp.array(pixel_coords_numpy)
        conditions_cuda = cp.array(conditions_numpy)
        accumulation_normalization_cuda = cp.array(accumulation_normalization_numpy)
        panorama_bgr_cuda = cp.array(panorama_bgr_numpy)

    elif COMPUTE_MODE == 2 :

        compute = VulkanCompute()
        workgroup_size = 32
        shader_file = 'lerp.spv'
        compute.Setup(colors_bgr_numpy, pixel_coords_numpy, conditions_numpy, accumulation_normalization_numpy, panorama_bgr_numpy, shader_file, workgroup_size, enable_validation_layers=True)

        binding_id = 1
        binding, buffer, buffer_memory, buffer_array_size = compute.buffer_info[binding_id]
        assert binding == binding_id
        fence = None

    while not event.is_set() :

        for pin_id in cameraData_shared.keys() :
            cameraDatum = cameraData_shared[pin_id]
            colors_bgr_numpy[pin_id,:,:,0:3] = cameraDatum.frame_bgr

            # print(f'{pin_id}|{colors_bgr_numpy[pin_id,0,0,0:3]}')

        start_time = time.time()

        ### shader begins ###

        if COMPUTE_MODE == 0 :

            # make panorama
            panorama_bgr_numpy.fill(0.0) 
            for pin_id in cameraData_shared.keys() :

                pixel = pixel_coords_numpy[pin_id,:,:,:]
                color = colors_bgr_numpy[pin_id, pixel[:, :, 0], pixel[:, :, 1], :]

                Lerp_vectorized(conditions_numpy[pin_id,:,:,:], 0.0, color, panorama_bgr_numpy)

            panorama_bgr_numpy *= accumulation_normalization_numpy

        elif COMPUTE_MODE == 1 :

            colors_bgr_cuda = cp.array(colors_bgr_numpy)

            # make panorama
            panorama_bgr_cuda.fill(0.0)
            for pin_id in cameraData_shared.keys() :

                pixel = pixel_coords_cuda[pin_id,:,:,:]
                color = colors_bgr_cuda[pin_id, pixel[:, :, 0], pixel[:, :, 1], :]

                Lerp_vectorized(conditions_cuda[pin_id,:,:,:], 0.0, color, panorama_bgr_cuda)

            panorama_bgr_cuda *= accumulation_normalization_cuda

            panorama_bgr_numpy = cp.asnumpy(panorama_bgr_cuda)

        elif COMPUTE_MODE == 2 :

            # get the results into a numpy array here
            if fence :
                panorama_bgr_numpy = compute.GetBufferAsNumpy(fence, binding_id=4)
                #VulkanCompute.SaveImage(panorama_bgr_numpy, 'test.png')

        ### shader ends ###
        pan_duration_s = time.time() - start_time

        panorama_bgr = (panorama_bgr_numpy[:,:,0:3]*255).astype(np.uint8)

        bytes.put(panorama_bgr.tobytes(), block=False)

        panoramas.put(panorama_bgr, block=False)

        if COMPUTE_MODE == 2 :
            compute.InitializeBuffer(colors_bgr_numpy, buffer_memory, buffer_array_size)

            fence = compute.SubmitCommandBuffer()

        print(f'Pan:{pan_duration_s * 1000} ms|FPS:{1.0 / pan_duration_s}')

        time.sleep(delay_sec)

    if COMPUTE_MODE == 2 :
        # complete execution before exiting
        if fence :
            panorama_bgr_numpy = compute.GetBufferAsNumpy(fence, binding_id=4)

    # wait for empty queues before exiting
    while not bytes.empty() :
        time.sleep(delay_sec)

    while not panoramas.empty() :
        time.sleep(delay_sec)

    daemon.Exit()

def camera_main(daemon, process_args):
    args, cameraData_shared, delay_sec, event = process_args

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    origin = (0,150)
    fontScale              = 5
    fontColor              = (0,0,255) # red in BGR
    thickness              = 10
    lineType               = cv2.LINE_8

    while not event.is_set() :

        camerasOK = True

        # start_time = time.time()
        for pin_id in cameraData_shared.keys() :
            cameraDatum = cameraData_shared[pin_id]

            if args.benchmark :

                cameraDatum.frame_bgr = args.black_bgr_frame.copy()
                cameraDatum.frame_bgr[:,:,2] = (pin_id + 1.0) / len(cameraData_shared)

            else :

                if cameraDatum.capture.isOpened() :
                    # Capture frame-by-frame
                    ret, cameraDatum.frame_bgr = cameraDatum.capture.read()
                    cameraDatum.frame_bgr /= 255.0
                    if not ret :
                        print(f'{pin_id} not reading frames')
                        cameraDatum.frame_bgr = args.empty_frame_bgr.copy()

                    if args.show_pin :
                        cv2.putText(cameraDatum.frame_bgr, 
                            f'Pin{pin_id}', 
                            origin, 
                            font, 
                            fontScale,
                            fontColor,
                            thickness,
                            lineType,
                            bottomLeftOrigin=False)

                else :
                    camerasOK = False

            cameraData_shared[pin_id] = cameraDatum
        # print(f'{time.time() - start_time}')

        # Display the resulting frame

        if not camerasOK :
            print('Unable to open all the required cameras')

        time.sleep(delay_sec)

    daemon.Exit()

class DaemonBase :
    def reset(self) :
        self.thread = ''
        self.main = ''

    def __init__(self) :
        self.reset() 

    def __init__(self, name, main) :
        self.main = main
        self.name = name

    def Start(self) :
        print(f'Starting {self.name}...')
        self.thread.start()

    def Exit(self) :
        print(f'Exiting {self.name}...')

class DaemonThread (DaemonBase) :

    def __init__(self, name, main, args) :
        super().__init__(name, main)
        self.thread = threading.Thread(target=main, args=(self, args,), daemon=True)

class DaemonProcess (DaemonBase) :

    def __init__(self, name, main, args) :
        super().__init__(name, main)
        self.thread = multiprocessing.Process(target=main, args=(self, args,), daemon=True)

class Script :
    def main():

        parser = argparse.ArgumentParser('Panorama')
        parser.add_argument('--path', type=str, help='set the capture destination folder')
        parser.add_argument('--save_mode', type=int, default=0, help='0:append images into capture destination folder,1: delete content before starting')
        parser.add_argument('--intrinsic_path', type=str, help='set the intrinsic image capture source folder/data export folder')
        parser.add_argument('--extrinsic_path', type=str, help='set the extrinsic image capture source folder/data export folder')
        parser.add_argument('--world_space_path', type=str, default='', help='set the world space capture source folder/data export folder')
        parser.add_argument('--pairs', type=int, nargs='+', help='pairs of cameras for stereo calibration')
        parser.add_argument('--flip_methods', type=int, nargs='+', help='flip methods')
        parser.add_argument('--show_pin', action='store_true', help='show pin on each camera feed')
        parser.add_argument('--stream', action='store_true', help='stream content')
        parser.add_argument('--udp_address', type=str, default='127.0.0.1', help='udp address')
        parser.add_argument('--udp_port', type=int, default=5000, help='udp port')
        parser.add_argument('--udp_packet_size', type=int, default=1316, help='udp packet size')
        parser.add_argument('--alpha', type=float, default=0.0, help='the vertical angle in polar coordinates will be mapped to [alpha, pi - alpha]')
        parser.add_argument('--video_path', type=str, default='', help='if valid file, the stream will be encoded and saved into a video file')
        parser.add_argument('--benchmark', action='store_true', help='Enable benchmarking mode (no actual camera capture, using video dummy data instead)')
        parser.add_argument('--height', dest='H', type=int, default=1080, help='height of the output image')
        parser.add_argument('--width', dest='W', type=int, default=1920, help='width of the output image')
        parser.add_argument('--multiprocessing_start', type=str, default='spawn', help='multiprocessing start method')

        Script.args = parser.parse_args()

        # fork Available on POSIX systems.
        # forkserver on POSIX platforms which support passing file descriptors over Unix pipes such as Linux
        # spawn is the default on Windows and macOS
        multiprocessing.set_start_method(Script.args.multiprocessing_start)

        assert(len(Script.args.pairs) == len(Script.args.flip_methods))

        cyclical = Script.args.pairs[0] == Script.args.pairs[-1]

        SaveMode = Script.args.save_mode
        
        # allowed_pins = [1,2,3,4,5]
        pin_data = GetPinsData(Script.args.pairs, Script.args.flip_methods)
        
        with multiprocessing.Manager() as manager:

            #allowed_pins = None
            cameraData = ScanCameras(pin_data)

            cameraData_shared = manager.dict(cameraData)

            num_cameras = len(cameraData_shared)

            assert num_cameras >= 0

            pin_ids = list(cameraData_shared.keys())

            print(f'Compute mode:{COMPUTE_MODE}')

            print(f'Using cameras:{cameraData_shared}')

            print(f'Using path {Script.args.path}')      

            ext = '.png'

            size_default = (Script.args.H,Script.args.W) 

            # empty frame is red
            Script.args.black_bgr_frame = np.zeros((Script.args.H, Script.args.W, 3), dtype=np.float32)
            Script.args.empty_frame_bgr = Script.args.black_bgr_frame.copy() 
            Script.args.empty_frame_bgr[:, :, 2] = 1.0 

            if not os.path.exists(Script.args.path) :
                os.mkdir(Script.args.path)

            if not os.path.exists(Script.args.path) :
                print(f'Invalid path "{Script.args.path}"')
                sys.exit(1)

            pair_list = CalibrationUtilities.MakePairs(Script.args.pairs, pin_ids)        

            # pairsPerFirst = {}
            # pairsPerSecond = {}
            # for i, e in enumerate(pair_list) :
            #     first, second = e
            #     assert first not in pairsPerFirst
            #     pairsPerFirst[first] = i
            #     assert second not in pairsPerSecond
            #     pairsPerSecond[second] = i

            if not pair_list :
                return

            print('Creating capture objects...')

            if Script.args.benchmark :
                for k in range(len(pin_ids)) :
                    pin_id = pin_ids[k]
                    cameraDatum = cameraData_shared[pin_id]
                    cameraDatum.frame_bgr = Script.args.empty_frame_bgr.copy()
            else :
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

                for k in range(len(pin_ids)) :
                    pin_id = pin_ids[k]
                    cameraDatum = cameraData_shared[pin_id]
                    pinDatum = pin_data[pin_id]

                    pipeline=CalibrationUtilities.make_gstreamer_pipeline(sensor_id=cameraDatum.sensor_id, flip_method=pinDatum.flip)
                    cameraDatum.capture = cv2.VideoCapture(pipeline, api_preference)
                    print(f'sensor:{cameraDatum.sensor_id},pin:{pin_id},open:{cameraDatum.capture.isOpened()}')
                    cameraDatum.frame_bgr = Script.args.empty_frame_bgr.copy()
            # create views in the window

            panoramas = multiprocessing.Queue(maxsize=0)
            bytes = multiprocessing.Queue(maxsize=0)
            
            running = True
            if SaveMode == 0 :
                pass
            elif SaveMode == 1 :
                # delete
                shutil.rmtree(Script.args.path, ignore_errors=False, onerror=None)
                os.mkdir(Script.args.path)
            else :
                print(f'Unsupported save mode:{SaveMode}')
                exit(1)

            captureIndex = 0

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

                cameraDatum = cameraData_shared[c0]

                cameraFilename = os.path.join(Script.args.intrinsic_path, f'calibration{c0}.json')
                
                camaraCalibrationLoaded, cameraDatum.IntrinsicMatrix, cameraDatum.Distortion, cameraDatum.ReprojectionError, cameraDatum.ImageSize = Utilities.LoadCameraCalibration(cameraFilename)

                cameraCalibrationOK = cameraCalibrationOK and camaraCalibrationLoaded

                if not cameraCalibrationOK :
                    break

                cameraDatum.f_pixels, cameraDatum.h_pixels, cameraDatum.ar = Utilities.GetCalibrationParameters(camaraCalibrationLoaded, cameraDatum.IntrinsicMatrix, cameraDatum.Distortion, cameraDatum.ImageSize, pixelSize_m)

                cameraData_shared[c0] = cameraDatum

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
            for c0, c1 in pair_list :
                key = (c0, c1)
                invKey = (c1, c0)

                stereoFilename = os.path.join(Script.args.extrinsic_path, f'stereoCalibration{c0}_{c1}.json')
                stereoCalibrationLoaded, R, T, E, F, S = Utilities.LoadStereoCalibration(stereoFilename)
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
                # worldSpaceFilename = os.path.join(Script.args.world_space_path, f'worldSpaceCalibration{c0}.json')
                # worldSpaceCalibrationLoaded, R, T, S = Utilities.LoadWorldSpaceCalibration(worldSpaceFilename)
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
            Ms_c0_ci = {}
            Ms_ci_c0 = {}

            # from c0 -> ci
            M = np.identity(4)

            for k0 in range(len(Script.args.pairs)-1) :

                first = Script.args.pairs[k0]

                Ms_c0_ci[first] = M.copy()
                Ms_ci_c0[first] = CalibrationUtilities.invertExtrisics(M)

                second = Script.args.pairs[k0+1]

                M = np.dot(ExtrinsicMatrices[(first, second)], M)

            last_pin_id = Script.args.pairs[-1]

            if not cyclical :
                Ms_c0_ci[last_pin_id] = M.copy()
                Ms_ci_c0[last_pin_id] = CalibrationUtilities.invertExtrisics(M)

            M_w_c0 = ComputeWorldToC0(Ms_ci_c0, pin_ids)

            for k0 in range(num_cameras) :
                pin_id = pin_ids[k0]

                cameraDatum = cameraData_shared[pin_id]

                # we express everything in the camera c0 reference framework, i.e. camera c0 reference framework is the world reference framework
                
                #world - > ci
                E_w_ci_4x4 = np.dot(Ms_c0_ci[pin_id], M_w_c0)
                ProjectionMatrices[pin_id] = np.dot(cameraDatum.IntrinsicMatrix, E_w_ci_4x4[0:3, :])

            print('Running...')

            window_visible = True

            i_uv = MakeUV(size_default)
            gammaTheta = UV2Angle_vectorized(i_uv, Script.args.alpha)
            ray_inW = Angle2Dir_vectorized(gammaTheta)
            ray_inW = ray_inW.reshape((3,-1))    

            Script.args.pixel_coords= {}
            Script.args.conditions = {}

            num_accumulations = np.zeros((Script.args.H,Script.args.W), dtype=np.float32)

            for pin_id, cameraDatum in cameraData_shared.items() :

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
                ps = np.reshape(ps[0:2, :], (2, Script.args.H, Script.args.W))
                #2,H,W -> H,W,2
                ps = np.transpose(ps, (1, 2, 0)).astype(np.int32)

                pixel_x = ps[:, :, 0]
                pixel_y = ps[:, :, 1]

                mask_x = (0 <= pixel_x) & (pixel_x < Script.args.W)
                mask_y = (0 <= pixel_y) & (pixel_y < Script.args.H)
                condition = mask_x & mask_y

                num_accumulations += condition

                # Combine into a single array of pixel coordinates
                # y is row, x is column
                pixel = np.stack((pixel_y, pixel_x), axis=2).astype(np.int32)

                condition_int = condition.astype(np.int32)

                # we do this so when sampling we don't have invalid pixel coordinates
                pixel[:,:,0] *= condition_int
                pixel[:,:,1] *= condition_int

                # replicate along rgb
                Script.args.conditions[pin_id] = np.tile(condition[:, :, np.newaxis], (1, 1, 3)).astype(np.float32)
                
                Script.args.pixel_coords[pin_id] = pixel

            #print(f'{num_accumulations.min()}|{num_accumulations.max()}')

            Script.args.accumulation_normalization = 1.0 / np.maximum(1.0, num_accumulations)
            # replicate along rgb
            Script.args.accumulation_normalization = np.tile(Script.args.accumulation_normalization[:, :, np.newaxis], (1, 1, 4)).astype(np.float32)

            ray_inW = np.reshape(ray_inW, (3, Script.args.H, Script.args.W))
            ray_inW = np.transpose(ray_inW, (1, 2, 0))

            output_id = 0

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            origin = (800,500)
            fontScale              = 5
            fontColor              = (255,0,0) # red in BGR
            thickness              = 10
            lineType               = cv2.LINE_8

            delta_time_sec_30fps = 1.0 / 30.0
            delta_time_sec_60fps = 1.0 / 60.0
            delta_time_sec_120fps = 1.0 / 120.0
            zero_delta_time_sec  = 1.0 / 1000.0

            # Create threads
            Script.daemons = []

            if Script.args.stream :
                encoding_stop_event = multiprocessing.Event()
                encoding_daemon = DaemonProcess('EncodingDaemon', encoding_main, (Script.args, bytes, delta_time_sec_60fps, encoding_stop_event))
                Script.daemons.append((encoding_daemon, encoding_stop_event))

            camera_stop_event = multiprocessing.Event()
            camera_daemon = DaemonProcess('CameraDaemon', camera_main, (Script.args, cameraData_shared, delta_time_sec_60fps, camera_stop_event))
            Script.daemons.append((camera_daemon, camera_stop_event))

            panorama_stop_event = multiprocessing.Event()
            panorama_daemon = DaemonProcess('PanoramaDaemon', panorama_main, (Script.args, cameraData_shared, panoramas, bytes, delta_time_sec_60fps, panorama_stop_event))
            Script.daemons.append((panorama_daemon, panorama_stop_event))

            # Start threads
            for daemon, event in Script.daemons :  
                daemon.Start()

            #in msec
            waitKeyPeriod_msec = int(delta_time_sec_60fps * 1000.0)

            panorama_bgr = Script.args.black_bgr_frame.copy()
            panorama_bgr = (panorama_bgr*255).astype(np.uint8)

            while running :

                key = cv2.waitKey(waitKeyPeriod_msec)
                # if cv2.waitKey(waitKeyPeriod) & 0xFF == ord('q') :

                if key == ord('q') :#or not window_visible:
                    running = False
                    
                # if cv2.getWindowProperty('foo', cv2.WND_PROP_VISIBLE):
                #     window_visible = True
                # else :
                #     window_visible = False

                # print(window_visible)


                #start_time = time.time()
                # save screenshot

                try :
                    #print(f'queue_size={Script.args.panoramas.qsize()}')
                    panorama_bgr = panoramas.get(block=False)
                    panorama_bgr.task_done()
                except Exception as e :
                    pass

                if key == ord('s') :
                    filename = os.path.join(Script.args.path, f'panorama_{output_id}.png')

                    ret = cv2.imwrite(filename=filename, img=panorama_bgr)

                    if ret :
                        print(f'Screenshot saved:{filename}')
                        output_id += 1
                    else : 
                        print(f'Unable to save screenshot:{filename}')
        
                cv2.imshow(window_name, panorama_bgr)

                #print(f'{time.time() - start_time}')

            for daemon, event in Script.daemons :
                event.set()

            time.sleep(1.0)

            for daemon, event in Script.daemons :
                while(daemon.thread.is_alive()) :
                    while not bytes.empty() :
                        bytes.get()
                    while not panoramas.empty() :
                        panoramas.get()
                    # print(multiprocessing.active_children())  # Lists active child processes
                    # print(f'Joining {daemon}...')
                    daemon.thread.join(timeout=1)

            # When everything done, release the captures
            for pin_id in cameraData_shared.keys() :
                cameraData_shared[pin_id].release()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    Script.main()
