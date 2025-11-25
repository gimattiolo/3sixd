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

if __name__ == "__main__":


    radius = 1.0

    Intrinsics = np.zeros((3,4), dtype=np.float32)

    num_cameras = 6

    step = 2 * np.pi / num_cameras

    ExtrinsicMatrices = {}
    ExtrinsicMatrices_noisy = {}
    M_w_c = [None] * num_cameras
    M_c_w = [None] * num_cameras

    angle = 0.0
    for i in range(0, num_cameras) :

        M_w_c[i] = np.eye(4, np.float32)
        M_c_w[i] = np.eye(4, np.float32)
        
        x = math.cos(angle)
        z = math.sin(angle)
        forward = np.array([x, 0.0, z], dtype=np.float32)
        
        t_c_w = radius * forward

        down = np.array([0,1,0], dtype=np.float32)

        right = np.cros(forward, down)

        R_c_w = np.eye(3, np.float32)

        #W - > c
        M_c_w[i][0:3, 0:1] = right
        M_c_w[i][0:3, 1:2] = down
        M_c_w[i][0:3, 2:3] = forward
        M_w_c[i][0:3, 3:4] = t_c_w

        R_w_c = np.transpose(R_c_w)
        t_w_c = -np.dot(R_w_c, t_c_w) 

        M_w_c[i][0:3, 0:3] = R_w_c
        M_w_c[i][0:3, 3:4] = t_w_c


    for i in range(0, num_cameras) :
        j = (i+1)% num_cameras
        key = (i, j)
        ExtrinsicMatrices[key] = np.dot(M_w_c[j], M_c_w[i])
        ExtrinsicMatrices_noisy[key] = 











