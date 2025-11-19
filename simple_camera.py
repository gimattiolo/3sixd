# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import time

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

# def gstreamer_pipeline(
#     sensor_id=0,
#     capture_width=1920,
#     capture_height=1080,
#     display_width=960,
#     display_height=540,
#     framerate=30,
#     flip_method=0,
# ):

#     return (
#         "nvarguscamerasrc sensor-id=%d ! "
#         "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#         "videoconvert ! "
#         "video/x-raw, format=(string)BGR ! "
#         "appsink"
#         % (
#             sensor_id,
#             capture_width,
#             capture_height,
#             framerate,
#             flip_method,
#             display_width,
#             display_height,
#         )
#     )

#return 'gst-launch-1.0 nvarguscamerasrc sensor-id=2 ! "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1" ! nvvidconv flip-method=0 ! nvegltransform ! nveglglessink -e'


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
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


def show_camera():
    window_title = "CSI Camera"

    sensor_id = 2

    # camera_id = sensor_id
    # api_preference=cv2.CAP_V4L2

    # camera_id = f'/dev/video{sensor_id}'
    # api_preference=cv2.CAP_V4L2

    camera_id = gstreamer_pipeline(sensor_id=sensor_id, flip_method=0)
    api_preference=cv2.CAP_GSTREAMER

    # camera_id=f'v4l2src device=/dev/video{sensor_id} ! video/x-raw,width=1920,height=1080 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
    # api_preference=cv2.CAP_GSTREAMER

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(f'{sensor_id=}')
    print(f'{camera_id=}')
    print(f'{api_preference=}')
    video_capture = cv2.VideoCapture(camera_id, api_preference)
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # video_capture.set(cv2.CAP_PROP_FPS, 30)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            frame_counter = -1
            
            
            while True:
                frame_counter += 1
                frame_available, frame = video_capture.read()
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                
                print(f'[{frame_counter}]{frame_available=}')
                if not frame_available : 
                    time.sleep(1)
                    continue
                
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                #Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break

                #time.sleep(1)
        except Exception as e :
            print(e)
        finally:
            video_capture.release()
            #cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    show_camera()
