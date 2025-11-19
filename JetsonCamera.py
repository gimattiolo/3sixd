# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import time
try:
    from  Queue import  Queue
except ModuleNotFoundError:
    from  queue import  Queue

import  threading
import signal
import sys


# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)


# std::string gstreamer_pipeline1 (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) :
#     return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
#            std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
#            "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
#            std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";


def gstreamer_pipeline(
    source_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    


    # return (

    #     f'nvarguscamerasrc sensor-id={source_id} ! '
    #     # f'"video/x-raw(memory:NVMM),width=(int){capture_width},height=(int){capture_height},format=(string)NV12,framerate=(fraction){framerate}/1" ! '
    #     f'"video/x-raw(memory:NVMM),width=(int){capture_width},height=(int){capture_height},framerate=(fraction){framerate}/1" ! '
    #     f'nvvidconv flip-method={flip_method} ! '
    #     #f'nvegltransform ! '
    #     f'"video/x-raw,width=(int){display_width},height=(int){display_height},format=(string)BGRx" ! '
    #     f'videoconvert ! '
    #     f'"video/x-raw,format=(string)BGR" ! '
    #     f'queue ! '
    #     f'appsink'
    # )

    return 'nvarguscamerasrc sensor-id=2 ! "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1" ! nvvidconv flip-method=2 ! "video/x-raw, format=(string)BGRx" ! videoconvert ! "video/x-raw, format=(string)BGR" ! appsink'

class FrameReader(threading.Thread):
    queues = []
    _running = True
    camera = None
    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera
 
    def run(self):
        while self._running:
            _, frame = self.camera.read()
            while self.queues:
                queue = self.queues.pop()
                queue.put(frame)
    
    def addQueue(self, queue):
        self.queues.append(queue)

    def getFrame(self, timeout = None):
        queue = Queue(1)
        self.addQueue(queue)
        return queue.get(timeout = timeout)

    def stop(self):
        self._running = False

class Previewer(threading.Thread):
    window_name = "Arducam"
    _running = True
    camera = None
    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera
    
    def run(self):
        self._running = True
        while self._running:
            cv2.imshow(self.window_name, self.camera.getFrame(2000))
            keyCode = cv2.waitKey(16) & 0xFF
        cv2.destroyWindow(self.window_name)

    def start_preview(self):
        self.start()
    def stop_preview(self):
        self._running = False

class Camera(object):
    frame_reader = None
    cap = None
    previewer = None

    def __init__(self, width=640, height=360):
        self.open_camera(width, height)

    def open_camera(self, width=640, height=360):
        	
        #print(f'{cv2.__version__=}')

        print(f'{cv2.getBuildInformation()}=')

        pipeline = gstreamer_pipeline(source_id=2, flip_method=0)
        print(f'{pipeline=}')
        try :
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        except Exception as e:
            print(e)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")
        if self.frame_reader == None:
            self.frame_reader = FrameReader(self.cap, "")
            self.frame_reader.daemon = True
            self.frame_reader.start()
        self.previewer = Previewer(self.frame_reader, "")

    def getFrame(self, timeout = None):
        return self.frame_reader.getFrame(timeout)

    def start_preview(self):
        self.previewer.daemon = True
        self.previewer.start_preview()

    def stop_preview(self):
        self.previewer.stop_preview()
        self.previewer.join()
    
    def close(self):
        self.frame_reader.stop()
        self.cap.release()

if __name__ == "__main__":
    camera = Camera()
    camera.start_preview()
    time.sleep(10)
    camera.stop_preview()
    camera.close()
