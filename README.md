<Dev environment>
We used UV (https://docs.astral.sh/uv/) for creating python enviroments to run the script with their dependencies. You should use it to easily setup the environment for running the scripts. 
The dependencies of the project are listed in the pyproject.toml and uv.lock files.
We depend on cv2(opencv), numpy, cupy and ffmpeg-python currently.
A custom build of opencv from source was required to compile opencv with GStreamer capabilities. On the linux distro running on device, GStreamer is the only lib able to detect and provide the camera frames to opencv.
A python wheel was created to package the python wrappers in order to install the package within UV.
For development/debugging we used vscode - https://code.visualstudio.com/. A .vscode folder is included in this repo along with the launch.json file.
To open vscode just type code in a terminal.

For info on opencv calibration look at 
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
http://rabinf24.uco.es/fsiv/opencv-doc-3.4.4/html/db/d58/group__calib3d__fisheye.html
https://plaut.github.io/fisheye_tutorial/

The code has been tested on jetson agx orin 64GB with 6 cameras:
camera arducam imx 519
https://www.arducam.com/imx519-autofocus-camera-module-for-raspberry-pi-arducam-b0371.html

camera chip arducam 12 MP multicamera for jetson agx orin

The fish eye cameras are 
Model No: M25156H18
Arducam 180 Degree Fisheye 1/2.3" M12 Lens with Lens Adapter for Raspberry Pi High Quality Camera
https://www.arducam.com/arducam-180-degree-fisheye-1-2-3-m12-mount-with-lens-adapter-for-raspberry-pi-high-quality-camera.html

<Scripts>
The main scripts of this repo are cameraCalibrationCapture.py, cameraCalibration.py and panorama.py.

* cameraCalibrationCapture.py

Use this script to capture a collection of pictures showing a calibration checkerboard like this one https://github.com/opencv/opencv/blob/4.x/doc/pattern.png.
There are few arguments that can be passed to the script. --path defines where the images will be saved on disk. 
The auto flag automatically captures images at interval when detecting the patterns, the pattern_size, representing the number of rows and columns on the checkerboard used for calibration and allowed pins which allows the script to go through all the cameras and pairs of cameras based on the sequence.

The ui shows 2 views from 2 cameras. It cyrcles through the first camera to the last, then it goes pair by pair based on argument --intrinsics_captures and --extrinsics_captures. The single view captures are neede for intrinsics or camera matrix calibration. The 2 views captures are needed for the extrisics calibration that figures out the relative rotation and trnslation of adjecent cameras.
When the pattern is detected the view shows it overlaid to the camera stream for a short amoutn of time
The text shows the time till next detection (configurable), the current number of pictures taken vs the total one for that apture type(single camera or pair). Also the pin on the chip for the cameras visualized is displayed. We use the pin as an id for the cameras so we can identify them reliably. 

To launch, in visual studio code you can run the configuration "calibration capture" in .vscode/launch.json.

* cameraCalibrationVerify.py

This script takes a folder of images in input and run the chess board corner detection on them and save the decorated images into an output path.
It is a good way to debug bad calibration that often come from too few good images of the calibration board.  An image is bad when the board is not fully in view or the board has reflections or shadows on it that cause the corners to be detected incorrectly.
In the image saved out you should see segments of different colors going through all the rows of the checkerboard in the image.
If they are not correclty aligned with the board in the image the calibration quality will be affected.
To launch, in visual studio code you can run the configuration "calibration verification" in .vscode/launch.json.

* cameraCalibration.py

This script takes as input the images recorded using the cameraCalibrationCapture.py script above and runs internal routines to optimize intrinsic and extrinsic parameters for each camera 
and for each feasible pair of cameras.
Multiple paths can be specified for outputting the calibration data after the process is completed. The data are saved as json files. 
Again pattern size and length must be provided as input, along with the camera pins and the feasible pairs. during calibration the scripts saves out to the output folder 
To launch, in visual studio code you can run the configuration "calibration" in .vscode/launch.json.

* panorama.py

This is the script that performs the stitching of all the views from the cameras, exploiting the calibration data.
It accepts paths specifying where the calibration files are and the a list of camera pairs to be used, that should match what was used for camera calibration.
A flag can be used to stream the frames after they have been stitched to a udp stream baed on input ip and port. If the output path is valid a video file gets encoded in real time and saved in the folder.
You can use 'q' to quit the application, and 's' to save a screenshot.
When passing argument --benchmark the panorama runs in benchamrk mode without camera. it doess need to use cmaera calibration files for the reprojection. 
Using argument --compute_mode one can switch between running the reporjection using 0:cpu (using numpy),1:cuda (using cupy),2:vulkan
To launch, in visual studio code you can run the configuration "panorama" in .vscode/launch.json.


The folder good_calibrations include the best calibation settings recovered so far.
