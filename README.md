<Dev environment>
We used UV (https://docs.astral.sh/uv/) for creating python enviroments to run the script with their dependencies. You should use it to easily setup the environment for running th scripts. 
The dependencies of the project are listed in the pyproject.toml and uv.lock files.
We depend on cv2(opencv), numpy, cupy and ffmpeg-python currently.
A custom build of opencv from source was required to compile opencv with GStreamer capabilities. On the linux distro running on device, GStreamer is the only lib able to detec and provide the camera frames to opencv.
A python wheel was created to pacakge the python wrappers in order to install the package within UV.
For development/debugging we used vscode - https://code.visualstudio.com/. A .vscode folder is included in this repo along with the launch.json file.


<Scripts>
The main scripts of this repo are cameraCalibrationCapture.py, cameraCalibration.py and panorama.py.

cameraCalibrationCapture.py

Use this script to capture a collection of pictures showing a calibration checkerboard like this one https://github.com/opencv/opencv/blob/4.x/doc/pattern.png.
There are few parameters that can be passed tho the script. Of course path defines wherer th images will be saved on disk. 
The auto flag automatically captures images at interval when detecting the patterns, the pattern_size, reqpresenting the number of rows and columns
on the checkerboard used for calibration and allowed pins whihc allows the script to go through all the cameras and pairs of cameras based on the sequence.

cameraCalibration.py

This script takes as input the images recorded using the script above and runs internal routines to optimize intrinsic and extrinsic parameters for each camera 
and for each feasible pair of cameras.
Multiple paths can be specified for outputting the calibration data after the process is completed. The data are saved as json files. 
Again pattern size and length must be provided as input, along with the camera pins and the feasible pairs. during calibration the scripts saves out to the output folder 
also the images used for calibration with the detected checkerboard pattern overalyed on them.
This is very useful to debug image artifatcs such as shadows and reflection whihc might cause the apttern to be detected inaccurately,
hence affecting the quality of the calibration.

panorama.py

This is the script that performe the stitching of all the views from the cameras, exploiting the calibration data.
It accepts apths specifying where the calibration files are and the a list of camera pairs to be used, that should match what was used for camera calibration.
A flag can be used to stream the fraems after they have been stitched to a udp stream baed on input ip and port. If the output path is valid a video file gets encoded in real time and saved in the folder.
You can use 'q' to quit the application, and 's' to save a screenshot.

The folder good_calibrations include the best calibation settings recovered so far.
