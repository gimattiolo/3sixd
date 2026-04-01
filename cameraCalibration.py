import numpy as np
import cv2
import argparse
import os
import sys
import time
import shutil

import CalibrationUtilities
import Utilities

def ComputeImagePointCorners(image, image_path, patternSize, searchSize, zeroZoneSize):
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    ret, corners = cv2.findChessboardCorners(gray, patternSize, None)
    if not ret :
        print(f'Unable to find chessboard corners')
        return False, None
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    corners_subPix = cv2.cornerSubPix(gray, corners, searchSize, zeroZoneSize, criteria)
    return True, corners_subPix

# see https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
# see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# see https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
def CameraCalibration(fisheye, patternSize, searchSize, zeroZoneSize, imageFiles, useIntrinsicsGuess, intrinsicMatrix, distortion) :
    # Array to store image points from all the images.
    imagePoints = [] # 2d points in image plane.

    image_size = None

    for i in range(0, len(imageFiles)) :
        image_path = imageFiles[i]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None :
            print(f'Unable to load {image_path}')
            continue

        if image_size is None :
            # we store channels, width and height
            image_size = image.shape[::-1]

        print(f'Processing image {i} {image_path}')
        success, corners_subPix = ComputeImagePointCorners(image, image_path, patternSize, searchSize, zeroZoneSize)
        if not success:
            continue
        imagePoints.append(corners_subPix)

    # Array to store object points from all the images.
    imageObjectPoints = [] # 3d points in real world space
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[ 0 : patternSize[0], 0 : patternSize[1] ].T.reshape(-1, 2)
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    for i in range(0, len(imagePoints)) :
        imageObjectPoints.append(objp)

    objectPoints = np.array(imageObjectPoints, dtype=np.float32)
    objectPoints = [np.expand_dims(points, -2) for points in objectPoints]

    if len(imagePoints) <= 0 :
        print(f'No image points found')
        return

    numImages = max(1, len(imagePoints))
       
    #W, H
    size = (image_size[1], image_size[2])

    print(f'Guess flag enabled : {useIntrinsicsGuess}')
       
    print(f'Running calibration routine. This might take a while and be unresponsive, depending on the number of input images : {numImages}')
    
    flags = 0

    if useIntrinsicsGuess :   
        flags += cv2.CALIB_USE_INTRINSIC_GUESS 
    
    if fisheye :

        # some default values that seem to work
        fx = fy = 0.5 * size[1]
        cx = 0.5 * size[0]
        cy = 0.5 * size[1]
        #intrinsicMatrix = np.eye(3)
        intrinsicMatrix = np.array(
            [
                [fx, 0.0, cx], 
                [0.0, fy, cy], 
                [0.0, 0.0, 1.0],
            ]
        )
        distortion = np.zeros((4, 1))


        flags += cv2.CALIB_USE_INTRINSIC_GUESS 
        #flags += cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC #seem to cause larger erros
        flags += cv2.fisheye.CALIB_CHECK_COND
        flags += cv2.fisheye.CALIB_FIX_SKEW

        N_OK = len(objectPoints)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        error, intrinsicMatrix, distortion, rvecs, tvecs = cv2.fisheye.calibrate(objectPoints, imagePoints, size, intrinsicMatrix, distortion, rvecs, tvecs, flags, criteria)

    else :

        error, intrinsicMatrix, distortion, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, size, intrinsicMatrix, distortion, None, None, flags, criteria)

    mean_error = 0.0
    for i in range(len(imagePoints)):
        imagePoints2, _ = cv2.projectPoints(objectPoints[i], rvecs[i], tvecs[i], intrinsicMatrix, distortion)
        numImagePoints = max(1, len(imagePoints2))
        error = cv2.norm(imagePoints[i], imagePoints2, cv2.NORM_L2) / numImagePoints
        mean_error += error

    mean_error /= numImages
    print(f"Mean reprojection error : {mean_error}")
    
    return error, intrinsicMatrix, distortion, mean_error, imagePoints, image_size

def StereoCalibration(objectPoints, imagePoints1, imagePoints2, IntrinsicMatrix1, Distortion1, IntrinsicMatrix2, Distortion2, imageSize) :

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    flags = cv2.CALIB_FIX_INTRINSIC
    
    # Output rotation matrix R together with the translation vector T brings points given in the first camera's coordinate system to points in the second camera's coordinate system. In more technical terms, the tuple of R and T performs a change of basis from the first camera's coordinate system to the second camera's coordinate system. Due to its duality, this tuple is equivalent to the position of the first camera with respect to the second camera coordinate system.
    # R - Rotation Matrix between first and second camera coordinate systems.
    # T - Translation vector between the coordinate systems of the cameras. In pixels
    # E - Essential matrix.
    # F - Fundamental matrix.
    
    size = (imageSize[1], imageSize[2])
    
    error, IntrinsicMatrix1, Distortion1, IntrinsicMatrix2, Distortion2, R, T, E, F = cv2.stereoCalibrate(objectPoints, imagePoints1, imagePoints2, IntrinsicMatrix1, Distortion1, IntrinsicMatrix2, Distortion2, size, flags, criteria)

    return error, R, T, E, F

def SaveJsonContent(jsonContent, filename) :
    fileObj = open(filename, 'w+')
    print(jsonContent)
    if fileObj is None :
        print(f'Unable to write calibration data into {filename}')
    else :
        print(f'Saving calibration json into {filename}')
        fileObj.writelines(jsonContent)
    fileObj.close()

def LoadJsonContent(filename) :
    fileObj = open(filename, 'r')
    jsonContent = None
    if fileObj is None :
        print(f'Unable to read calibration data from {filename}')
    else :
        print(f'Reading calibration json from {filename}')
        jsonContent = fileObj.read()
        print(jsonContent)
    fileObj.close()
    return jsonContent

def PrepareImages(calibrationPaths, fileIndices, debugMaxNumCameraImage):
    numCameras = len(fileIndices)
    imageFilesPerCamera = {}
    # sizesPerCamera = {}

    for i in range(0, numCameras) :
        imageFilesPerCamera[ fileIndices[i] ] = []
        # sizesPerCamera[ fileIndices[i] ] = []

    # sort the entries by pin
    imageFilesPerCamera = dict(sorted(imageFilesPerCamera.items()))
    #sizesPerCamera = dict(sorted(sizesPerCamera.items()))

    for path in calibrationPaths :
        fileList = os.listdir(path)
        for i in range(0, len(fileList)):
            filename = fileList[i]
        
            name, file_extension = os.path.splitext(filename)

            if file_extension.lower() == '.json' :
                continue

            pin_id = CalibrationUtilities.GetCameraIndex(filename)

            if pin_id == -1 :
                print(f'Unable to extract camera index from string {filename}')
                continue

            if not (pin_id in fileIndices):
                continue

            if len(imageFilesPerCamera[pin_id]) >= debugMaxNumCameraImage :
                continue
            
            fullFilename = os.path.join(path, filename)
        
            # image = cv2.imread(fullFilename, cv2.IMREAD_COLOR)
            # if image is None :
            #     print(f'Unable to load {fullFilename}')
            #     continue
                
            # we store channels, width and height
            # size = image.shape[::-1]
            # sizesPerCamera[pin_id].append(size)

            print (f'Loading image #{i} {fullFilename} for camera {pin_id}')
            imageFilesPerCamera[pin_id].append(fullFilename)

    # for pin_id in sizesPerCamera.keys() :
    #     for i in range(1, len(sizesPerCamera[pin_id])) :
    #         if sizesPerCamera[pin_id][i] != sizesPerCamera[pin_id][i - 1] :
    #             print(f'Camera {pin_id} : image {i} size {sizesPerCamera[pin_id][i]} is different from image {i - 1} size {sizesPerCamera[pin_id][i - 1]}')
    #             return

    for pin_id in imageFilesPerCamera.keys() :
        if len(imageFilesPerCamera[pin_id]) == 0 :
            print(f'No image found in {calibrationPaths} for camera {pin_id}')

    return imageFilesPerCamera#, sizesPerCamera

def main():
    parser = argparse.ArgumentParser('Create calibration data from images')
    parser.add_argument('--intrinsic', action="store_true", help='create intrinsic matrix json file (requires --intrinsicpath)')
    parser.add_argument('--intrinsic_paths', type=str, nargs='+', help='set the intrinsic image capture source folder/data export folder')
    parser.add_argument('--extrinsic', action="store_true", help='create extrinsic matrix json file (requires both --intrinsicpath and --extrinsicpath)')
    parser.add_argument('--extrinsic_paths', type=str, nargs='+', help='set the extrinsic image capture source folder/data export folder')
    parser.add_argument('--world_space', action="store_true", help='generate a worldspace transform for each camera')
    parser.add_argument('--world_space_path', type=str, help='set the world space image capture source folder/data export folder')
    parser.add_argument('--file_indices', type=int, nargs='+', help='file camera indices to process')
    parser.add_argument('--pattern_size', type=int, nargs=2, help='2D size of checkerboard pattern to detect')
    parser.add_argument('--pattern_side_length', type=float, default=0.0, help='length of checkerboard square side in millimeters')
    parser.add_argument('--search_size', type=int, nargs=2, help='search window half-size for finding the checkerboard')
    parser.add_argument('--zero_zone_size', type=int, nargs=2, help='search zone dead region half-size that is ignored when looking for checkerboard gradients')
    parser.add_argument('--max_images', type=int, default=-1, help='maximum number of images to load')
    parser.add_argument('--pairs', type=int, nargs='+', help='pairs of cameras for stereo calibration')
    parser.add_argument('--use_intrinsics_guess', action="store_true", help='use guess for intrinsics')
    parser.add_argument('--output_path', type=str, help='output folder for calibration files')
    parser.add_argument('--fisheye', action="store_true", help='true if using fish eye lenses)')

    args = parser.parse_args()

    modeTotals = (1 if args.intrinsic else 0) + (1 if args.extrinsic else 0) + (1 if args.world_space else 0)
    if modeTotals != 1:
        print ('You must run in either --intrinsic, --extrinsic, or --worldspace mode, and not multiple modes at once.')
        sys.exit(1)

    if args.intrinsic and not args.intrinsic_paths:
        print ('You must set the --intrinsic_path to run in --intrinsic mode')
        sys.exit(1)

    if args.extrinsic and not (args.intrinsic_paths and args.extrinsic_paths):
        print ('You must set the --intrinsic_path and --extrinsic_path to run in --extrinsic mode')
        sys.exit(1)

    if args.world_space and not (args.intrinsic_paths and args.world_space_paths):
        print ('You must set the --intrinsicpath and --worldspacepath to run in --worldspace mode')
        sys.exit(1)

    genericPath = CalibrationUtilities.GetCalibrationPath()
    intrinsic_paths = [ genericPath ]
    if args.intrinsic_paths:
        intrinsic_paths = args.intrinsic_paths
        print(f'Using intrinsic image/data path {intrinsic_paths}')

    extrinsic_paths = [ genericPath ]
    if args.extrinsic_paths:
        extrinsic_paths = args.extrinsic_paths
        print(f'Using extrinsic image/data path {extrinsic_paths}')

    world_space_path = genericPath
    if args.world_space_path:
        world_space_path = args.world_space_path
        print(f'Using worldspace image/data path {world_space_path}')

    patternSize = CalibrationUtilities.GetPatternSize()
    if args.pattern_size:
        patternSize = tuple(args.pattern_size)
    print(f'Using pattern size {patternSize}')      
    
    searchSize = CalibrationUtilities.GetSearchSize()
    if args.search_size:
        searchSize = tuple(args.search_size)
    print(f'Using search size {searchSize}')      
    
    zeroZoneSize = CalibrationUtilities.GetZeroZoneSize()
    if args.zero_zone_size:
        zeroZoneSize = tuple(args.zero_zone_size)
    print(f'Using zero zone size {zeroZoneSize}')      

    fileIndices = args.file_indices
    numCameras = len(fileIndices)

    debugMaxNumCameraImage = np.inf
    if args.max_images >= 0:
        debugMaxNumCameraImage = args.max_images
    
    print(f'Debug Max Num Camera Images {debugMaxNumCameraImage}')

    if args.intrinsic and intrinsic_paths :
        for path in intrinsic_paths :
            if not os.path.exists(path) :
                print(f'Invalid intrinsic path {intrinsic_paths}')
                sys.exit(1)

    if args.extrinsic and extrinsic_paths :
        for path in extrinsic_paths :
            if not os.path.exists(path) :
                print(f'Invalid extrinsic path {path}')
                sys.exit(1)

    if args.world_space and world_space_path and not os.path.exists(world_space_path) :
        print(f'Invalid worldspace path {world_space_path}')
        sys.exit(1)

    sideLength = CalibrationUtilities.GetCalibrationPatternSquareSideLengthInMeters()
    if args.pattern_side_length > 0.0:
        sideLength = args.pattern_side_length
    patternSideLength = np.array( [ sideLength ] )

    if args.intrinsic: 
    
        startTime = time.time()
    
        imageFilesPerCamera = PrepareImages(intrinsic_paths, fileIndices, debugMaxNumCameraImage)
        intrinsicMatrices = {}
        distortions = {}
        mean_errors = {}

        imagePoints = {}
        for pin_id in imageFilesPerCamera.keys() :
            intrinsicMatrices[pin_id] = []
            distortions[pin_id] = []
            mean_errors[pin_id] = []
            imagePoints[pin_id] = []
            print(f'Calibrating camera {pin_id}')
            
            cameraFilename = os.path.join(args.output_path, f'calibration{pin_id}.json')

            camaraCalibrationLoaded, intrinsicMatrices[pin_id], distortions[pin_id], e, image_size, fisheye = Utilities.LoadCameraCalibration(cameraFilename)

            useIntrinsicsGuess = args.use_intrinsics_guess and camaraCalibrationLoaded

            error, intrinsicMatrices[pin_id], distortions[pin_id], mean_errors[pin_id], imagePoints[pin_id], image_size = CameraCalibration(args.fisheye, patternSize, searchSize, zeroZoneSize, imageFilesPerCamera[pin_id], useIntrinsicsGuess, intrinsicMatrices[pin_id], distortions[pin_id])
            jsonContent = CalibrationUtilities.CameraCalibrationToJson(args.fisheye, intrinsicMatrices[pin_id], distortions[pin_id], mean_errors[pin_id], image_size) 

            filename = os.path.join(args.output_path, f'calibration{pin_id}.json')
            SaveJsonContent(jsonContent, filename)

        print(f'Completed in {(time.time() - startTime) / 60.0} minutes')

    elif args.extrinsic:
        startTime = time.time()
    
        imageFilesPerCamera = PrepareImages(extrinsic_paths, fileIndices, debugMaxNumCameraImage)

        pin_ids = list(imageFilesPerCamera.keys())

        pairs = CalibrationUtilities.MakePairs(args.pairs, pin_ids)


        # load intrinsic data from disk for each camera
        intrinsicMatrices = {}
        distortions = {}
        rois = {}
        mapxs = {}
        mapys = {}
        for k0 in range(numCameras) :
            c = pin_ids[k0]
            filename = os.path.join(args.output_path, f'calibration{c}.json')
            jsonContent = LoadJsonContent(filename)
            intrinsicMatrix, distortion, reprojectionError, imageSize, fisheye = CalibrationUtilities.JsonToCameraCalibration(jsonContent)
            assert fisheye == args.fisheye

            h,  w = imageSize
            new_camera_matrix, roi, mapx, mapy = CalibrationUtilities.ComputeUndistortRectifyMap(args.fisheye, (w, h), (w, h), intrinsicMatrix, distortion, blaance=1.0, image_size2=None, image_size3=None)

            intrinsicMatrices[c] = new_camera_matrix
            distortions[c] = distortion
            rois[c] = roi
            mapxs[c] = mapx
            mapys[c] = mapy

        imagePoints = {}

        validImagesPerPair = {}
        validPointsPerPair = {}

        image_size = None

        for k0 in range(numCameras) :
            c = pin_ids[k0]
            imagePoints[c] = []
            # compute image points
            image_paths = imageFilesPerCamera[c]
            
            for i in range(0, len(image_paths)):

                image_path = image_paths[i]
                print(f'Loading {image_path}')
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                if image is None :
                    print(f'Unable to load {image_path}')
                    continue

                if image_size is None :
                    # we store channels, width and height
                    image_size = image.shape[::-1]

                image = CalibrationUtilities.UndistortImage(image, mapxs[c], mapys[c])

                #image = CalibrationUtilities.CropUndistortedImage(image, rois[c])

                success, corners_subPix = ComputeImagePointCorners(image, image_path, patternSize, searchSize, zeroZoneSize)
                if success:
                    imagePoints[c].append(corners_subPix)
                else:
                    print(f'No image points found on image {os.path.basename(image_path)}')
                    sys.exit(1)

        for c0, c1 in pairs :
            validImagesPerPair[(c0, c1)] = ([], [])
            validPointsPerPair[(c0, c1)] = ([], [])

        # filter out images without matching point pairs
        for c0, c1 in pairs :

            print(f'Collecting necessary images for pair {c0} {c1}')
            for i0 in range(0, len(imageFilesPerCamera[c0])) :
                image_path0 = imageFilesPerCamera[c0][i0]
                name0 = os.path.basename(image_path0)
                points0 = imagePoints[c0][i0]
                directory0 = os.path.dirname(image_path0)

                captureIndex0 = CalibrationUtilities.GetCaptureIndex(name0)
                
                for i1 in range(0, len(imageFilesPerCamera[c1])) :
                    image_path1 = imageFilesPerCamera[c1][i1]
                    name1 = os.path.basename(image_path1)
                    points1 = imagePoints[c1][i1]
                    directory1 = os.path.dirname(image_path1)

                    captureIndex1 = CalibrationUtilities.GetCaptureIndex(name1)

                    if captureIndex0 == captureIndex1 and directory0 == directory1 :

                        i = len(validImagesPerPair[(c0, c1)][0]) 
                            
                        validImagesPerPair[(c0, c1)][0].append(imageFilesPerCamera[c0][i0])
                        validPointsPerPair[(c0, c1)][0].append(points0)

                        validImagesPerPair[(c0, c1)][1].append(imageFilesPerCamera[c1][i1])
                        validPointsPerPair[(c0, c1)][1].append(points1)

                        n0 = len(validPointsPerPair[(c0, c1)][0])
                        n1 = len(validPointsPerPair[(c0, c1)][1])
                        
                        m0 = len(validImagesPerPair[(c0, c1)][0])
                        m1 = len(validImagesPerPair[(c0, c1)][1])

                        if n0 != n1 or n0 != m0 or m0 != m1 :
                            print(f'Different data for {c0} and {c1} : {n0} {n1} {m0} {m1}')
                            sys.exit(1)

                        print(f'[{i}]Using {image_path0}, {image_path1}')

        # Generate extrinsic data between cameras
        for c0, c1 in pairs :

            # Array to store object points from all the images.
            imageObjectPoints = [] # 3d points in real world space
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
            objp[:,:2] = np.mgrid[ 0 : patternSize[0], 0 : patternSize[1] ].T.reshape(-1, 2)
            for i in range(0, len(validPointsPerPair[(c0, c1)][0])) :
                imageObjectPoints.append(objp)

            objectPointsArray = np.array(imageObjectPoints)
            
            objectPointsArray *= patternSideLength
            
            print(f'Calibrating stereo pair {c0} {c1}')
            
            error, R, T, E, F = StereoCalibration(objectPointsArray, validPointsPerPair[(c0, c1)][0], validPointsPerPair[(c0, c1)][1], intrinsicMatrices[c0], distortions[c0], intrinsicMatrices[c1], distortions[c1], image_size) 
            
            jsonContent = CalibrationUtilities.StereoCalibrationToJson(intrinsicMatrices[c0], distortions[c0], intrinsicMatrices[c1], distortions[c1], R, T, E, F, patternSideLength)
            filename = os.path.join(args.output_path, f'stereoCalibration{c0}_{c1}.json')
            SaveJsonContent(jsonContent, filename)

    elif args.world_space:

        startTime = time.time()
        # take a single checkerboard laid out on the ground, and compute its origin as the world space origin.
        imageFilesPerCamera = PrepareImages(world_space_path, fileIndices, debugMaxNumCameraImage)

        pin_ids = list(imageFilesPerCamera.keys())

        # load intrinsic data from disk for each camera
        intrinsicMatrices = {}
        distortions = {}
        for k0 in range(0, numCameras) :
            c = pin_ids[k0]
            filename = os.path.join(intrinsic_paths, f'calibration{c}.json')
            jsonContent = LoadJsonContent(filename)
            intrinsicMatrix, distortion, reprojectionError, imageSize, fisheye = CalibrationUtilities.JsonToCameraCalibration(jsonContent)
            intrinsicMatrices[c] = intrinsicMatrix
            distortions[c] = distortion

        imagePoints = []

        validImagesPerPair = {}
        validPointsPerPair = {}
         
        for k0 in range(0, numCameras) :
            c = pin_ids[k0]
            imagePoints[c] = []
            # compute image points
            image_paths = imageFilesPerCamera[c]

            for i in range(0, len(image_paths)):
                image_path = image_paths[i]

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None :
                    print(f'Unable to load {image_path}')
                    continue

                success, corners_subPix = ComputeImagePointCorners(image, image_path, patternSize, searchSize, zeroZoneSize)
                if success:
                    imagePoints[c].append(corners_subPix)
                else:
                    print('No image points found on image {os.path.basename(image images[i][1])}')
                    sys.exit(1)

        # Generate worldspace transform for each camera
        for k0 in range(0, numCameras) :
            c0 = pin_ids[k0]
            # Array to store object points from all the images.
            imageObjectPoints = [] # 3d points in real world space
            # prepare object points, like (0,0,0), (1,0,0) ... (m-2, 0, -(n-1)) ... (m-1, 0, -(n-1))
			# laying in the X,Z plane, with the horizontal checkerboards increasing in X, and vertical
			# checkers increasing in -Z.
            objp = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
            grid = np.mgrid[ 0 : patternSize[0], 0 : patternSize[1] ].T.reshape(-1, 2)
            objp[:,0:1] = grid[:,0:1]
            objp[:,2:3] = -grid[:,1:2]

            for i in range(0, len(imagePoints[c0])) :
                imageObjectPoints.append(objp)

            objectPointsArray = np.array(imageObjectPoints)
            
            objectPointsArray *= patternSideLength

            success, Rvec, T, _ = cv2.solvePnPRansac(objectPointsArray[0], imagePoints[c0][0], intrinsicMatrices[c0], distortions[c0])
            # convert rot vector to rot matrix
            R, _ = cv2.Rodrigues(Rvec)

            if success:
                jsonContent = CalibrationUtilities.WorldSpaceCalibrationToJson(intrinsicMatrices[c0], distortions[c0], R, T, patternSideLength)
                filename = os.path.join(world_space_path, 'worldSpaceCalibration' + str(fileIndices[c0]) + '.json')
                SaveJsonContent(jsonContent, filename)
            else:
                print(f'Could not compute worldspace transform for file indice {c0}')

        print(f'Completed in {(time.time() - startTime) / 60.0} minutes')

if __name__ == "__main__":
    main()
