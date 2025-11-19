import numpy as np
import cv2
import glob
import argparse
import os
import sys
import time
import pymongo

import CalibrationUtilities
import WaveUtilities

def ComputeImagePointCorners(image, patternSize, searchSize, zeroZoneSize):
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    ret, corners = cv2.findChessboardCorners(gray, patternSize, None)
    if not ret :
        print(f'Unable to find chessboard corners')
        return False, None

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    corners_subPix = cv2.cornerSubPix(gray, corners, searchSize, zeroZoneSize, criteria)
    return True, corners_subPix

# see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# see https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
def CameraCalibration(patternSize, searchSize, zeroZoneSize, images, imageSize, useGuess, intrinsicMatrix, distortion) :
    # Array to store image points from all the images.
    imagePoints = [] # 2d points in image plane.

    for i in range(0, len(images)) :
        print(f'Processing image {i} {images[i][1]}')
        success, corners_subPix = ComputeImagePointCorners(images[i][0], patternSize, searchSize, zeroZoneSize)
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

    objectPoints = np.array(imageObjectPoints)

    if len(imagePoints) <= 0 :
        print(f'No image points found')
        return

    numImages = max(1, len(imagePoints))
       
    size = (imageSize[1], imageSize[2])

    print(f'Guess flag enabled : {useGuess}')
       
    print(f'Running calibration routine. This might take a while and be unresponsive, depending on the number of input images : {numImages}')
    flags = 0
    
    if useGuess :   
        flags = cv2.CALIB_USE_INTRINSIC_GUESS 
    
    error, intrinsicMatrix, distortion, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, size, intrinsicMatrix, distortion, None, None, flags, criteria)
    
    mean_error = 0.0
    for i in range(len(imagePoints)):
        imagePoints2, _ = cv2.projectPoints(objectPoints[i], rvecs[i], tvecs[i], intrinsicMatrix, distortion)
        numImagePoints = max(1, len(imagePoints2))
        error = cv2.norm(imagePoints[i], imagePoints2, cv2.NORM_L2) / numImagePoints
        mean_error += error

    mean_error /= numImages
    print(f"Mean reprojection error : {mean_error}")
    
    return error, intrinsicMatrix, distortion, mean_error, imagePoints

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

def PrepareImages(calibrationPath, fileIndices, debugMaxNumCameraImage):
    numCameras = len(fileIndices)
    imagesPerCamera = []
    sizesPerCamera = []

    for i in range(0, numCameras) :
        imagesPerCamera.append([])
        sizesPerCamera.append([])

    fileList = os.listdir(calibrationPath)
    for i in range(0, len(fileList)):
        filename = fileList[i]
       
        name, file_extension = os.path.splitext(filename)

        if file_extension.lower() == '.json' :
            continue

        camIndex = CalibrationUtilities.GetCameraIndex(filename)

        if camIndex == -1 :
            print(f'Unable to extract camera index from string {filename}')
            continue

        if not (camIndex in fileIndices):
            continue

        c = fileIndices.index(camIndex)

        if len(imagesPerCamera[c]) >= debugMaxNumCameraImage :
            continue
        
        fullFilename = os.path.join(calibrationPath, filename)
    
        image = cv2.imread(fullFilename, cv2.IMREAD_COLOR)
        if image is None :
            print(f'Unable to load {fullFilename}')
            continue
            
        # we store channels, width and height
        size = image.shape[::-1]
        sizesPerCamera[c].append(size)

        print (f'Loading image #{i} {fullFilename} {size} for camera {c}')
        imagesPerCamera[c].append((image, fullFilename))

    for c in range(0, numCameras) :
        for i in range(1, len(sizesPerCamera[c])) :
            if sizesPerCamera[c][i] != sizesPerCamera[c][i - 1] :
                print(f'Camera {c} : image {i} size {sizesPerCamera[c][i]} is different from image {i - 1} size {sizesPerCamera[c][i - 1]}')
                return

    for c in range(0, numCameras) :
        if len(imagesPerCamera[c]) == 0 :
            print(f'No image found in {calibrationPath} for camera {c}')

    return imagesPerCamera, sizesPerCamera

def main():
    parser = argparse.ArgumentParser('Create calibration data from images')
    parser.add_argument('--intrinsic', dest='intrinsic', action="store_true", help='create intrinsic matrix json file (requires --intrinsicpath)')
    parser.add_argument('--intrinsicpath', dest='intrinsic_path', type=str, help='set the intrinsic image capture source folder/data export folder')
    parser.add_argument('--extrinsic', dest='extrinsic', action="store_true", help='create extrinsic matrix json file (requires both --intrinsicpath and --extrinsicpath)')
    parser.add_argument('--extrinsicpath', dest='extrinsic_path', type=str, help='set the extrinsic image capture source folder/data export folder')
    parser.add_argument('--worldspace', dest='world_space', action="store_true", help='generate a worldspace transform for each camera')
    parser.add_argument('--worldspacepath', dest='world_space_path', type=str, help='set the world space image capture source folder/data export folder')
    parser.add_argument('--fileindices', dest='file_indices', type=int, nargs='+', help='file camera indices to process')
    parser.add_argument('--patternsize', dest='pattern_size', type=int, nargs=2, help='2D size of checkerboard pattern to detect')
    parser.add_argument('--patternsidelength', dest='pattern_side_length', type=float, default=0.0, help='length of checkerboard square side in millimeters')
    parser.add_argument('--searchsize', dest='search_size', type=int, nargs=2, help='search window half-size for finding the checkerboard')
    parser.add_argument('--zerozonesize', dest='zero_zone_size', type=int, nargs=2, help='search zone dead region half-size that is ignored when looking for checkerboard gradients')
    parser.add_argument('--maximages', dest='max_images', type=int, default=-1, help='maximum number of images to load')
    args = parser.parse_args()

    modeTotals = (1 if args.intrinsic else 0) + (1 if args.extrinsic else 0) + (1 if args.world_space else 0)
    if modeTotals != 1:
        print ('You must run in either --intrinsic, --extrinsic, or --worldspace mode, and not multiple modes at once.')
        sys.exit(1)

    if args.intrinsic and not args.intrinsic_path:
        print ('You must set the --intrinsicpath to run in --intrinsic mode')
        sys.exit(1)

    if args.extrinsic and not (args.intrinsic_path and args.extrinsic_path):
        print ('You must set the --intrinsicpath and --extrinsicpath to run in --extrinsic mode')
        sys.exit(1)

    if args.world_space and not (args.intrinsic_path and args.world_space_path):
        print ('You must set the --intrinsicpath and --worldspacepath to run in --worldspace mode')
        sys.exit(1)

    genericPath = CalibrationUtilities.GetCalibrationPath()
    intrinsicPath = genericPath
    if args.intrinsic_path:
        intrinsicPath = args.intrinsic_path
        print(f'Using intrinsic image/data path {intrinsicPath}')

    extrinsicPath = genericPath
    if args.extrinsic_path:
        extrinsicPath = args.extrinsic_path
        print(f'Using extrinsic image/data path {extrinsicPath}')

    worldSpacePath = genericPath
    if args.world_space_path:
        worldSpacePath = args.world_space_path
        print(f'Using worldspace image/data path {worldSpacePath}')

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

    if args.intrinsic and intrinsicPath and not os.path.exists(intrinsicPath) :
        print(f'Invalid intrinsic path {intrinsicPath}')
        sys.exit(1)

    if args.extrinsic and extrinsicPath and not os.path.exists(extrinsicPath) :
        print(f'Invalid extrinsic path {extrinsicPath}')
        sys.exit(1)

    if args.world_space and worldSpacePath and not os.path.exists(worldSpacePath) :
        print(f'Invalid worldspace path {worldSpacePath}')
        sys.exit(1)

    sideLength = CalibrationUtilities.GetCalibrationPatternSquareSideLengthInMeters()
    if args.pattern_side_length > 0.0:
        sideLength = args.pattern_side_length
    S = np.array( [ sideLength ] )

    if args.intrinsic: 
    
        startTime = time.time()
    
        imagesPerCamera, sizesPerCamera = PrepareImages(intrinsicPath, fileIndices, debugMaxNumCameraImage)
        intrinsicMatrices = []
        distortions = []
        mean_errors = []

        imagePoints = []
        for c in range(0, len(imagesPerCamera)) :
            intrinsicMatrices.append([])
            distortions.append([])
            mean_errors.append([])
            imagePoints.append([])
            print(f'Calibrating camera {c}')
            
            cameraFilename = os.path.join(intrinsicPath, 'calibration' + str(c) + '.json')

            camaraCalibrationLoaded, intrinsicMatrices[c], distortions[c], e, _ = WaveUtilities.LoadCameraCalibration(cameraFilename)

            error, intrinsicMatrices[c], distortions[c], mean_errors[c], imagePoints[c] = CameraCalibration(patternSize, searchSize, zeroZoneSize, imagesPerCamera[c], sizesPerCamera[c][0], camaraCalibrationLoaded, intrinsicMatrices[c], distortions[c])
            jsonContent = CalibrationUtilities.CameraCalibrationToJson(intrinsicMatrices[c], distortions[c], mean_errors[c], sizesPerCamera[c][0]) 

            filename = os.path.join(intrinsicPath, 'calibration' + str(fileIndices[c]) + '.json')
            SaveJsonContent(jsonContent, filename)

        print(f'Completed in {(time.time() - startTime) / 60.0} minutes')

    elif args.extrinsic:
        startTime = time.time()
    
        imagesPerCamera, sizesPerCamera = PrepareImages(extrinsicPath, fileIndices, debugMaxNumCameraImage)

        # load intrinsic data from disk for each camera
        intrinsicMatrices = []
        distortions = []
        for c in range(0, numCameras) :
            filename = os.path.join(intrinsicPath, 'calibration' + str(fileIndices[c]) + '.json')
            jsonContent = LoadJsonContent(filename)
            intrinsicMatrix, distortion, reprojectionError, imageSize = CalibrationUtilities.JsonToCameraCalibration(jsonContent)
            intrinsicMatrices.append(intrinsicMatrix)
            distortions.append(distortion)

        imagePoints = []

        validImagesPerPair = {}
        validPointsPerPair = {}
 
        for c in range(0, numCameras) :
            imagePoints.append([])
            # compute image points
            images = imagesPerCamera[c]
            
            for i in range(0, len(images)):
                success, corners_subPix = ComputeImagePointCorners(images[i][0], patternSize, searchSize, zeroZoneSize)
                if success:
                    imagePoints[c].append(corners_subPix)
                else:
                    print('No image points found on image {os.path.basename(image images[i][1])}')
                    sys.exit(1)

        for c0 in range(0, numCameras) :
            for c1 in range(c0 + 1, numCameras) :
                validImagesPerPair[(c0, c1)] = ([], [])
                validPointsPerPair[(c0, c1)] = ([], [])

        # filter out images without matching point pairs
        for c0 in range(0, numCameras) :
            for c1 in range(c0 + 1, numCameras) :
                print(f'Collecting necessary images for pair {c0} {c1}')
                for i0 in range(0, len(imagesPerCamera[c0])) :
                    image0 = imagesPerCamera[c0][i0][0]
                    name0 = os.path.basename(imagesPerCamera[c0][i0][1])
                    points0 = imagePoints[c0][i0]

                    captureIndex0 = CalibrationUtilities.GetCaptureIndex(name0)
                    
                    for i1 in range(0, len(imagesPerCamera[c1])) :
                        image1 = imagesPerCamera[c1][i1][0]
                        name1 = os.path.basename(imagesPerCamera[c1][i1][1])
                        points1 = imagePoints[c1][i1]

                        captureIndex1 = CalibrationUtilities.GetCaptureIndex(name1)
                        
                        if captureIndex0 == captureIndex1 :

                            i = len(validImagesPerPair[(c0, c1)][0]) 
                                
                            validImagesPerPair[(c0, c1)][0].append(imagesPerCamera[c0][i0])
                            validPointsPerPair[(c0, c1)][0].append(points0)

                            validImagesPerPair[(c0, c1)][1].append(imagesPerCamera[c1][i1])
                            validPointsPerPair[(c0, c1)][1].append(points1)

                            n0 = len(validPointsPerPair[(c0, c1)][0])
                            n1 = len(validPointsPerPair[(c0, c1)][1])
                            
                            m0 = len(validImagesPerPair[(c0, c1)][0])
                            m1 = len(validImagesPerPair[(c0, c1)][1])

                            if n0 != n1 or n0 != m0 or m0 != m1 :
                                print(f'Different data for {c0} and {c1} : {n0} {n1} {m0} {m1}')
                                sys.exit(1)

                            print(f'Will use image pair {i} {name0} {name1}')

        # Generate extrinsic data between cameras
        for c0 in range(0, numCameras) :
            for c1 in range(c0 + 1, numCameras) :

                # Array to store object points from all the images.
                imageObjectPoints = [] # 3d points in real world space
                # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                objp = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
                objp[:,:2] = np.mgrid[ 0 : patternSize[0], 0 : patternSize[1] ].T.reshape(-1, 2)
                for i in range(0, len(validPointsPerPair[(c0, c1)][0])) :
                    imageObjectPoints.append(objp)

                objectPointsArray = np.array(imageObjectPoints)
                
                objectPointsArray *= S
                
                print(f'Calibrating stereo pair {fileIndices[c0]} {fileIndices[c1]}')
                
                error, R, T, E, F = StereoCalibration(objectPointsArray, validPointsPerPair[(c0, c1)][0], validPointsPerPair[(c0, c1)][1], intrinsicMatrices[c0], distortions[c0], intrinsicMatrices[c1], distortions[c1], sizesPerCamera[c0][0]) 
                
                jsonContent = CalibrationUtilities.StereoCalibrationToJson(intrinsicMatrices[c0], distortions[c0], intrinsicMatrices[c1], distortions[c1], R, T, E, F, S)
                filename = os.path.join(extrinsicPath, 'stereoCalibration' + str(fileIndices[c0]) + '_' + str(fileIndices[c1]) + '.json')
                SaveJsonContent(jsonContent, filename)
    elif args.world_space:
        startTime = time.time()
        # take a single checkerboard laid out on the ground, and compute its origin as the world space origin.
        imagesPerCamera, sizesPerCamera = PrepareImages(worldSpacePath, fileIndices, debugMaxNumCameraImage)

        # load intrinsic data from disk for each camera
        intrinsicMatrices = []
        distortions = []
        for c in range(0, numCameras) :
            filename = os.path.join(intrinsicPath, 'calibration' + str(fileIndices[c]) + '.json')
            jsonContent = LoadJsonContent(filename)
            intrinsicMatrix, distortion, reprojectionError, imageSize = CalibrationUtilities.JsonToCameraCalibration(jsonContent)
            intrinsicMatrices.append(intrinsicMatrix)
            distortions.append(distortion)

        imagePoints = []

        validImagesPerPair = {}
        validPointsPerPair = {}
         
        for c in range(0, numCameras) :
            imagePoints.append([])
            # compute image points
            images = imagesPerCamera[c]
            for i in range(0, len(images)):
                success, corners_subPix = ComputeImagePointCorners(images[i][0], patternSize, searchSize, zeroZoneSize)
                if success:
                    imagePoints[c].append(corners_subPix)
                else:
                    print('No image points found on image {os.path.basename(image images[i][1])}')
                    sys.exit(1)

        # Generate worldspace transform for each camera
        for c0 in range(0, numCameras) :
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
            
            objectPointsArray *= S

            success, Rvec, T, _ = cv2.solvePnPRansac(objectPointsArray[0], imagePoints[c0][0], intrinsicMatrices[c0], distortions[c0])
            # convert rot vector to rot matrix
            R, _ = cv2.Rodrigues(Rvec)

            if success:
                jsonContent = CalibrationUtilities.WorldSpaceCalibrationToJson(intrinsicMatrices[c0], distortions[c0], R, T, S)
                filename = os.path.join(worldSpacePath, 'worldSpaceCalibration' + str(fileIndices[c0]) + '.json')
                SaveJsonContent(jsonContent, filename)
            else:
                print(f'Could not compute worldspace transform for file indice {c0}')

        print(f'Completed in {(time.time() - startTime) / 60.0} minutes')

if __name__ == "__main__":
    main()