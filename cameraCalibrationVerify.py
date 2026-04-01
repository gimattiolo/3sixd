import numpy as np
import cv2
import argparse
import os
import sys
import time
import shutil

import CalibrationUtilities

def main():
    parser = argparse.ArgumentParser('Verify calibration images')
    parser.add_argument('--pattern_size', type=int, nargs=2, help='2D size of checkerboard pattern to detect')
    parser.add_argument('--input_paths', type=str, nargs='+', help='input folder for calibration files')
    parser.add_argument('--output_path', type=str, help='output folder for calibration files')

    args = parser.parse_args()

    patternSize = CalibrationUtilities.GetPatternSize()
    if args.pattern_size:
        patternSize = tuple(args.pattern_size)
    print(f'Using pattern size {patternSize}')      
    
    startTime = time.time()

    for path in args.input_paths :

        debug_path = os.path.join(path, 'debug')
        shutil.rmtree(debug_path, ignore_errors=True, onerror=None)

        fileList = os.listdir(path)

        os.mkdir(debug_path)

        for i in range(0, len(fileList)):
            filename = fileList[i]
        
            name, file_extension = os.path.splitext(filename)

            if file_extension.lower() == '.json' :
                continue
            
            image_path = os.path.join(path, filename) 

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
            ret, corners = cv2.findChessboardCorners(gray, patternSize, None)
            if not ret :
                print(f'Unable to find chessboard corners in {image_path}')
                continue
            
            decorated_frame = cv2.drawChessboardCorners(image, patternSize, corners, ret)      
            file_name = os.path.basename(image_path)              
            filename_without_ext, ext = os.path.splitext(file_name)
            file_name = f'{filename_without_ext}_decorated{ext}'
            filepath = os.path.join(debug_path, file_name)
            if not cv2.imwrite(filepath, decorated_frame) :
                print(f'Unable to save frame to {filepath}')

    print(f'Completed in {(time.time() - startTime) / 60.0} minutes')

if __name__ == "__main__":
    main()
