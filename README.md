# OpenCV Cube Solver

This is a program that can scan the cube faces in different lighting conditions, solves it using Two phase Algorithm and then narrates the solution on screen.
There are two modes in the program viz. Calibration Mode and Solving Mode.
Calibration Mode is used to calibrate the colours which may get affected due to ambient lighting conditions.
Solving Mode works in two steps, scaning the cube faces and narrating the genrated solution.

## Flow of program

### Step 1 : Preprocessig
In this step image is converted to grayscale and to remove noise, this grayscale image is blurred.

### Step 2 : Edge detection
After preprocessing the frame, Canny edge detection is applied to find the edges in frame. Then the edges are filtered out accordiing to shape and only squarish edges are taken into consideration. Only the square shapes which are in a group of 9 are considered, rest are discarded. These 9 squarish edges represents the face of 3x3 cube.

### Step 3 : Recording faces
The colour of each cell in 3x3 grid is estimated by calculating the euclidean distance of the observed colours to standard colours and then stored. Similarly all the faces are scanned.

### Step 4 : Generating and narrating solution
After successful completion of scanning of all faces, solution is generated using Two Phase Algorithm. On showing the cube to camera, the program displays the next move. Following these moves will solve the cube.
