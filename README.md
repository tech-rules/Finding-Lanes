## Advanced lane detection using OpenCV
The goals of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
The code for this step is contained in the second cell of IPython notebook [Camera_Undistort.ipynb](Camera_Undistort.ipynb). 
Most of the calibration images in the camera_cal/ directory have 9 corners along X-axis and 6 corners along Y-axis. I start by preparing "object points (objp)", which will be the (x, y, z) coordinates of the chessboard corners with 9 X-corners and 6 Y-corners. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.
```python
# Prepare objectpoints
NUM_XCOR = 9
NUM_YCOR = 6
objp = np.zeros((NUM_YCOR*NUM_XCOR,3), np.float32)
objp[:,:2] = np.mgrid[0:NUM_XCOR, 0:NUM_YCOR].T.reshape(-1,2)
```
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
```python
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (NUM_XCOR, NUM_YCOR), None)

    # If found, add object points, image points
    if ret == True:
        img_size = (img.shape[1], img.shape[0])
        objpoints.append(objp)
        imgpoints.append(corners)
```
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the calibration images using the `cv2.undistort()` function in the third code cell of [Camera_Undistort.ipynb](Camera_Undistort.ipynb). One example result is:
![](readme_images/undistort.png?raw=true)   
Finally, the result of the camera calibration (camera matrix and distortion coefficients) are then stored in a pickle file for later use in the image and video pipelines.
```python
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_dist.p", "wb" ) )
```

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Distortion-corrected versions of the test images can be found in [output_images](output_images). Here is one example as a reference (undistorted_test1):
![](output_images/undistorted_test1.jpg?raw=true)

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. You can check my steps in the function `th_bin()` in the 2nd code cell of [Pipeline.ipynb](Pipeline.ipynb). Here's an example of my output for this step.
![](readme_images/binary.png?raw=true)   

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I performed perspective transform using OpenCV's `warpPerspective()` function. The transformation matrix for `warpPerspective()` was generated using `getPerspectiveTransform()` with the following hardcoded source and destination points:
```python
src = np.float32([[596, 450], [210, 719], [1100, 719], [685, 450]])
dst = np.float32([[310, 120], [310, 719], [1000, 719], [1000, 120]])
```
The resulting transformation matrix (and inverse transformation matrix) were saved in a pickle file for use with the image and video pilelines.
```python
# Create matrices for the transforma nd inverse transform
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Save M and Minv in a pickle file
Mat_pickle = {}
Mat_pickle["M"] = M
Mat_pickle["Minv"] = Minv
pickle.dump(Mat_pickle, open( "M_and_Minv.p", "wb" ) )
```
Perspective transformation in my code is performed by a function called `warp()`, which is in the 2nd code cell of [Perspective_Transform.ipynb](Perspective_Transform.ipynb).  The `warp()` function takes as inputs an image (`img`) and transformation matrix(`M`) and returns the perspective transformed image by calling the OpenCV function `warpPerspective()`. Here is the result of perspective transformation with the above chosen source and destination  points:
![](readme_images/perspective.png?raw=true)   

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used two different mechanisms in order to find the pixels associated with the lane-lines. For the first 10 frames, I used the following steps (lines 96-113 of [video_pipeline.py](video_pipeline.py)):
* Take a histogram of bottom half of the thresholded binary and perspective transformed image
* Find the peaks of left-half and right-half of the histogram
* These peak  locations become the starting points at the bottom of the image
* The image is divided into multiple stripes and an upward sliding window is idenatified for each stripe
* The starting bottom window is around the left and right peaks of histogram, and each subsequent window location, as we traverse up the stripes, is based upon the median non-zero pixel location in the previous window
* As you reach to the top stripe, these windows cumulatively create a mask, and non-zero pixels within this mask are used as the detected lane-line pixels
* Finally `np.polyfit()` function is used to fit a second order curve to the detected lane-line pixels

For the subsequent frames (after the initial 10 frames) the following steps are used (line 115-123 in [video_pipeline.py](video_pipeline.py)):
* Create a window of +/- 100 pixels around the lane-lines in the previous frame
* Use this window as the region-of-interest for the current frame and detect non-zero pixels
* Same as above, use `np.polyfit()` function to fit a second order curve to the detected lane-line pixels
* I defined a `Line()` class to keep track of previous polyfit values, curvature, x-intercept and frame count
* left_line and right_light are the two objects of the `Line()` class in [video_pipeline.py](video_pipeline.py)

An example result on a test image, after fitting the polynomials for right and left lines:
![](readme_images/polyfit_nohist.png?raw=true)   

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I first estimated the meters-per-pixel (in the perspective transformed straight-lines image) based upon the following facts:
* The lane is about 3.7 meters wide (for calculating meters-per-pixel in X-direction)
* Each dashed lines are about 3 meters long (for calculating meters-per-pixel in Y-direction)

I then fit another set of polynomials with lane detected X and Y values converted to meters. The coefficients of these left and right "world space" polynomials were used to calculate the left and right curvature (at the bottom points) using the formula given in [radius of curvature link](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). The two curvatures were then averaged and displayed on the frame image using OpenCV `putText()` function. You can find this code in (line 128-129, and 175 in [video_pipeline.py](video_pipeline.py)).

For calculating the position of the vehicle with respect to center, I assumed that the camera is mounted at the center of the car. So, the distance between the center of the frame-image and average of the the bottom-intercepts of left and right lanes, when converted from pixels to meters, gives us an estimate of the position of the vehicle. this arithmetic is performed in function `process_image()` of [video_pipeline.py](video_pipeline.py), where variable `center_dist` is calculated (line 176-177).

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

OpenCV `fillPoly()` function was used to plot and fill (with green color) the area between the detected lane-lines. The resulting image was perspective transformed back to the original car view using Minv and `warp()` function. OpenCV function `addWeighted()` was then used to blend the original undistorted image with the lane-drawn image and returned as the result of `process_image()` function in [video_pipeline.py](video_pipeline.py). Here is an example of my result on a test image:
![](readme_images/pipeline.png?raw=true)   

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I have added my final video output to this github repo. The link is:
[project_output.mp4](project_output.mp4)

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This was a great project for me to become familiar with the image processing and transformation techniques avaialable to us. The main challenge that I faced was trying to find color and gradient thresholds that work most of the time. Compared to the deep learning approach where the features and parameters are learned during training, the approach in this project was more of feature engineering. Given the limited amount of time I had, my parameter selection, though it worked on the test images and project video, it is unlikely to generalize to more complex cases e.g. different lighting conditions and shadows, roads with uphill or downhill slopes, and extremely sharp turns. In order to make the lane detection more robust and more "genralized" I believe a deep learning appraoch would be more suited. Deep learning approach will require much more data during training than what we had in this project, but this kind of data is becoming more and more easily avialable now.

