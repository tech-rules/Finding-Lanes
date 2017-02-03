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
The code for my perspective transform includes a function called `warp()`, which appears in the 2nd code cell of [Perspective_Transform.ipynb](Perspective_Transform.ipynb).  The `warp()` function takes as inputs an image (`img`) and transformation matrix(`M`). Here is the result of perspective transformation with the above chosen source and destination  points:
![](readme_images/perspective.png?raw=true)   

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
![](readme_images/polyfit_nohist.png?raw=true)   


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:
![](readme_images/pipeline.png?raw=true)   

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

