import numpy as np
import cv2
import pickle
from moviepy.editor import VideoFileClip

def th_bin(img, s_th=(170, 255), x_th=(20, 100), rg_th=(120, 255)):
    '''Function to convert an RGB image to a binary image, based upon color and gradient thresholds

    Args:
        img: input RGB image
        s_th: lower and upper thresholds for s-channel of HLS color space
        x_th: lower and upper thresholds for x-direction gradient sobel-x
        rg_th: lower and upper threshold for R and G channels of RGB color space

    Returns:
        combined_binary: Binary image of the same size as img, after applying the thresholds
    '''
    img = np.copy(img)
    # Prepare thresholded binary for R-Channel
    r_channel = img[:, :, 0]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= rg_th[0]) & (r_channel <= rg_th[1])] = 1
    # Prepare thresholded binary for G-Channel
    g_channel = img[:, :, 1]
    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel >= rg_th[0]) & (g_channel <= rg_th[1])] = 1
    # Prepare thresholded binary for S-Channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_th[0]) & (s_channel <= s_th[1])] = 1
    # Prepare thresholded binary for X-gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x dir
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= x_th[0]) & (scaled_sobel <= x_th[1])] = 1
    # Combine the binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((r_binary == 1) & (g_binary == 1) & ((s_binary == 1) | (sxbinary == 1)))] = 1
    return combined_binary

def warp(img, M):
    '''Function to apply perspective transform to an image

    Args:
        img: input image
        M: Transformation matrix for perspective transform

    Returns:
        warped: Perspective transformed image of the same size as img
    '''
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

class Line():
    '''Class to keep track of lane lines from frame-to-frame in a video

    Example:
        left_line and right_line objects to track of lane-lines

    Attributes:
        fit: Float array of size 3. 2nd degree polynomial coefficients from np.polyfit()
        curv: Current radius of curvature of the lane-line, at the bottom edge of image
        x: Intercept X-value (in pixels) on the bottom edge of image
        frame_count: Current frame count in a video, saturates at count of 10
    '''
    def __init__(self):
        self.fit = None
        self.curv = None
        self.x = None
        self.frame_count = 0


def gen_fit(img, start_x, line, win_h=40, win_w=100, margin=50):
    '''Function to identify lane-line pixels, fit polynomial, calculate curvature and intercept

    Args:
        img: Input image, typically perspective transformed binary image
        start_x: X-start point at bottom edge, typically peak of histogram
        line: Object of Line() class, used to retrieve and store the results
        win_h: Height of the sliding window for lane pixel detection
        win_w: Width of the sliding window for lane pixel detection
        margin: Margin for the window around previous frame poly line

    Returns:
        Nothing. Results are captures in the line object of Line() class
    '''
    ym_per_pix = 3 / 64  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 640  # meteres per pixel in x dimension

    y_size = img.shape[0]

    if (line.frame_count < 10):
        # Use sliding window from bottom edge, going up
        line.frame_count += 1
        num_stripes = int(y_size / win_h)
        mask = np.zeros_like(img)
        median = start_x
        for i in range(num_stripes):
            y_li = y_size - (i + 1) * win_h
            y_ri = y_li + win_h
            x_li = max(0, (median - win_w))
            x_ri = min(img.shape[1], (median + win_w))
            mask[y_li:y_ri, x_li:x_ri] = np.ones((win_h, x_ri - x_li))
            win = img[y_li:y_ri, x_li:x_ri]
            win_ones = np.where(win == 1)
            if (len(win_ones[1]) > 1):
                median = int(np.median(win_ones[1])) + x_li
            masked_img = cv2.bitwise_and(img, mask)
            masked_ones = np.where(masked_img == 1)
    else:
        # Use window around previous frame's line
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        lane_inds = ((nonzerox > (line.fit[0] * (nonzeroy ** 2) + line.fit[1] * nonzeroy + line.fit[2] - margin)) & \
                     (nonzerox < (line.fit[0] * (nonzeroy ** 2) + line.fit[1] * nonzeroy + line.fit[2] + margin)))
        lanex = nonzerox[lane_inds]
        laney = nonzeroy[lane_inds]
        masked_ones = [laney, lanex]

    if (len(masked_ones[0]) > 2500):
        # Update the line, find curvature and X-intercept value at bottom edge
        line.fit = np.polyfit(masked_ones[0], masked_ones[1], 2)
        fit_cr = np.polyfit(masked_ones[0] * ym_per_pix, masked_ones[1] * xm_per_pix, 2)
        line.curv = ((1 + (2 * fit_cr[0] * y_size + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
        line.x = line.fit[0] * y_size ** 2 + line.fit[1] * y_size + line.fit[2]
    # Else the detection is not certain. Retain the previous frame values

def process_image(img):
    '''Main image pipeline

    Args:
        img: Input image, or a frame from incoming video

    Returns:
        result: Output image, after lane detection plotted back on the road
    '''
    global mtx, dist, M, Minv
    global left_line, right_line

    # Remove camera distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Apply color and gradient thresholds
    binary = th_bin(undist)
    # Apply perspective transform
    warped = warp(binary, M)
    # Histogram of warped image
    histogram = np.sum(warped[warped.shape[0] / 2:, :], axis=0)
    # Find left and right peaks
    mid_x = warped.shape[1] / 2
    left_peak = int(np.argmax(histogram[0:mid_x]))
    left_peak_value = int(np.amax(histogram[0:mid_x]))
    right_peak = int(np.argmax(histogram[mid_x:warped.shape[1]]) + mid_x)
    right_peak_value = int(np.amax(histogram[mid_x:warped.shape[1]]))
    # Checks for the peak locations and values
    left_peak_check = (left_peak > 200) & (left_peak < 500) & (left_peak_value > 30)
    right_peak_check = (right_peak > 900) & (right_peak < 1200) & (right_peak_value > 30)
    peak_delta_check = ((right_peak - left_peak) > 500) & ((right_peak - left_peak) < 900)

    if (left_peak_check & right_peak_check & peak_delta_check):
        gen_fit(warped, left_peak, left_line)
        gen_fit(warped, right_peak, right_line)
    # Else retain previous frame values

    # Draw the lanes back on the road
    yvals = np.linspace(0, 100, num=101) * 7.2
    left_fitx = left_line.fit[0] * yvals ** 2 + left_line.fit[1] * yvals + left_line.fit[2]
    right_fitx = right_line.fit[0] * yvals ** 2 + right_line.fit[1] * yvals + right_line.fit[2]

    # Calculate average curvature and distance from center
    average_curvature = (left_line.curv + right_line.curv) / 2
    average_x = (left_line.x + right_line.x) / 2
    center_dist = abs(average_x - warped.shape[1] / 2) * 3.7 / 640

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp(color_warp, Minv)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    font = cv2.FONT_HERSHEY_COMPLEX
    color = (255, 255, 255)
    cv2.putText(result, "Radius of curvature = {:.0f} m".format(average_curvature), (50, 50), font, 1, color, 2)
    cv2.putText(result, "Distance from center = {:.2f} m".format(center_dist), (50, 90), font, 1, color, 2)

    '''
    # Diagnostics
    diagScreen = np.zeros((720, 1600, 3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = result
    binary_stack = np.dstack((binary, binary, binary))
    diagScreen[0:240, 1280:1600] = cv2.resize(255*binary_stack, (320, 240), interpolation=cv2.INTER_AREA)
    warped_stack = np.dstack((warped, warped, warped))
    diagScreen[240:480, 1280:1600] = cv2.resize(255*warped_stack, (320, 240), interpolation=cv2.INTER_AREA)
    diagScreen[480:720, 1280:1600] = cv2.resize(color_warp, (320, 240), interpolation=cv2.INTER_AREA)
    result =  diagScreen
    '''
    return result

############################
# Main function starts here
############################

# Read the camera distortion pickle file and load mtx, dist
with open("camera_dist.p", mode='rb') as f:
    dist_pickle = pickle.load(f)

mtx, dist = dist_pickle["mtx"], dist_pickle["dist"]

# Read the perspective transform pickle file and load M, Minv
with open("M_and_Minv.p", mode='rb') as f:
    M_pickle = pickle.load(f)

M, Minv = M_pickle["M"], M_pickle["Minv"]

left_line = Line()
right_line = Line()

project_output = 'project_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(project_output, audio=False)

challenge_output = 'challenge_output.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
challenge_clip = clip1.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
