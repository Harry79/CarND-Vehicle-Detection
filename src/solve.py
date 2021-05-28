import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import numpy.linalg as linalg
import detectCars
import filterCars

def saveRGB(filename, img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, bgr)

def saveYCrCb(filename, img):
    bgr = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(filename, bgr*255)

def combined(img):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def initUndist():
    with open('camcalib.pkl', 'rb') as input:
        global mtx
        mtx = pickle.load(input)
        global dist
        dist = pickle.load(input)

#    w = 325
    w = 400
    t = -400
    b = 716

    src = np.float32([[232,693], [1048, 693], [ 672,443-3], [608,443-3]])
    dst = np.float32([[640-w,b], [640+w, b], [640+w,t], [640-w,t]])
    global M
    M = cv2.getPerspectiveTransform(src, dst)

    global lanewidthArr
    lanewidthArr = []

last_bboxes = []

def process_image(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    #cv2.imshow('undist',undist)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    binary_output = combined(undist)

    #cv2.imshow('image',binary_output*255)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    img_size = (undist.shape[1], undist.shape[0])

    #cv2.imwrite('binary_output.jpg', binary_output*255)

    binary_warped = cv2.warpPerspective(binary_output, M, img_size)
    #cv2.imwrite('order-detected-warped.jpg', binary_warped*255)

    #warped = cv2.warpPerspective(undist, M, img_size)
    #saveRGB('warped.jpg', warped)

    #cv2.imshow('image',bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #binary_output = combined(warped)

    #cv2.imshow('image',binary_output*255)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite('order-warped-detected.jpg', binary_output*255)

    #cv2.imshow('image',hls_binary*255)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 60
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)



    ## VISUALIZATON
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    #debug stuff
    if 0:
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    margin = 10
    window_img = np.zeros_like(out_img)

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/(700*4.29/3.7) # meters per pixel in x dimension

    bottomIdx = len(right_fitx)-1
    #print("rightfitx = ", right_fitx[bottomIdx])
    #print("leftfitx = ", left_fitx[bottomIdx])

    posinlane = (right_fitx[bottomIdx]+left_fitx[bottomIdx]-binary_warped.shape[1])/2.0*xm_per_pix
    lanewidth = (right_fitx[bottomIdx]-left_fitx[bottomIdx])*xm_per_pix

    #print("posinlane = ", posinlane);
    #print("lanewidth = ", lanewidth);

    lanewidthArr.append(lanewidth)
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,255, 0))

    if 0:
        result = cv2.addWeighted(result, 1, window_img, 0.9, 0)

    #saveRGB('result.jpg', result)

    #cv2.imshow('image',result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #print("len(ploty) = ", len(ploty))
    #print("len(leftx) = ", len(leftx))
    #print("len(rightx) = ", len(rightx))

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    y_eval = np.max(lefty)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    y_eval = np.max(righty)
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    # https://stackoverflow.com/questions/15263597/convert-floating-point-number-to-certain-precision-then-copy-to-string
    newer_method_string = "{:.3f}".format(posinlane) + " m"
    #http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(window_img,newer_method_string,(600,700), font, 1,(255,255,255),2,cv2.LINE_AA)

    newer_method_string = "{:.0f}".format(left_curverad) + " m"
    cv2.putText(window_img,newer_method_string,(300,700), font, 1,(255,255,255),2,cv2.LINE_AA)
    newer_method_string = "{:.0f}".format(right_curverad) + " m"
    cv2.putText(window_img,newer_method_string,(900,700), font, 1,(255,255,255),2,cv2.LINE_AA)


    warpedback = cv2.warpPerspective(window_img, linalg.inv(M), img_size)

    bboxes = detectCars.detect_cars(undist)

    global last_bboxes;
    merged_bboxes = last_bboxes + bboxes
    last_bboxes = bboxes

    undist = filterCars.filter_cars(undist, merged_bboxes)

    undist = cv2.addWeighted(undist, 1, warpedback, 0.9, 0)

    return undist
    #saveRGB('undist-overlay.jpg', undist)
