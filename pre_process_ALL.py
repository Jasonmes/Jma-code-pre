# pre-processing Free_Throw_Area for GD home-court Raw_to_Reference_Line

import cv2
import numpy as np
import transpose as trans

imageName = str('C_test5.PNG')
image = cv2.imread(imageName)
height = image.shape[0]
width = image.shape[1]
# load image

color_ft = [([0, 87, 185], [88, 142, 255])]
color_crt = [([0, 148, 178], [178, 255, 255])]
# set color profile for masking

# %%%%%%  FT-box side line %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img_masked = trans.masking(image, color_ft)

img_binary = trans.bgr2binary(img_masked)

img_canny = trans.bi2edge_ft(img_binary)

# hull1, hull2 = trans.convex(img_canny)
# second choice when houghlines fail

# img_hull_1, img_hull_2 = trans.convex_show(image,hull1,hull2)
# for review purpose

Lico, hl_pts_ft = trans.hough_line_ft(img_canny)

# img_hough_ft = trans.hough_line_show(image, hl_pts_ft)
# for review purpose

Lico_ver_right, Lico_ver_left, Lico_hor_bot, Lico_hor_up, court_side = trans.sort_ft(Lico)

Lico_ver_right, Lico_ver_left, Lico_hor_bot, Lico_hor_up = \
    trans.trim_ft(Lico_ver_right, Lico_ver_left, Lico_hor_bot, Lico_hor_up, court_side)

Lico_right = np.mean(Lico_ver_right, axis=0)
Lico_left = np.mean(Lico_ver_left, axis=0)
Lico_bot = np.mean(Lico_hor_bot, axis=0)
Lico_up = np.mean(Lico_hor_up, axis=0)

# img_ft_sorted = trans.ft_sorted_show(image, Lico_ver_right, Lico_ver_left, Lico_hor_bot, Lico_hor_up, width)
# for review purpose

img_ft_combined = trans.ft_combined_show(image, Lico_right, Lico_left, Lico_bot, Lico_up, width)
# for review purpose

# %%%%%%  Court side-line and bot-line %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img_masked_2 = trans.masking(image, color_crt)

img_binary_2 = trans.bgr2binary(img_masked_2)

img_canny_2 = trans.bi2edge_crt(img_binary_2)

Lico_2, hl_pts_crt = trans.hough_line_crt(img_canny_2)

# img_hough_crt = trans.hough_line_show(image, hl_pts_crt)
# for review purpose

Lico_crt_hor, Lico_crt_ver = trans.sort_crt(Lico_2, court_side, height, width)

Lico_top = np.mean(Lico_crt_hor, axis=0)
Lico_side = np.mean(Lico_crt_ver, axis=0)

# img_crt_sorted = trans.crt_sorted_show(image, Lico_crt_hor, Lico_crt_ver, width)
# for review purpose

img_crt_combined = trans.crt_combined_show(img_ft_combined, Lico_top, Lico_side, width)
# for review purpose

# %%%%%%  Construction of linear coefficient matrix  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# general form Y - ax = b

pt_up_left, pt_up_right, pt_up_side, pt_bot_left, pt_bot_right, \
           pt_bot_side, pt_top_side, pt_top_left, pt_top_right = trans.ref_pts(Lico_right, Lico_left,
                                                                               Lico_bot, Lico_up, Lico_top, Lico_side)

img_pts = trans.ref_pts_show(img_crt_combined, pt_up_left, pt_up_right, pt_up_side, pt_bot_left, pt_bot_right,
           pt_bot_side, pt_top_side, pt_top_left, pt_top_right)
# for review purpose

'''

cv2.namedWindow("img_masked", 0)
cv2.resizeWindow("img_masked", 960, 540)
cv2.imshow("img_masked", img_masked)

cv2.namedWindow("img_binary", 0)
cv2.resizeWindow("img_binary", 960, 540)
cv2.imshow("img_binary", img_binary)

cv2.namedWindow("img_canny", 0)
cv2.resizeWindow("img_canny", 960, 540)
cv2.imshow("img_canny", img_canny)


cv2.namedWindow("img_hull_1", 0)
cv2.resizeWindow("img_hull_1", 960, 540)
cv2.imshow("img_hull_1", img_hull_1)

cv2.namedWindow("img_hull_2", 0)
cv2.resizeWindow("img_hull_2", 960, 540)
cv2.imshow("img_hull_2", img_hull_2)


cv2.namedWindow("img_hough_ft", 0)
cv2.resizeWindow("img_hough_ft", 960, 540)
cv2.imshow("img_hough_ft", img_hough_ft)

cv2.namedWindow("img_ft_sorted", 0)
cv2.resizeWindow("img_ft_sorted", 960, 540)
cv2.imshow("img_ft_sorted", img_ft_sorted)

cv2.namedWindow("img_ft_combined", 0)
cv2.resizeWindow("img_ft_combined", 960, 540)
cv2.imshow("img_ft_combined", img_ft_combined)

cv2.waitKey(0)

cv2.namedWindow("img_masked_2", 0)
cv2.resizeWindow("img_masked_2", 960, 540)
cv2.imshow("img_masked_2", img_masked_2)

cv2.namedWindow("img_binary_2", 0)
cv2.resizeWindow("img_binary_2", 960, 540)
cv2.imshow("img_binary_2", img_binary_2)


cv2.namedWindow("img_canny_2", 0)
cv2.resizeWindow("img_canny_2", 960, 540)
cv2.imshow("img_canny_2", img_canny_2)

cv2.namedWindow("img_hough_crt", 0)
cv2.resizeWindow("img_hough_crt", 960, 540)
cv2.imshow("img_hough_crt", img_hough_crt)

cv2.namedWindow("img_crt_sorted", 0)
cv2.resizeWindow("img_crt_sorted", 960, 540)
cv2.imshow("img_crt_sorted", img_crt_sorted)

cv2.namedWindow("img_crt_combined", 0)
cv2.resizeWindow("img_crt_combined", 960, 540)
cv2.imshow("img_crt_combined", img_crt_combined)
'''

cv2.namedWindow("img_pts", 0)
cv2.resizeWindow("img_pts", 960, 540)
cv2.imshow("img_pts", img_pts)

cv2.waitKey(0)

# visual review


