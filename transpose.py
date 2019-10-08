# core functions related to transposing broadcast image to 2D bird-view

import cv2
import numpy as np


def masking(image, color_thresh):

    img_masked = []
    for (lower, upper) in color_thresh:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        img_masked = cv2.bitwise_and(image, image, mask=mask)

    return img_masked
    # masking using color channel


def bgr2binary(img_masked):

    img_gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY)
    ret_idle1, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    return img_binary
    # bgr2gray and gray2binary in one step


def bi2edge_ft(img_binary):

    loop_count = 15
    img_binary2 = img_binary
    for i in range(loop_count):
        img_erode = cv2.erode(img_binary2, kernel=None, iterations=5)
        img_dilate = cv2.dilate(img_erode, kernel=None, iterations=5)
        img_blur = cv2.GaussianBlur(img_dilate, (5, 5), 0)
        ret_idle2, img_binary2 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    img_binary3 = cv2.erode(img_binary2, kernel=None, iterations=20)
    # size adjustment
    img_canny = cv2.Canny(img_binary3, 250, 350)
    # edge finding

    return img_canny
    # erosion and dilation to fill void, gaussian blur for AA, shrink to re-align size, then canny edge


def bi2edge_crt(img_binary):

    loop_count = 15
    img_binary2 = img_binary
    for i in range(loop_count):
        img_erode = cv2.erode(img_binary2, kernel=None, iterations=6)
        img_dilate = cv2.dilate(img_erode, kernel=None, iterations=6)
        img_blur = cv2.GaussianBlur(img_dilate, (5, 5), 0)
        ret_idle2, img_binary2 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    img_binary3 = cv2.erode(img_binary2, kernel=None, iterations=22)
    # size adjustment
    img_canny = cv2.Canny(img_binary3, 250, 350)
    # edge finding

    return img_canny
    # erosion and dilation to fill void, gaussian blur for AA, shrink to re-align size, then canny edge


def convex(img_canny):

    cat, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    approx1 = cv2.approxPolyDP(cnt, 10, True)
    hull1 = cv2.convexHull(approx1)
    # increased resolution - more lines
    approx2 = cv2.approxPolyDP(cnt, 20, True)
    hull2 = cv2.convexHull(approx2)
    # reduced resolution - less lines

    return hull1, hull2
    # convex hull to give back-up ref point for transpose, two sets in total


def convex_show(image, hull1, hull2):
    img_hull_1 = image.copy()
    i_last = []
    for i in range(len(hull1) - 1):
        x1, y1 = hull1[i][0]
        x2, y2 = hull1[i + 1][0]
        cv2.line(img_hull_1, (x1, y1), (x2, y2), (0, 0, 255), 5)
        i_last = i
    x1, y1 = hull1[i_last + 1][0]
    x2, y2 = hull1[0][0]
    cv2.line(img_hull_1, (x1, y1), (x2, y2), (0, 0, 255), 5)

    img_hull_2 = image.copy()
    i_last = []
    for i in range(len(hull2) - 1):
        x1, y1 = hull2[i][0]
        x2, y2 = hull2[i + 1][0]
        cv2.line(img_hull_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        i_last = i
    x1, y1 = hull2[i_last + 1][0]
    x2, y2 = hull2[0][0]
    cv2.line(img_hull_2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_hull_1, img_hull_2
    # showing of hull_1 and hull_2 on image


def hough_line_ft(img_canny):

    hough_line_ft = cv2.HoughLines(img_canny, 1, np.pi / 360, 70)
    dim = len(hough_line_ft)
    Lico = np.zeros(shape=(dim, 2))
    hl_pts_ft = np.zeros(shape=(dim, 4))

    m = 0
    for line in hough_line_ft:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))

        hl_pts_ft[m][0] = x1
        hl_pts_ft[m][1] = y1
        hl_pts_ft[m][2] = x2
        hl_pts_ft[m][3] = y2

        slope = (y2 - y1) / (x2 - x1)
        intercept = slope * (0 - x1) + y1
        Lico[m][0] = slope
        Lico[m][1] = intercept
        m = m + 1

    return Lico, hl_pts_ft


def hough_line_crt(img_canny):

    hough_line_crt = cv2.HoughLines(img_canny, 1, np.pi / 360, 150)
    dim = len(hough_line_crt)
    Lico = np.zeros(shape=(dim, 2))
    hl_pts_crt = np.zeros(shape=(dim, 4))

    m = 0
    for line in hough_line_crt:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))

        hl_pts_crt[m][0] = x1
        hl_pts_crt[m][1] = y1
        hl_pts_crt[m][2] = x2
        hl_pts_crt[m][3] = y2

        slope = (y2 - y1) / (x2 - x1)
        intercept = slope * (0 - x1) + y1
        Lico[m][0] = slope
        Lico[m][1] = intercept
        m = m + 1

    return Lico, hl_pts_crt


def hough_line_show(image, hl_pts_ft):

    img_hough = image.copy()
    loop = len(hl_pts_ft[:, 0])
    for i in range(loop):
        p1 = (int(hl_pts_ft[i][0]), int(hl_pts_ft[i][1]))
        p2 = (int(hl_pts_ft[i][2]), int(hl_pts_ft[i][3]))
        cv2.line(img_hough, p1, p2, (0, 0, 255), 3, 4)

    return img_hough


def sort_ft(Lico):

    Lico_thresh = np.mean(Lico[:, 0], axis=0)

    if Lico_thresh >= 0:
        court_side = 1      # 1 is right
    else:
        court_side = 0      # 0 is left

    Lico_ver = np.empty(shape=[0, 2], dtype=float)
    Lico_hor = np.empty(shape=[0, 2], dtype=float)

    Lico_ver_left = np.empty(shape=[0, 2], dtype=float)
    Lico_ver_right = np.empty(shape=[0, 2], dtype=float)

    Lico_hor_up = np.empty(shape=[0, 2], dtype=float)
    Lico_hor_bot = np.empty(shape=[0, 2], dtype=float)

    for i in range(len(Lico)):
        Slope = Lico[i][0]
        if court_side == 1:
            if Slope >= Lico_thresh:
                Lico_ver = np.append(Lico_ver, [Lico[i]], axis=0)
            else:
                Lico_hor = np.append(Lico_hor, [Lico[i]], axis=0)
        else:
            if Slope < Lico_thresh:
                Lico_ver = np.append(Lico_ver, [Lico[i]], axis=0)
            else:
                Lico_hor = np.append(Lico_hor, [Lico[i]], axis=0)

    Lico_ver_y0_inter = -Lico_ver[:, 1] / Lico_ver[:, 0]
    Lico_ver_thresh = (np.max(Lico_ver_y0_inter) + np.min(Lico_ver_y0_inter)) / 2

    for i in range(len(Lico_ver)):
        intercept = Lico_ver_y0_inter[i]

        if intercept >= Lico_ver_thresh:
            Lico_ver_right = np.append(Lico_ver_right, [Lico_ver[i]], axis=0)
        else:
            Lico_ver_left = np.append(Lico_ver_left, [Lico_ver[i]], axis=0)

    Lico_hor_thresh = (np.max(Lico_hor[:, 1]) + np.min(Lico_hor[:, 1])) / 2

    for i in range(len(Lico_hor)):
        intercept = Lico_hor[i][1]
        if intercept >= Lico_hor_thresh:
            Lico_hor_bot = np.append(Lico_hor_bot, [Lico_hor[i]], axis=0)
        else:
            Lico_hor_up = np.append(Lico_hor_up, [Lico_hor[i]], axis=0)

    return Lico_ver_right, Lico_ver_left, Lico_hor_bot, Lico_hor_up, court_side


def sort_crt(Lico, court_side, height, width):
    var_tlr = 1.1

    Lico_inter = Lico[:, 1]
    Lico_abv_zero = Lico_inter[Lico_inter > 0]
    Lico_min_abv_zero_1 = np.min(Lico_abv_zero)
    Lico_trhesh_ver_1 = height * 0.1
    Lico_thresh_hor_1 = Lico_min_abv_zero_1 * var_tlr

    Lico_inter_0 = Lico[:, 0] * width + Lico[:, 1]
    Lico_abv_zero_0 = Lico_inter_0[Lico_inter_0 > 0]
    Lico_min_abv_zero_0 = np.min(Lico_abv_zero_0)
    Lico_trhesh_ver_0 = height * 0.1
    Lico_thresh_hor_0 = Lico_min_abv_zero_0 * var_tlr

    Lico_crt_hor = np.empty(shape=[0, 2], dtype=float)
    Lico_crt_ver = np.empty(shape=[0, 2], dtype=float)

    if court_side == 1:

        for i in range(len(Lico)):

            intercept = Lico[i][1]

            if Lico_min_abv_zero_1 <= intercept <= Lico_thresh_hor_1:
                Lico_crt_hor = np.append(Lico_crt_hor, [Lico[i]], axis=0)
            elif intercept <= Lico_trhesh_ver_1:
                Lico_crt_ver = np.append(Lico_crt_ver, [Lico[i]], axis=0)

    else:
        for i in range(len(Lico)):

            intercept = Lico[i][0] * width + Lico[i][1]

            if Lico_min_abv_zero_0 <= intercept <= Lico_thresh_hor_0 :
                Lico_crt_hor = np.append(Lico_crt_hor, [Lico[i]], axis=0)
            elif intercept <= Lico_trhesh_ver_0:
                Lico_crt_ver = np.append(Lico_crt_ver, [Lico[i]], axis=0)

    return Lico_crt_hor, Lico_crt_ver


def trim_ft (Lico_ver_right, Lico_ver_left, Lico_hor_bot, Lico_hor_up, court_side):

    var_tlr = 0.3
    Lico_ver_left_trim = np.empty(shape=[0, 2], dtype=float)
    Lico_ver_right_trim = np.empty(shape=[0, 2], dtype=float)
    Lico_hor_up_trim = np.empty(shape=[0, 2], dtype=float)
    Lico_hor_bot_trim = np.empty(shape=[0, 2], dtype=float)

    Lico_ver_right_tr = np.mean(Lico_ver_right[:, 0], axis=0)
    Lico_ver_left_tr = np.mean(Lico_ver_left[:, 0], axis=0)
    Lico_hor_bot_tr = np.mean(Lico_hor_bot[:, 0], axis=0)
    Lico_hor_up_tr = np.mean(Lico_hor_up[:, 0], axis=0)

    if court_side == 1:

        for i in range(len(Lico_ver_right[0])):
            if Lico_ver_right[i][0] >= Lico_ver_right_tr*(1-var_tlr) \
                and Lico_ver_right[i][0] <= Lico_ver_right_tr* (1+var_tlr):
                Lico_ver_right_trim = np.append(Lico_ver_right_trim, [Lico_ver_right[i]], axis=0)

        for i in range(len(Lico_ver_left)):
            if Lico_ver_left[i][0] >= Lico_ver_left_tr*(1-var_tlr) \
                and Lico_ver_left[i][0] <= Lico_ver_left_tr* (1+var_tlr):
                Lico_ver_left_trim = np.append(Lico_ver_left_trim, [Lico_ver_left[i]], axis=0)

        for i in range(len(Lico_hor_bot)):
            if Lico_hor_bot[i][0] <= Lico_hor_bot_tr*(1-var_tlr) \
                and Lico_hor_bot[i][0] >= Lico_hor_bot_tr* (1+var_tlr):
                Lico_hor_bot_trim = np.append(Lico_hor_bot_trim, [Lico_hor_bot[i]], axis=0)

        for i in range(len(Lico_hor_up)):
            if Lico_hor_up[i][0] <= Lico_hor_up_tr*(1-var_tlr) \
                and Lico_hor_up[i][0] >= Lico_hor_up_tr* (1+var_tlr):
                Lico_hor_up_trim = np.append(Lico_hor_up_trim, [Lico_hor_up[i]], axis=0)
    else:
        for i in range(len(Lico_ver_right[0])):
            if Lico_ver_right[i][0] <= Lico_ver_right_tr * (1 - var_tlr) \
                    and Lico_ver_right[i][0] >= Lico_ver_right_tr * (1 + var_tlr):
                Lico_ver_right_trim = np.append(Lico_ver_right_trim, [Lico_ver_right[i]], axis=0)

        for i in range(len(Lico_ver_left)):
            if Lico_ver_left[i][0] <= Lico_ver_left_tr * (1 - var_tlr) \
                    and Lico_ver_left[i][0] >= Lico_ver_left_tr * (1 + var_tlr):
                Lico_ver_left_trim = np.append(Lico_ver_left_trim, [Lico_ver_left[i]], axis=0)

        for i in range(len(Lico_hor_bot)):
            if Lico_hor_bot[i][0] >= Lico_hor_bot_tr * (1 - var_tlr) \
                    and Lico_hor_bot[i][0] <= Lico_hor_bot_tr * (1 + var_tlr):
                Lico_hor_bot_trim = np.append(Lico_hor_bot_trim, [Lico_hor_bot[i]], axis=0)

        for i in range(len(Lico_hor_up)):
            if Lico_hor_up[i][0] >= Lico_hor_up_tr * (1 - var_tlr) \
                    and Lico_hor_up[i][0] <= Lico_hor_up_tr * (1 + var_tlr):
                Lico_hor_up_trim = np.append(Lico_hor_up_trim, [Lico_hor_up[i]], axis=0)

    Lico_ver_left = Lico_ver_left_trim
    Lico_ver_right = Lico_ver_right_trim
    Lico_hor_up = Lico_hor_up_trim
    Lico_hor_bot = Lico_hor_bot_trim

    return Lico_ver_right, Lico_ver_left, Lico_hor_bot, Lico_hor_up


def ft_sorted_show(image, Lico_ver_right, Lico_ver_left, Lico_hor_bot, Lico_hor_up, width):

    img_ft_sorted = image.copy()

    for i in range(len(Lico_ver_right)):
        x_1 = 0
        x_2 = width
        y_right_1 = Lico_ver_right[i][0] * x_1 + Lico_ver_right[i][1]
        y_right_2 = Lico_ver_right[i][0] * x_2 + Lico_ver_right[i][1]
        p_right_1 = (int(x_1), int(y_right_1))
        p_right_2 = (int(x_2), int(y_right_2))
        cv2.line(img_ft_sorted, p_right_1, p_right_2, (0, 0, 255), 3, 4)

    for i in range(len(Lico_ver_left)):
        x_1 = 0
        x_2 = width
        y_left_1 = Lico_ver_left[i][0] * x_1 + Lico_ver_left[i][1]
        y_left_2 = Lico_ver_left[i][0] * x_2 + Lico_ver_left[i][1]
        p_left_1 = (int(x_1), int(y_left_1))
        p_left_2 = (int(x_2), int(y_left_2))
        cv2.line(img_ft_sorted, p_left_1, p_left_2, (0, 255, 0), 3, 4)

    for i in range(len(Lico_hor_bot)):
        x_1 = 0
        x_2 = width
        y_bot_1 = Lico_hor_bot[i][0] * x_1 + Lico_hor_bot[i][1]
        y_bot_2 = Lico_hor_bot[i][0] * x_2 + Lico_hor_bot[i][1]
        p_bot_1 = (int(x_1), int(y_bot_1))
        p_bot_2 = (int(x_2), int(y_bot_2))
        cv2.line(img_ft_sorted, p_bot_1, p_bot_2, (255, 0, 0), 3, 4)

    for i in range(len(Lico_hor_up)):
        x_1 = 0
        x_2 = width
        y_up_1 = Lico_hor_up[i][0] * x_1 + Lico_hor_up[i][1]
        y_up_2 = Lico_hor_up[i][0] * x_2 + Lico_hor_up[i][1]
        p_up_1 = (int(x_1), int(y_up_1))
        p_up_2 = (int(x_2), int(y_up_2))
        cv2.line(img_ft_sorted, p_up_1, p_up_2, (0, 125, 200), 3, 4)

    return img_ft_sorted


def crt_sorted_show(image, Lico_crt_hor, Lico_crt_ver, width):
    img_crt_sorted = image.copy()

    for i in range(len(Lico_crt_hor)):
        x_1 = 0
        x_2 = width
        y_top_1 = Lico_crt_hor[i][0] * x_1 + Lico_crt_hor[i][1]
        y_top_2 = Lico_crt_hor[i][0] * x_2 + Lico_crt_hor[i][1]
        p_top_1 = (int(x_1), int(y_top_1))
        p_top_2 = (int(x_2), int(y_top_2))
        cv2.line(img_crt_sorted, p_top_1, p_top_2, (255, 0, 255), 3, 4)

    for i in range(len(Lico_crt_ver)):
        x_1 = 0
        x_2 = width
        y_side_1 = Lico_crt_ver[i][0] * x_1 + Lico_crt_ver[i][1]
        y_side_2 = Lico_crt_ver[i][0] * x_2 + Lico_crt_ver[i][1]
        p_side_1 = (int(x_1), int(y_side_1))
        p_side_2 = (int(x_2), int(y_side_2))
        cv2.line(img_crt_sorted, p_side_1, p_side_2, (244, 164, 95), 3, 4)

    return img_crt_sorted


def ft_combined_show(image, Lico_right, Lico_left, Lico_bot, Lico_up, width):

    img_ft_combined = image.copy()
    x_1 = 0
    x_2 = width
    y_right_1 = Lico_right[0] * x_1 + Lico_right[1]
    y_right_2 = Lico_right[0] * x_2 + Lico_right[1]
    p_right_1 = (int(x_1), int(y_right_1))
    p_right_2 = (int(x_2), int(y_right_2))

    y_left_1 = Lico_left[0] * x_1 + Lico_left[1]
    y_left_2 = Lico_left[0] * x_2 + Lico_left[1]
    p_left_1 = (int(x_1), int(y_left_1))
    p_left_2 = (int(x_2), int(y_left_2))

    y_bot_1 = Lico_bot[0] * x_1 + Lico_bot[1]
    y_bot_2 = Lico_bot[0] * x_2 + Lico_bot[1]
    p_bot_1 = (int(x_1), int(y_bot_1))
    p_bot_2 = (int(x_2), int(y_bot_2))

    y_up_1 = Lico_up[0] * x_1 + Lico_up[1]
    y_up_2 = Lico_up[0] * x_2 + Lico_up[1]
    p_up_1 = (int(x_1), int(y_up_1))
    p_up_2 = (int(x_2), int(y_up_2))

    cv2.line(img_ft_combined, p_right_1, p_right_2, (0, 0, 255), 3, 4)
    cv2.line(img_ft_combined, p_left_1, p_left_2, (0, 255, 0), 3, 4)
    cv2.line(img_ft_combined, p_bot_1, p_bot_2, (255, 0, 0), 3, 4)
    cv2.line(img_ft_combined, p_up_1, p_up_2, (0, 125, 200), 3, 4)

    return img_ft_combined


def crt_combined_show(image, Lico_top, Lico_side, width):

    img_crt_combined = image.copy()
    x_1 = 0
    x_2 = width
    y_top_1 = Lico_top[0] * x_1 + Lico_top[1]
    y_top_2 = Lico_top[0] * x_2 + Lico_top[1]
    p_top_1 = (int(x_1), int(y_top_1))
    p_top_2 = (int(x_2), int(y_top_2))

    y_side_1 = Lico_side[0] * x_1 + Lico_side[1]
    y_side_2 = Lico_side[0] * x_2 + Lico_side[1]
    p_side_1 = (int(x_1), int(y_side_1))
    p_side_2 = (int(x_2), int(y_side_2))

    cv2.line(img_crt_combined, p_top_1, p_top_2, (255, 0, 255), 3, 4)
    cv2.line(img_crt_combined, p_side_1, p_side_2, (244, 164, 95), 3, 4)

    return img_crt_combined


def ref_pts(Lico_right, Lico_left, Lico_bot, Lico_up, Lico_top, Lico_side):

    eq_A_right = [-Lico_right[0], 1]
    eq_A_left = [-Lico_left[0], 1]
    eq_A_bot = [-Lico_bot[0], 1]
    eq_A_up = [-Lico_up[0], 1]
    eq_A_top = [-Lico_top[0], 1]
    eq_A_side = [-Lico_side[0], 1]

    eq_b_right = Lico_right[1]
    eq_b_left = Lico_left[1]
    eq_b_bot = Lico_bot[1]
    eq_b_up = Lico_up[1]
    eq_b_top = Lico_top[1]
    eq_b_side = Lico_side[1]

    mat_A_up_left = np.row_stack((eq_A_up, eq_A_left))
    mat_b_up_left = np.row_stack((eq_b_up, eq_b_left))
    mat_A_up_right = np.row_stack((eq_A_up, eq_A_right))
    mat_b_up_right = np.row_stack((eq_b_up, eq_b_right))
    mat_A_up_side = np.row_stack((eq_A_up, eq_A_side))
    mat_b_up_side = np.row_stack((eq_b_up, eq_b_side))
    mat_A_bot_left = np.row_stack((eq_A_bot, eq_A_left))
    mat_b_bot_left = np.row_stack((eq_b_bot, eq_b_left))
    mat_A_bot_right = np.row_stack((eq_A_bot, eq_A_right))
    mat_b_bot_right = np.row_stack((eq_b_bot, eq_b_right))
    mat_A_bot_side = np.row_stack((eq_A_bot, eq_A_side))
    mat_b_bot_side = np.row_stack((eq_b_bot, eq_b_side))
    mat_A_top_side = np.row_stack((eq_A_top, eq_A_side))
    mat_b_top_side = np.row_stack((eq_b_top, eq_b_side))
    mat_A_top_left = np.row_stack((eq_A_top, eq_A_left))
    mat_b_top_left = np.row_stack((eq_b_top, eq_b_left))
    mat_A_top_right = np.row_stack((eq_A_top, eq_A_right))
    mat_b_top_right = np.row_stack((eq_b_top, eq_b_right))

    up_left = np.linalg.solve(mat_A_up_left, mat_b_up_left)
    up_right = np.linalg.solve(mat_A_up_right, mat_b_up_right)
    up_side = np.linalg.solve(mat_A_up_side, mat_b_up_side)
    bot_left = np.linalg.solve(mat_A_bot_left, mat_b_bot_left)
    bot_right = np.linalg.solve(mat_A_bot_right, mat_b_bot_right)
    bot_side = np.linalg.solve(mat_A_bot_side, mat_b_bot_side)
    top_side = np.linalg.solve(mat_A_top_side, mat_b_top_side)
    top_left = np.linalg.solve(mat_A_top_left, mat_b_top_left)
    top_right = np.linalg.solve(mat_A_top_right, mat_b_top_right)

    pt_up_left = (int(up_left[0]), int(up_left[1]))
    pt_up_right = (int(up_right[0]), int(up_right[1]))
    pt_up_side = (int(up_side[0]), int(up_side[1]))
    pt_bot_left = (int(bot_left[0]), int(bot_left[1]))
    pt_bot_right = (int(bot_right[0]), int(bot_right[1]))
    pt_bot_side = (int(bot_side[0]), int(bot_side[1]))
    pt_top_side = (int(top_side[0]), int(top_side[1]))
    pt_top_left = (int(top_left[0]), int(top_left[1]))
    pt_top_right = (int(top_right[0]), int(top_right[1]))

    return pt_up_left, pt_up_right, pt_up_side, pt_bot_left, pt_bot_right, \
           pt_bot_side, pt_top_side, pt_top_left, pt_top_right


def ref_pts_show(image, pt_up_left, pt_up_right, pt_up_side, pt_bot_left, pt_bot_right,
           pt_bot_side, pt_top_side, pt_top_left, pt_top_right):

    img_pts = image.copy()
    cv2.circle(img_pts, pt_up_left, 5, (45, 0, 255), 15)
    cv2.circle(img_pts, pt_up_side, 5, (45, 255, 255), 15)
    cv2.circle(img_pts, pt_bot_left, 5, (45, 0, 255), 15)
    cv2.circle(img_pts, pt_bot_side, 5, (45, 255, 255), 15)
    cv2.circle(img_pts, pt_top_left, 5, (45, 255, 255), 15)
    cv2.circle(img_pts, pt_top_right, 5, (45, 0, 255), 15)
    cv2.circle(img_pts, pt_top_side, 5, (45, 255, 255), 15)
    cv2.circle(img_pts, pt_bot_right, 5, (45, 0, 255), 15)
    cv2.circle(img_pts, pt_up_right, 5, (45, 0, 255), 15)

    return img_pts




