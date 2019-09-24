# /usr/bin/python3

import cv2
import os
import glob
import numpy as np


def scale_image(inp_image, scale=0.1):
    width = int(inp_image.shape[1] * 0.1)
    height = int(inp_image.shape[0] * 0.1)
    dim = (width, height)
    # resize image
    return cv2.resize(inp_image, dim, interpolation=cv2.INTER_AREA)


def scale_all_images_in_folder(inp_folder, output_folder, scale=0.1):
    for f in glob.glob(os.path.join(inp_folder, '*')):
        print(f)
        file_name = f.split('/')[-1]
        print(file_name)
        inp_image = cv2.imread(f)

        scaled_image = scale_image(inp_image, scale)

        cv2.imwrite(os.path.join(output_folder, file_name), scaled_image)


def gen_plant_mask(inp_img):
    # generate a mask to figure out the plant region from an RGB image

    # green value greater than 30 in number
    # and greater than 2R and 2B

    hsv_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, (36, 25, 25), (70, 255, 255))
    # mask = np.logical_and(np.logical_and(np.greater(inp_img[:,:,1], green_val_threshold),
    #                                     inp_img[:,:,1] >= green_mult_threshold*inp_img[:,:,0]
    #                                    ),
    #                      inp_img[:,:,1] >= green_mult_threshold*inp_img[:,:,2])

    # open the mask
    #mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    bool_mask = mask > 0

    int_mask = np.zeros_like(mask, dtype=np.uint8)
    int_mask[bool_mask] = 255
    return int_mask


def match_sift_double_iter(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    gray_im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray_im1, mask=gen_plant_mask(img1))
    kp2, des2 = sift.detectAndCompute(gray_im2, mask=gen_plant_mask(img2))

    # FLANN parameters
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    # search_params = dict(checks=50)   # or pass empty dictionary

    #flann = cv2.FlannBasedMatcher(index_params,search_params)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    matches_big = bf.knnMatch(des1, des2, k=5)

    good_matches = []
    pts1 = []
    pts2 = []

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.70*n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    print("selected points with ratio: ", len(pts1))

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(gray_im1, kp1, gray_im2,
                              kp2, matches, None, **draw_params)

    pts1 = np.array([pts1])
    pts2 = np.array([pts2])

    camera_matrix = load_camera_matrix()
    # perform RANSAC
    ransac_e_model, ransac_inliers = cv2.findEssentialMat(
        pts1, pts2, camera_matrix)
    print(ransac_e_model)
    ransac_e_model = ransac_e_model[:, 0:3]

    ransac_f_model = np.matmul(
        np.linalg.inv(np.transpose(camera_matrix)),
        np.matmul(ransac_e_model,
                  np.linalg.inv(camera_matrix)
                  )
    )
    # we have the ransac model -> figure out how many points fit in this model\
    # we are considering overall number, not just the points which pass the ratio test

    print("total input to new:")
    print(len(matches_big))
    match_count_new = 0
    matches_new = list()
    matchMask_new = list()
    for i in range(len(matches_big)):
        point1 = np.append(kp1[matches_big[i][0].queryIdx].pt, 1)
        point2_list = [np.reshape(np.append(
            kp2[match_temp.trainIdx].pt, 1), (1, -1)) for match_temp in matches_big[i]]

        rel_error = np.array([abs(np.asscalar(np.matmul(
            point2, np.matmul(ransac_f_model, point1)))) for point2 in point2_list])

        best_point = np.argmin(rel_error)

        if(rel_error[best_point] < 1e-4):
            match_count_new += 1
            matchMask_new.append([1, 0])
        else:
            matchMask_new.append([0, 0])
        matches_new.append([matches_big[i][best_point],
                            matches_big[i][best_point]])

    draw_params_new = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchMask_new,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # cv.drawMatchesKnn expects list of lists as matches.
    img4 = cv2.drawMatchesKnn(gray_im1, kp1, gray_im2,
                              kp2, matches_new, None, **draw_params_new)

    print("Extra matching val")
    print(match_count_new)

    return img3, img4


def get_chessboard_points(inp_image):
    num_pts_x = 8
    num_pts_y = 6

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((num_pts_x*num_pts_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_pts_x, 0:num_pts_y].T.reshape(-1, 2)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
        inp_image, (num_pts_x, num_pts_y), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # if corners  is not None:
        #corners2 = cv2.cornerSubPix(inp_image, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(
            inp_image, (num_pts_x, num_pts_y), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

        cv2.imwrite('./temp1.jpg', img)
    else:
        print("not found")

    cv2.destroyAllWindows()


def calibrate_camera(inp_image_folder):
    images = [cv2.imread(f) for f in glob.glob(
        os.path.join(inp_image_folder, '*.*'))]
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    for img in gray_images:
        get_chessboard_points(img)


def load_camera_matrix():
    fx = 357.5901
    fy = 384.8436
    cx = 230.8863
    cy = 173.6757
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def load_image(imName, path='../data/lettuce_home/'):
    img = cv2.imread(path + imName)

    return img


# calibrate_camera('../data/calibration/')
