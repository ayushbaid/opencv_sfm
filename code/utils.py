# /usr/bin/python3

import cv2
import os
import glob
import numpy as np
import copy
import matplotlib.pyplot as plt


def scale_image(inp_image, scale=0.1):
    width = int(inp_image.shape[1] * scale)
    height = int(inp_image.shape[0] * scale)
    dim = (width, height)
    # resize image
    return cv2.resize(inp_image, dim, interpolation=cv2.INTER_AREA)


def scale_all_images_in_folder(inp_folder, output_folder, scale=0.1):
    for f in glob.glob(os.path.join(inp_folder, '*.jpg')):
        file_name = f.split('/')[-1]
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
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    bool_mask = mask > 0

    int_mask = np.zeros_like(mask, dtype=np.uint8)
    int_mask[bool_mask] = 255
    return int_mask


def convert_img_to_homogenous(inp_points):
    return np.append(inp_points, np.ones((inp_points.shape[0], 1)), axis=1)


def evaluate_fundamental_matrix_old(test_set_im1,
                                    test_set_im2,
                                    fundamental_mat,
                                    threshold=3e-4):
    # test sets are of size Nx3 (homogenous coordinates)
    temp_mat = np.matmul(test_set_im2, fundamental_mat)
    sum_mat = np.abs(np.sum(np.multiply(temp_mat, test_set_im1), axis=1))
    inliear_mask = sum_mat < threshold

    inlier_eval = np.asscalar(np.sum(inliear_mask, axis=None))

    return inlier_eval, inliear_mask


def evaluate_fundamental_matrix(test_set_im1,
                                test_set_im2,
                                fundamental_mat,
                                threshold=1):
    # test sets are of size Nx3 (homogenous coordinates)

    # left side test
    left_epipolar_lines = np.matmul(test_set_im1, fundamental_mat.T)
    line_dist_normalizer_left = np.sqrt(
        np.sum(left_epipolar_lines[:, :-1]**2, axis=1))
    distances_left = np.abs(
        np.sum(np.multiply(left_epipolar_lines, test_set_im2), axis=1) /
        line_dist_normalizer_left)
    inlier_mask_left = distances_left < threshold

    # right side test
    right_epipolar_lines = np.matmul(test_set_im2, fundamental_mat)
    line_dist_normalizer_right = np.sqrt(
        np.sum(right_epipolar_lines[:, :-1]**2, axis=1))
    distances_right = np.abs(
        np.sum(np.multiply(right_epipolar_lines, test_set_im1), axis=1) /
        line_dist_normalizer_right)
    inlier_mask_right = distances_right < threshold

    inlier_mask = np.logical_and(inlier_mask_left, inlier_mask_right)
    inlier_eval = np.asscalar(np.sum(inlier_mask, axis=None))

    return inlier_eval, inlier_mask


def custom_ransac(sampling_set_im1,
                  sampling_set_im2,
                  test_set_im1,
                  test_set_im2,
                  camera_matrix,
                  num_iters=1000):
    min_num_points = 8
    if (sampling_set_im1.shape[0] < min_num_points):
        raise Exception("insufficient number of points")

    essential_mat = None
    inliear_mask = None
    max_inlier_count = -1

    for iter_idx in range(num_iters):
        # get samples for hypothesis generation
        sampling_idx = np.random.choice(sampling_set_im1.shape[0],
                                        min_num_points,
                                        replace=False)

        essential_mat_candidate, _ = cv2.findEssentialMat(
            sampling_set_im1[sampling_idx, :],
            sampling_set_im2[sampling_idx, :], camera_matrix)

        f_mat_candidate = generate_f_matrix(essential_mat_candidate,
                                            camera_matrix)

        inlier_count_set1, _ = evaluate_fundamental_matrix(
            convert_img_to_homogenous(sampling_set_im1),
            convert_img_to_homogenous(sampling_set_im2), f_mat_candidate)

        inlier_count_set2, inliear_mask_candidate_set2 = evaluate_fundamental_matrix(
            test_set_im1, test_set_im2, f_mat_candidate)

        inlier_count = inlier_count_set1 + 1.0 * inlier_count_set2

        if (inlier_count > max_inlier_count):
            essential_mat = np.copy(essential_mat_candidate)
            inliear_mask = np.copy(inliear_mask_candidate_set2)
            #print('Found {} count instead of {}'.format(inlier_count, max_inlier_count))
            max_inlier_count = inlier_count

    return essential_mat, inliear_mask


def get_feature_points_with_matches(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    gray_im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray_im1, mask=gen_plant_mask(img1))
    kp2, des2 = sift.detectAndCompute(gray_im2, mask=gen_plant_mask(img2))

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    #bf = cv2.BFMatcher()
    matches = flann.knnMatch(des1, des2, k=2)

    return kp1, kp2, matches


def filter_matches_ratio_test(keypoints1,
                              keypoints2,
                              matches,
                              ratio_threshold=0.7):
    pts1 = []
    pts2 = []

    # Need to draw only good matches, so create a mask
    matches_mask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if ratio_threshold >= 0.9999 or m.distance < ratio_threshold * n.distance:
            matches_mask[i] = [1, 0]
            pts2.append(keypoints2[m.trainIdx].pt)
            pts1.append(keypoints1[m.queryIdx].pt)

    return np.array(pts1), np.array(pts2), matches_mask


def draw_matches(img1_gray, keypoints1, img2_gray, keypoints2, matches,
                 matches_mask):
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matches_mask,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_matches = cv2.drawMatchesKnn(img1_gray, keypoints1, img2_gray,
                                     keypoints2, matches, None, **draw_params)

    return img_matches


def modify_matches_mask(matches_mask, inliers):
    matches_new = copy.deepcopy(matches_mask)

    for i in range(len(matches_mask)):
        matches_new[i][0] = int(inliers[i])

    return matches_new


def generate_f_matrix(e_matrix, camera_matrix):
    return np.matmul(np.linalg.inv(np.transpose(camera_matrix)),
                     np.matmul(e_matrix, np.linalg.inv(camera_matrix)))


def get_matches_and_e(img1, img2):
    # get the point matches and the essential matrix between two images

    keypoints1, keypoints2, matches = get_feature_points_with_matches(
        img1, img2)

    keypoints1, keypoints2, matches = get_feature_points_with_matches(
        img1, img2)

    # perform ratio test and get points
    pts1_filtered, pts2_filtered, _ = filter_matches_ratio_test(
        keypoints1, keypoints2, matches, ratio_threshold=0.7)

    # perform ratio test and get points
    pts1_all, pts2_all, _ = filter_matches_ratio_test(keypoints1,
                                                      keypoints2,
                                                      matches,
                                                      ratio_threshold=1.0)

    ransac_test_pts1 = convert_img_to_homogenous(pts1_all)
    ransac_test_pts2 = convert_img_to_homogenous(pts2_all)

    if (len(pts1_filtered) < 8):
        #raise Exception("too few points being detected and passing ratio test")
        return None

    print("selected points with ratio: ", len(pts1_filtered))

    camera_matrix = load_camera_matrix()

    # custom ransac
    ransac_e_model_custom, inlier_custom = custom_ransac(
        pts1_filtered, pts2_filtered, ransac_test_pts1, ransac_test_pts2,
        camera_matrix)

    pts1_matches = pts1_all[inlier_custom, :]
    pts2_matches = pts2_all[inlier_custom, :]

    # decompose the essential matrix
    r1, r2, t = cv2.decomposeEssentialMat(ransac_e_model_custom)

    return pts1_matches, pts2_matches, ransac_e_model_custom, [r1, r2], t


def match_sift_double_iter_compare(img1, img2):
    keypoints1, keypoints2, matches = get_feature_points_with_matches(
        img1, img2)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # perform ratio test and get points
    pts1_filtered, pts2_filtered, matches_mask_filtered = filter_matches_ratio_test(
        keypoints1, keypoints2, matches, ratio_threshold=0.7)

    # perform ratio test and get points
    pts1_all, pts2_all, matches_mask_all = filter_matches_ratio_test(
        keypoints1, keypoints2, matches, ratio_threshold=1.0)

    ransac_test_pts1 = convert_img_to_homogenous(pts1_all)
    ransac_test_pts2 = convert_img_to_homogenous(pts2_all)

    if (len(pts1_filtered) < 8):
        #raise Exception("too few points being detected and passing ratio test")
        return None

    print("selected points with ratio: ", len(pts1_filtered))

    camera_matrix = load_camera_matrix()
    # perform RANSAC
    ransac_e_model_default, ransac_inlier_original = cv2.findEssentialMat(
        pts1_filtered, pts2_filtered, camera_matrix)

    # ransac_e_model_default, ransac_inlier_original = custom_ransac(pts1_all,
    #                                                  pts2_all, ransac_test_pts1, ransac_test_pts2,
    #                                                  camera_matrix)

    if ransac_e_model_default is None or ransac_e_model_default.shape[1] < 3:
        print("E estimation failed")
        return (None, None, None)

    ransac_e_model_default = ransac_e_model_default[0:3, :]
    _, inlier_default = evaluate_fundamental_matrix(
        ransac_test_pts1, ransac_test_pts2,
        generate_f_matrix(ransac_e_model_default, camera_matrix))

    # custom ransac
    ransac_e_model_custom, inlier_custom = custom_ransac(
        pts1_filtered, pts2_filtered, ransac_test_pts1, ransac_test_pts2,
        camera_matrix)

    print(ransac_e_model_custom)
    print(ransac_e_model_default)

    # get the default image
    default_img = draw_matches(img1_gray, keypoints1, img2_gray, keypoints2,
                               matches, matches_mask_filtered)
    cv2.putText(default_img, '{}'.format(len(pts1_filtered)), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    # get the default ransac inlier count
    matches_sub = []
    mask_sub = []
    for i in range(len(matches)):
        if (matches_mask_filtered[i][0] == 1):
            matches_sub.append(matches[i])
            mask_sub.append([1, 0])

    ransac_default_img_no_extra = draw_matches(
        img1_gray, keypoints1, img2_gray, keypoints2, matches_sub,
        modify_matches_mask(mask_sub, ransac_inlier_original))
    cv2.putText(ransac_default_img_no_extra,
                '{}'.format(np.asscalar(np.sum(ransac_inlier_original))),
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2,
                cv2.LINE_AA)

    # get the default ransac inlier count
    ransac_default_img = draw_matches(
        img1_gray, keypoints1, img2_gray, keypoints2, matches,
        modify_matches_mask(matches_mask_all, inlier_default))
    cv2.putText(ransac_default_img,
                '{}'.format(np.asscalar(np.sum(inlier_default))), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    # get the default ransac inlier count
    ransac_custom_img = draw_matches(
        img1_gray, keypoints1, img2_gray, keypoints2, matches,
        modify_matches_mask(matches_mask_all, inlier_custom))
    cv2.putText(ransac_custom_img,
                '{}'.format(np.asscalar(np.sum(inlier_custom))), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    return np.concatenate(
        (np.concatenate((default_img, ransac_default_img_no_extra), axis=1),
         np.concatenate((ransac_default_img, ransac_custom_img), axis=1)),
        axis=0)


def get_chessboard_points(inp_image):
    num_pts_x = 4
    num_pts_y = 4

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((num_pts_x * num_pts_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_pts_x, 0:num_pts_y].T.reshape(-1, 2)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(inp_image, (num_pts_x, num_pts_y),
                                             None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # if corners  is not None:
        #corners2 = cv2.cornerSubPix(inp_image, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(inp_image, (num_pts_x, num_pts_y),
                                        corners, ret)

        cv2.imwrite('./temp1.jpg', img)
    else:
        print("not found")

    cv2.destroyAllWindows()


def calibrate_camera(inp_image_folder):
    images = [
        cv2.imread(f) for f in glob.glob(os.path.join(inp_image_folder, '*.*'))
    ]
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    for img in gray_images:
        get_chessboard_points(img)


def load_camera_matrix():
    #fx = 357.5901
    #fy = 384.8436
    #cx = 230.8863
    #cy = 173.6757
    # return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    fx = 614
    fy = 614
    cx = 220.3939
    cy = 385.2142
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    #return np.array([[fy, 0, cy], [0, fx, cx], [0, 0, 1]])


def triangulate_points(proj_mat_1, proj_mat_2, im_pts_1, im_pts_2):
    return cv2.triangulatePoints(proj_mat_1, proj_mat_2, im_pts_1, im_pts_2)


def plot_3d_points(points_4d):
    pts3D = points_4d[:, :3] / np.repeat(points_4d[:, 3], 3).reshape(-1, 3)

    # plot with matplotlib
    Ys = pts3D[:, 0]
    Zs = pts3D[:, 1]
    Xs = pts3D[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xs, Ys, Zs, c='r', marker='o')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    plt.title('3D point cloud: Use pan axes button below to inspect')
    plt.show()


def load_image(imName, path='../data/lettuce_home/'):
    img = cv2.imread(path + imName)

    return img
