import data_read
import numpy as np
import cv2
from numpy.linalg import inv
import math



def get_homography(R1, R2, K1, K2):
    """
    Get Hmography Matrix according to Rotation of imaging plane
    :param R_rec:
    :param R1: Rotation of cam2
    :param R2: Rotation of cam3
    :param K1: intrinsic parameter of cam2
    :param K2: intrinsic parameter of cam3
    :return:
    """
    h1 = np.dot(np.dot(K1, R1), np.linalg.inv(K1))
    h2 = np.dot(np.dot(K2, R2), np.linalg.inv(K2))
    return [h1, h2]


def img_rectify(im1, im2, rot1, rot2, trans1, trans2, dist1, dist2):
    """
    Get rectified image using cv2.warpPerspective
    :param im1: raw image of cam2
    :param im2: raw image of cam3
    :param R1: homography matrix of cam2
    :param R2: homography matrix of cam3
    :param T1: translation of cam2
    :param T2: translation of cam3
    :return:
    """
    R = np.dot(inv(rot1), rot2)
    T = trans1 - trans2
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(rot1, dist1, rot2, dist2, np.shape(im1[:, :, 0]), R, T)
    [h1, h2] = get_homography(R1, R2, K[:, :, 2], K[:, :, 3])
    img1 = cv2.warpPerspective(im1, h1, (HEIGHT, WIDTH))
    img2 = cv2.warpPerspective(im2, h2, (HEIGHT, WIDTH))
    return img1, img2


def int2str(num, min_len):
    """
    transfer from int to string, and set format for saving files
    :param num:
    :param min_len:
    :return:
    """
    s = str(num)
    while len(s) < min_len: s = '0'+s
    return s


if __name__ == '__main__':

    ITER = 1  # Number of random sampling for rotation and translation per image
    BOOL = True  # Bool for judging whether randomly rotate or translate
    MIN_LEN = 9
    TRAINPATH_IMG = '../Training/Image/'
    TRAINPATH_TXT = '../Training/rectified information.txt'
    TESTPATH_IMG = '../Test/Image/'
    TESTPATH_TXT = '../Test/rectified information.txt'
    VALIPATH_IMG = '../Validation/Image/'
    VALIPATH_TXT = '../Validation/rectified information.txt'
    img_rec = []  # List for storing rectified images
    weight_path = '../Weight/'

    # Get camera parameter
    Cam2 = data_read.CAM(2)
    Cam3 = data_read.CAM(3)
    # R_rec, T_rec = data_read.rand_transformation(Cam3.Rotation, Cam3.Translation, 0.05, 0.1, ROTATION=False, TRANSLATION=True)
    changed_vec_rot = np.array([[0.0], [0.0], [0.1]])
    R_rec = cv2.Rodrigues(cv2.Rodrigues(Cam3.Rotation)[0] + changed_vec_rot)[0]
    R_diff1 = np.dot(np.linalg.inv(Cam2.Rotation), R_rec)
    R_diff2 = np.dot(np.linalg.inv(Cam2.Rotation), Cam3.Rotation)
    # R_diff = np.dot(inv(Cam2.Rotation), Cam3.Rotation)

    changed_vec_trans = np.array([[0.0], [0.1], [0.0]])
    T_rec = Cam3.Translation + changed_vec_trans
    # T_diff = Cam2.Translation - Cam3.Translation
    T_diff1 = Cam2.Translation - T_rec
    T_diff2 = Cam2.Translation - Cam3.Translation

    # Calculate rectified information
    # RecRotationNorm = np.linalg.norm(cv2.Rodrigues(np.dot(np.linalg.inv(Cam3.Rotation), R_rec))[0])/(0.05 * math.sqrt(3))
    # RecTranslationNorm = np.linalg.norm(T_rec - Cam3.Translation)/(0.1 * math.sqrt(3))
    # print(T_rec - Cam3.Translation)
    # print(RecTranslationNorm)

    R1, R2, P1, P2, _, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=Cam2.CamMatrix, cameraMatrix2=Cam3.CamMatrix,
                                                distCoeffs1=Cam2.DistCoef, distCoeffs2=Cam3.DistCoef,
                                                R=R_diff1, T=T_diff1, imageSize=Cam2.shape, )

    R3, R4, P3, P4, _, _, _ = cv2.stereoRectify(cameraMatrix1=Cam2.CamMatrix, cameraMatrix2=Cam3.CamMatrix,
                                                      distCoeffs1=Cam2.DistCoef, distCoeffs2=Cam3.DistCoef,
                                                      R=R_diff2, T=T_diff2, imageSize=Cam2.shape)


    print('rotation_vec1', cv2.Rodrigues(np.dot(inv(R1), R3))[0])

    # print('rotation_vec2', cv2.Rodrigues(np.dot(inv(R2), R4))[0])

    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix=Cam2.CamMatrix, distCoeffs=Cam2.DistCoef, R=R1,
                                               newCameraMatrix=P1, size=Cam2.shape[::-1], m1type=cv2.CV_32F)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix=Cam2.CamMatrix, distCoeffs=Cam2.DistCoef, R=R3,
                                               newCameraMatrix=P3, size=Cam3.shape[::-1], m1type=cv2.CV_32F)
    img = cv2.imread(Cam2.img_list[1])
    img2 = cv2.imread(Cam3.img_list[1])
    img_rec = cv2.remap(img, map1x, map1y, cv2.INTER_LINEAR)

    cv2.imshow('img_rec', img_rec)
    cv2.waitKey(0)
    # cv2.imwrite(weight_path + 'raw.png', img_rec)


    # R3, R4, P3, P4, _, _, _ = cv2.stereoRectify(cameraMatrix1=Cam2.CamMatrix, cameraMatrix2=Cam3.CamMatrix,
    #                                                   distCoeffs1=Cam2.DistCoef, distCoeffs2=Cam3.DistCoef,
    #                                                   R=R_diff2, T=T_diff, imageSize=Cam2.shape)

    #
    # map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix=Cam2.CamMatrix, distCoeffs=Cam2.DistCoef, R=R1,
    #                                            newCameraMatrix=P1, size=Cam2.shape[::-1], m1type=cv2.CV_32F)
    #
    # map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix=Cam3.CamMatrix, distCoeffs=Cam3.DistCoef, R=R2,
    #                                            newCameraMatrix=P2, size=Cam3.shape[::-1], m1type=cv2.CV_32F)
    #
    # img = cv2.imread(Cam2.img_list[1])
    #
    # img_rec = cv2.remap(img, map1x, map1y, cv2.INTER_LINEAR)
    # # img_rec2 = img_rec[y: y + h, x: x + w, :]
    # cv2.imshow('img_rec', img_rec)
    # # cv2.imshow('img_rec2', img_rec2)
    # cv2.waitKey(0)
    # # print(np.shape(img_rec))


    # map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix=K[:, :, 3], distCoeffs=D[:, 3], R=R2,
    #                                            newCameraMatrix=P2, size=(1500, 700), m1type=cv2.CV_32F)
    # print(roi1)
    # print(roi2)
    # img_rec = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    # img_rec2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    # img_rec_roi = img_rec[roi1[0]:roi1[0] + roi1[2], roi1[1]:roi1[1] + roi1[3], :]
    # img_rec_roi2 = img_rec2[roi2[0]:roi2[0] + roi2[2], roi2[1]:roi2[1] + roi2[3], :]
    # cv2.imshow('raw', img_rec)
    # cv2.imshow('s', img_rec_roi)
    # cv2.imshow('s2', img_rec_roi2)
    # cv2.waitKey(0)

    # Get threshold for random sampling
    # [R_thresh, T_thresh] = data_read.set_threshold(R, T)

    # For relative difference
    # R_diff = np.dot(np.linalg.inv(R[:, :, 2]), R[:, :, 3])

    # print(R_diff_norm)
    # T_diff = np.dot(np.linalg.inv(R[:, :, 2]), T[:, 2] - T[:, 3])

    # path = '/home/open/eth/2020spring/PLR/Raw Data/2011_09_26/2011_09_26_drive_0001_extract/image_02/data/0000000006.png'
    # img = cv2.imread(path)
    # height, width = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K[:, :, 2], D[:, 2], (width, height), 1, (width, height))
    # dst = cv2.undistort(img, K[:, :, 2], D[:, 2], None, newcameramtx)
    # x, y, w, h = roi
    # print(roi)
    # dst_roi = dst[y:y+h, x:x+w, :]
    # cv2.imshow('dst', dst)
    # cv2.imshow('dst_roi', dst_roi)
    # cv2.waitKey(0)

    # path1 = '/home/open/eth/2020spring/PLR/Raw Data/2011_09_26/2011_09_26_drive_0001_extract/image_02/data/0000000000.png'
    # path2 = '/home/open/eth/2020spring/PLR/Raw Data/2011_09_26/2011_09_26_drive_0001_extract/image_03/data/0000000000.png'
    # img1 = cv2.imread(path1)
    # img2 = cv2.imread(path2)
    # # R_diff = np.dot(np.linalg.inv(R[:, :, 2]), R[:, :, 3])
    # # T_diff = np.dot(np.linalg.inv(R[:, :, 2]), T[:, 2] - T[:, 3])
    # [R_rand, T_rand] = data_read.rand_transformation(R[:, :, 3], T[:, 3], R_thresh, T_thresh, BOOL, BOOL)
    # R_diff = np.dot(np.linalg.inv(R[:, :, 2]), R_rand)
    # T_diff = T[:, 2] - T_rand
    # R1, R2, P1, P2, _, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=K[:, :, 2], cameraMatrix2=K[:, :, 3],
    #                                             distCoeffs1=D[:, 2], distCoeffs2=D[:, 3],
    #                                             R=R_diff, T=T_diff, imageSize=np.shape(img1[:, :, 2]), alpha=0)
    #
    # map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix=K[:, :, 2], distCoeffs=D[:, 2], R=R1,
    #                                            newCameraMatrix=P1, size=(1500, 700), m1type=cv2.CV_32F)
    # map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix=K[:, :, 3], distCoeffs=D[:, 3], R=R2,
    #                                            newCameraMatrix=P2, size=(1500, 700), m1type=cv2.CV_32F)
    # print(roi1)
    # print(roi2)
    # img_rec = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    # img_rec2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    # img_rec_roi = img_rec[roi1[0]:roi1[0] + roi1[2], roi1[1]:roi1[1] + roi1[3], :]
    # img_rec_roi2 = img_rec2[roi2[0]:roi2[0] + roi2[2], roi2[1]:roi2[1] + roi2[3], :]
    # cv2.imshow('raw', img_rec)
    # cv2.imshow('s', img_rec_roi)
    # cv2.imshow('s2', img_rec_roi2)
    # cv2.waitKey(0)


    # # Iteration for getting random dataset
    # img_len = len(img[0])
    # file = open(VALIPATH_TXT, 'a')
    # # file = open(TESTPATH_TXT, 'a')
    # # file = open(TRAINPATH_TXT, 'a')  # Write rotation and translation information
    # for i in range(img_len):
    #     for j in range(ITER):
    #         [R_rand, T_rand] = data_read.rand_transformation(R[:, :, 3], T[:, 3], R_thresh, T_thresh, BOOL, BOOL)
    #         img_rec.append(img_rectify(img[2][i], img[3][i], R[:, :, 2], R_rand, T[:, 2], T_rand, D[:, 2], D[:, 3]))
    #         # Save rectified image
    #         # cv2.imwrite(TRAINPATH_IMG + 'image02_' + int2str(ITER * i + j, MIN_LEN) + '.png', img_rec[ITER * i + j][0])
    #         # cv2.imwrite(TRAINPATH_IMG + 'image03_' + int2str(ITER * i + j, MIN_LEN) + '.png', img_rec[ITER * i + j][1])
    #         # cv2.imwrite(TESTPATH_IMG + 'image02_' + int2str(ITER * i + j, MIN_LEN) + '.png', img_rec[ITER * i + j][0])
    #         # cv2.imwrite(TESTPATH_IMG + 'image03_' + int2str(ITER * i + j, MIN_LEN) + '.png', img_rec[ITER * i + j][1])
    #         cv2.imwrite(VALIPATH_IMG + 'image02_' + int2str(ITER * i + j, MIN_LEN) + '.png', img_rec[ITER * i + j][0])
    #         cv2.imwrite(VALIPATH_IMG + 'image03_' + int2str(ITER * i + j, MIN_LEN) + '.png', img_rec[ITER * i + j][1])
    #
    #         # Save rectified rotation and translation
    #         R_rec = float('%.9e' % np.linalg.norm(cv2.Rodrigues(np.dot(inv(R[:, :, 3]), R_rand))[0]))
    #         T_rec = float('%.9e' % np.linalg.norm(T[:, 3] - T_rand))
    #         R_relative = R_rec / R_diff_norm
    #         T_relative = T_rec / T_diff_norm
    #
    #         image2_name = 'image02_' + int2str(ITER * i + j, MIN_LEN) + '.png'
    #         image3_name = 'image03_' + int2str(ITER * i + j, MIN_LEN) + '.png'
    #         savefile = np.array([image2_name, image3_name, R_relative, T_relative]).reshape([1, 4])
    #         np.savetxt(file, savefile, fmt='%s')
    # file.close()
