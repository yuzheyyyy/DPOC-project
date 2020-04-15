import skimage.io as io
import cv2
import numpy as np
import math
import os

# Path of parameter file
dirpath_parm = '../Raw Data/2011_09_26_cali/calib_cam_to_cam.txt'
# Path of Raw Image
dirpath_img = ['../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/',
                '../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_01/data/',
                '../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data',
                '../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/']
# Keyword for getting extrinsic parameters
keyword = [['R_00:', 'T_00:', 'K_00:', 'D_00:', 'R_rect_00:'],
           ['R_01:', 'T_01:', 'K_01:', 'D_01:', 'R_rect_01:'],
           ['R_02:', 'T_02:', 'K_02:', 'D_02:', 'R_rect_02:'],
           ['R_03:', 'T_03:', 'K_03:', 'D_03:', 'R_rect_03:']]
# Initialization for parameters
R = np.zeros([3, 3, 4])
K = np.zeros([3, 3, 4])
R_rec = np.zeros([3, 3, 4])
T = np.zeros([3, 4])
D = np.zeros([5, 4])


class CAM():
    def __init__(self, i):
        R, K, T, D, R_rec = self.read_exparam()
        img_list = self.read_img()
        self.CamMatrix = K[:, :, i]
        self.Translation = T[:, i].reshape([3, 1])
        self.Rotation = np.dot(R[:, :, i], R_rec[:, :, i])
        self.DistCoef = 0
        self.img_list = img_list[i]
        self.shape = cv2.imread(self.img_list[1]).shape[:2]
        self.size = len(self.img_list)

    def read_exparam(self):
        """
        Read raw parameters for both intrinsic and extrinsic parameters
        :return:
        """
        f = open(dirpath_parm, 'r')
        lines = f.readlines()
        for line in lines:
            for i in range(4):
                if keyword[i][0] in line:
                    line = line.lstrip(keyword[i][0])
                    R[:, :, i] = np.array(list(map(float, line.split()))).reshape((3, 3))
                elif keyword[i][1] in line:
                    line = line.lstrip(keyword[i][1])
                    T[:, i] = np.array(list(map(float, line.split())))
                elif keyword[i][2] in line:
                    line = line.lstrip(keyword[i][2])
                    K[:, :, i] = np.array(list(map(float, line.split()))).reshape((3, 3))
                elif keyword[i][3] in line:
                    line = line.lstrip(keyword[i][3])
                    D[:, i] = np.array(list(map(float, line.split())))
                elif keyword[i][4] in line:
                    line = line.lstrip(keyword[i][4])
                    R_rec[:, :, i] = np.array(list(map(float, line.split()))).reshape((3, 3))
        return R, K, T, D, R_rec

    def read_img(self):
        """
        Read Raw Image
        :return:
        """
        cam0, cam1, cam2, cam3 = [], [], [], []
        file_cam0 = sorted(os.listdir(dirpath_img[0]))
        file_cam1 = sorted(os.listdir(dirpath_img[1]))
        file_cam2 = sorted(os.listdir(dirpath_img[2]))
        file_cam3 = sorted(os.listdir(dirpath_img[3]))
        for file in file_cam0:
            cam0.append(os.path.join(dirpath_img[0], file))
        for file in file_cam1:
            cam0.append(os.path.join(dirpath_img[1], file))
        for file in file_cam2:
            cam2.append(os.path.join(dirpath_img[2], file))
        for file in file_cam3:
            cam3.append(os.path.join(dirpath_img[3], file))
        return [cam0, cam1, cam2, cam3]


def rand_transformation(rot, trans, R_threshold, T_threshold, ROTATION=True, TRANSLATION=True):
    """
    Fixed cam2, and randomly change rotation and translation of cam3
    :param rot: raw rotation matrix needed to be changed
    :param trans: raw translation vector needed to be changed
    :param R_threshold:
    :param T_threshold:
    :param ROTATION: whether change rotation
    :param TRANSLATION: whether change translation
    :return:
    """
    if ROTATION:
        added_rot = R_threshold * np.random.rand(3, 1)
        rot_vec = cv2.Rodrigues(rot)[0]
        new_rot_vec = rot_vec + added_rot
        rot = cv2.Rodrigues(new_rot_vec)[0]
    if TRANSLATION:
        added_trans = T_threshold * np.random.rand(3, 1)
        trans = trans + added_trans
    return rot, trans, added_rot, added_trans




