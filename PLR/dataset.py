import data_read
import numpy as np
import cv2
from numpy.linalg import inv
import math
import numba
import warnings

# Ignore Warnings
warnings.filterwarnings('ignore')


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


def create_mask(image):
    '''
    Create a mask for empty pixels in image
    Resulting mask has black (0) for non-empty pixels
    '''
    # define range of empty pixels
    lower_val = np.array([0,0,0])
    upper_val = np.array([0,0,0])

    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(image, lower_val, upper_val)
    return mask


@numba.jit()
def crop_largest_rectangle_with_original_ratio(image):
    '''
        Idea Adapted from the link:
        https://stackoverflow.com/questions/7332065/find-the-largest-convex-black-area-in-an-image
    '''
    mask = create_mask(image)

    # storing max consecutive non-empty pixels
    h = np.zeros((image.shape[0], image.shape[1],))

    # check maximum consecutive pixels, column by column, from bottom to top
    for j in range(mask.shape[1]):
        consecutive = 0
        for i in range(mask.shape[0] - 1, -1, -1):
            # if the pixel is empty, continue to next
            if mask[i, j] == 255:
                consecutive = 0
            else:
                consecutive += 1
            h[i, j] = consecutive
    # storing max width and area found for each pixel
    w = np.zeros((image.shape[0], image.shape[1],))
    a = np.zeros((image.shape[0], image.shape[1],))
    # check consecutive rows with higher
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 255:
                continue
            current_height = h[i, j]
            current_width = 1
            while h[i, j + current_width - 1] >= current_height:
                current_width += 1
                if j + current_width - 1 >= mask.shape[1]:
                    break
            w[i, j] = current_width
            a[i, j] = current_height * current_width

    # index of top left corner of the rectangle found with max area
    index = np.unravel_index(a.argmax(), a.shape)

    # keep original image width:height ratio
    if h[index] / mask.shape[0] * mask.shape[1] > w[index]:
        # rectangle taller than original image
        # shift the height down to make the cropped region centered
        shift = int((h[index] - w[index] / mask.shape[1] * mask.shape[0]) / 2)
        return (index[0] + shift, index[1],
                int(w[index] / mask.shape[1] * mask.shape[0]), int(w[index]))
    # rectangle wider than original image
    shift = int((w[index] - h[index] / mask.shape[0] * mask.shape[1]) / 2)
    return (index[0], index[1] + shift,
            int(h[index]), int(h[index] / mask.shape[0] * mask.shape[1]))


@numba.jit()
def find_largest_rectangle(img1, img2, SHOW=True):
    roi1 = crop_largest_rectangle_with_original_ratio(img1)
    roi2 = crop_largest_rectangle_with_original_ratio(img2)
    if roi1[3] < roi2[3]:
        index1 = (roi1[0], roi1[1])
        shift_h = int((roi2[2] - roi1[2]) / 2)
        shift_w = int((roi2[3] - roi1[3]) / 2)
        h = roi1[2]
        w = roi1[3]
        index2 = (roi2[0] + shift_h, roi2[1] + shift_w)
    else:
        index2 = (roi2[0], roi2[1])
        shift_h = int((roi1[2] - roi2[2]) / 2)
        shift_w = int((roi1[3] - roi2[3]) / 2)
        h = roi2[2]
        w = roi2[3]
        index1 = (roi1[0] + shift_h, roi2[1] + shift_w)

    index_botright_1 = (int(index1[0] + h), int(index1[1] + w))
    index_botright_2 = (int(index2[0] + h), int(index2[1] + w))

    size = (img1.shape[1], img1.shape[0])
    out1 = cv2.resize(img1[index1[0]:index_botright_1[0],
                      index1[1]:index_botright_1[1]], size)
    out2 = cv2.resize(img2[index2[0]:index_botright_2[0],
                      index2[1]:index_botright_2[1]], size)
    # display
    if SHOW:
        demo1 = cv2.rectangle(img1, index1[::-1],
                              index_botright_1[::-1],
                              (0, 255, 0), thickness=1)
        demo2 = cv2.rectangle(img2, index2[::-1],
                              index_botright_2[::-1],
                              (0, 255, 0), thickness=1)
        comparison = np.concatenate([demo1, demo2], axis=1)
        cv2.imwrite('crop_demo.png', comparison)
        cv2.imshow("Visualization of Largest Rectangle Found", comparison)
        cv2.waitKey(0)
        selected = np.concatenate([out1, out2], axis=1)
        cv2.imshow("Selected Region", selected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out1, out2


def img_rectify(img1, img2, Camera2, Camera3, R_diff, T_diff):
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(cameraMatrix1=Camera2.CamMatrix, cameraMatrix2=Camera3.CamMatrix,
                                                distCoeffs1=Camera2.DistCoef, distCoeffs2=Camera3.DistCoef,
                                                R=R_diff, T=T_diff, imageSize=Camera2.shape,)
    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix=Camera2.CamMatrix, distCoeffs=Camera2.DistCoef, R=R1,
                                               newCameraMatrix=P1, size=Camera2.shape[::-1], m1type=cv2.CV_32F)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix=Camera3.CamMatrix, distCoeffs=Camera3.DistCoef, R=R2,
                                               newCameraMatrix=P2, size=Camera3.shape[::-1], m1type=cv2.CV_32F)
    img_rec = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img_rec2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    return find_largest_rectangle(img_rec, img_rec2, SHOW=False)


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
    weight_path = '../Weight/'

    # Weight for label
    Weight = np.array([[0.082], [3.704], [3.72], [1.0], [1.0], [1.0]])

    # Get camera parameter
    Cam2 = data_read.CAM(2)
    Cam3 = data_read.CAM(3)

    # FILE FOR SAVING RECTIFIED INFORMATION
    file = open(TRAINPATH_TXT, 'a')  # Write rotation and translation information

    # Iterations for getting datasets
    for i in range(Cam2.size):
        imgL = cv2.imread(Cam2.img_list[i])
        imgR = cv2.imread(Cam3.img_list[i])
        for j in ITER:
            # Get Random Rotation and Translation
            R_rec, T_rec, Rectified_rot, Rectified_trans = data_read.rand_transformation(
                Cam3.Rotation, Cam3.Translation, 0.05, 0.1, ROTATION=True, TRANSLATION=True)
            R_diff = np.dot(inv(Cam2.Rotation), R_rec)
            T_diff = Cam2.Translation - T_rec

            # Get rectified two images
            imgL_rec, imgR_rec = img_rectify(img1=imgL, img2=imgR, Camera2=Cam2, Camera3=Cam3, R_diff=R_diff, T_diff=T_diff)

            # SAVE IMG
            cv2.imwrite(TRAINPATH_IMG + 'image02_' + int2str(ITER * i + j, MIN_LEN) + '.png', imgL_rec)
            cv2.imwrite(TRAINPATH_IMG + 'image03_' + int2str(ITER * i + j, MIN_LEN) + '.png', imgR_rec)

            # Calculate rectified information
            RecInfo = np.sum(Rectified_rot * Weight[3:6] / 0.05 + Rectified_trans * Weight[:3] / 0.1)

            # SAVE LABEL
            np.savetxt(file, RecInfo, fmt='%.5f')
    file.close()








