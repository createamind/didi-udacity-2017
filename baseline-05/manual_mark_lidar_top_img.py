# https://aspratyush.wordpress.com/2013/12/22/ginput-in-opencv-python/

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import cv2
import numpy as np
import os
import glob

a = []

TOP_Y_MIN = -20  # 40
TOP_Y_MAX = +20
TOP_X_MIN = -20
TOP_X_MAX = +20  # 70.4
TOP_Z_MIN = -2.0  ###<todo> determine the correct values!
TOP_Z_MAX = 0.4

TOP_X_STEP = 0.1  # 0.1
TOP_Y_STEP = 0.1
TOP_Z_STEP = 0.4


def top_to_lidar_coords(xx, yy):
    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_STEP) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_STEP) + 1
    y = Xn * TOP_Y_STEP - (xx + 0.5) * TOP_Y_STEP + TOP_Y_MIN
    x = Yn * TOP_X_STEP - (yy + 0.5) * TOP_X_STEP + TOP_X_MIN

    return x, y


def click_xy(image, name='IMG', resize=1, callback=None, callback_params=None):
    """
    get the user input to set the box
    
    :param image: 
    :param name: 
    :param resize: 
    :param callback: 
    :param callback_params: 
    :return: 
    """
    global a

    img = np.copy(image)
    H, W = img.shape[0:2]

    # define the event
    def getxy(event, x, y, flags, params):
        global a
        if event == cv2.EVENT_LBUTTONDOWN:
            a.append(np.array([x, y], np.int32))
            if callback is not None:
                callback(img, int(x), int(y), callback_params)
            else:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
                if len(a) == 2:
                    cv2.rectangle(img, (a[0][0], a[0][1]), (a[1][0], a[1][1]), (0, 0, 255), thickness=1, lineType=8,
                                  shift=0)
            cv2.imshow(name, img)
            print("(x, y) = ", (x, y))

    # Set mouse CallBack event
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(name, getxy)

    # show the image
    print("Click to select a point OR press ANY KEY to continue...")
    cv2.resizeWindow(name, round(resize * W), round(resize * H))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # obtain the matrix of the selected points
    print('')
    print('clicked points:')
    boxes3d = np.zeros([1, 8, 3], dtype=np.float32)
    if len(a) != 0:
        b = np.array(a)
        print(b, b.dtype, b.shape)

        a = []
        if b.shape[0] == 2 and b.shape[1] == 2:
            x1, y1, x2, y2 = b[0][0], b[0][1], b[1][0], b[1][1]
            points = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
            for k in range(4):
                xx, yy = points[k]
                x, y = top_to_lidar_coords(xx, yy)
                boxes3d[0, k, :] = x, y, -2  ## <todo>
                boxes3d[0, k + 4, :] = x, y, 0.4

            return boxes3d
        else:
            return boxes3d
    else:
        print('None')
        return boxes3d


def label_box_in_lidar_top_view(lidar_top_img_path, box_label_path):
    """
    manually mark boxes on objects in lidar top view images
    
    :param lidar_top_img_path: full path of the lidar top view images folder
    :param box_label_path: full path of the marked box data file folder
    :return: None
    """
    os.makedirs(box_label_path, exist_ok=True)

    failed = list()
    for ndx, file in enumerate(sorted(glob.glob(lidar_top_img_path + '/*.png'))):
        name = os.path.basename(file).replace('.png', '')
        print('processing file: %s.png' % name)

        img = cv2.imread(file, 1)
        m = click_xy(img, resize=2)

        np.save(box_label_path + '/' + name + '.npy', m)
        

# main --------------------------------------------------------------------------
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # img = cv2.imread('/mnt/dataset/temp/ark-04/data/lidar_top_img/00001.png', 1)
    # m = click_xy(img, resize=2)

    lidar_top_img_path = '/mnt/dataset/temp/ark-04/data/lidar_top_img'
    box_label_path = '/mnt/dataset/temp/ark-04/data/gt_boxes3d'

    label_box_in_lidar_top_view(lidar_top_img_path, box_label_path)

    print('Done!')
