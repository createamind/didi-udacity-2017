#https://aspratyush.wordpress.com/2013/12/22/ginput-in-opencv-python/

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import cv2
import numpy as np
import os

a = []
def click_xy(image, name='IMG', resize=1, callback=None, callback_params=None):
    global a

    img  = np.copy(image)
    H,W  = img.shape[0:2]


    #define the event
    def getxy(event, x, y, flags, params):
        global a
        if event == cv2.EVENT_LBUTTONDOWN :
            a.append(np.array([x,y],np.int32))
            if callback is not None:
                callback(img, int(x), int(y), callback_params)
            else:
                cv2.circle(img,(int(x), int(y)),2,(0,0,255),-1)
            cv2.imshow(name, img)
            print ("(x, y) = ", (x,y))


    #Set mouse CallBack event
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(name, getxy)

    #show the image
    print ("Click to select a point OR press ANY KEY to continue...")
    cv2.resizeWindow(name, round(resize*W), round(resize*H))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #obtain the matrix of the selected points
    print ('')
    print ('clicked points:')
    if len(a)!= 0:
        b = np.array(a)
        print (b, b.dtype, b.shape)
        a = []
        return b
    else:
        print ('None')
        return None


# main --------------------------------------------------------------------------
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    img=cv2.imread('/root/share/project/didi/data/didi/didi-2/Out/1/15/processed/lidar_top_img/1490991818039206000.png',1)
    m = click_xy(img,resize=2)

    exit()
