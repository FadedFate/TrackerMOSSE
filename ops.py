import cv2
import numpy as np
import random
def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img

def create_labels(x , y , sigma , center):
    xc = center[0]
    yc = center[1]
    expp = ((x-xc)**2 + (y-yc)**2)/(2*sigma)
    return np.exp(-expp)

def pre_pos(img):
    win = np.outer(np.hanning(img.shape[0]) , np.hanning(img.shape[1]))   
    
    eps = 1e-5
    img = img.astype(np.float32)
    img = np.log(img+1)
    img = (img-np.mean(img)/(np.std(img+eps)))
    img = img * win
    return img

def rand_warp(img):
    a = -180/16
    b = 180/16
    r =  a + (b-a)*random.random()
    sz = img.shape
    scale = 1 + 0.1*random.random()
    border = int(img.mean())
    center = np.ones(2) + np.array(sz , dtype = int) // 2
    M = cv2.getRotationMatrix2D(tuple(center)[::-1], r, 1)
    img_warp = cv2.warpAffine(img , M , sz[::-1] , borderValue = border)
    img_warp = cv2.resize(img_warp , (int(scale * sz[1]) , int(scale * sz[0])))

    sz_new = img_warp.shape
    img_warp = img_warp[sz_new[0]//2 - sz[0] // 2 : sz_new[0]//2 - sz[0] // 2 + sz[0] , sz_new[1]//2 - sz[1] // 2 : sz_new[1]//2 - sz[1] // 2 + sz[1]  ]
    return img_warp
    # img = imresize(imresize(imrotate(img, r), scale), [sz(1) sz(2)])

def box_deter(box , img_now):
    if box[0] < 0:
        box[0] = 0
    if box[1] < 0 :
        box[1] = 0
    if box[0] + box[2] + 1> img_now.shape[0]:
        box[0] =  img_now.shape[0] - box[2] - 1
    if box[1] + box[3] + 1> img_now.shape[1]:
        box[1] =  img_now.shape[1] - box[3] - 1  
    return box
