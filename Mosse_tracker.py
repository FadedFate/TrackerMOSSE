from __future__ import absolute_import, division, print_function
import numpy as np 
import cv2
from ops import read_image
from ops import create_labels
from ops import pre_pos
from ops import rand_warp
from ops import box_deter
from got10k.trackers import Tracker
import time
import glob

__all__ = ['Mosse_tracker']

class TrackerMOSSE(Tracker):
    def __init__(self, eta = 0.125 , sigma = 100 , N=128):
        super(TrackerMOSSE, self).__init__('MOSSE', True)
        self.eta = eta
        self.sigma = sigma
        self.N = N

    def init(self, img, box):
        # print(box)
        box = np.array([
            box[1] ,
            box[0] ,
            box[3], box[2]], dtype=np.float32)
        img_now = read_image(img)
        self.box = box_deter(box,img_now)
        center = np.array([box[1] + box[3]/2 , box[0] + box[2]/2] )
        gsize = img_now.shape
        x = np.arange(gsize[1])
        y = np.arange(gsize[0])
        x_index , y_index = np.meshgrid(x, y)
        label_now = create_labels(x_index,y_index,self.sigma,center)

        if img_now.ndim == 3:
            img_now = cv2.cvtColor(img_now, cv2.COLOR_RGB2GRAY)
        img_s = img_now[int(box[0]) : int(box[0] + box[2]) , int(box[1]) : int(box[1] + box[3])]
        img_g = label_now[int(box[0]) : int(box[0] + box[2]) , int(box[1]) : int(box[1] + box[3])]
        # print(img_g.shape)
        self.img_G = np.fft.fft2(img_g)
        img_fi = cv2.resize(img_s ,self.img_G.shape[::-1])  # width  height

        img_fi = pre_pos(img_fi)
        self.Ai = (self.img_G*(np.fft.fft2(img_fi).conj()))
        self.Bi = (np.fft.fft2(img_fi)*(np.fft.fft2(img_fi).conj()))
    
        for i in range(0,self.N):
            img_fi= pre_pos(rand_warp(img_s))   
            self.Ai = self.Ai + (self.img_G*(np.fft.fft2(img_fi).conj()))
            self.Bi = self.Bi + (np.fft.fft2(img_fi)*(np.fft.fft2(img_fi).conj()))
        
        self.Ai = self.eta*self.Ai
        self.Bi = self.eta*self.Bi

    def update(self, img ):
        img_ori = read_image(img)
        img_now = img_ori
        if img_now.ndim == 3:
            img_now = cv2.cvtColor(img_now, cv2.COLOR_RGB2GRAY)
        else:
            img_now = img_ori
        
        Hi = self.Ai/self.Bi
        img_fi = img_now[int(self.box[0]) : int(self.box[0] + self.box[2]) , int(self.box[1]) : int(self.box[1] + self.box[3])]          
        img_fi = pre_pos(cv2.resize(img_fi, self.img_G.shape[::-1])) 
        # print(img_fi.shape)
        img_gi = np.abs(np.fft.ifft2(Hi * np.fft.fft2(img_fi)))
        img_gik = np.zeros(img_gi.shape)
        img_gik = cv2.normalize(img_gi , img_gik, 0, 255, cv2.NORM_MINMAX)
        img_gi = np.asarray(img_gik, dtype=np.uint8)
        # print(img_gi)

        gi_max_1 , gi_max_2 = np.where(img_gi == np.amax(img_gi))
        loc = np.array((gi_max_1.mean() , gi_max_2.mean()))
        # print(loc)
        dheight = loc[0] - self.img_G.shape[0]/2 
        dwidth = loc[1] - self.img_G.shape[1]/2 
        # print(box)
        box = np.array([self.box[0]+dheight , self.box[1]+dwidth ,  self.box[2], self.box[3]])
        box = box_deter(box,img_now)
        # print(box)
        # print(img_now.shape)
        self.box = box
        img_fi = img_now[int(box[0]) : int(box[0] + box[2]) , int(box[1]) : int(box[1] + box[3])]       
        img_fi = pre_pos(cv2.resize(img_fi, self.img_G.shape[::-1])) 
        self.Ai = self.eta*(self.img_G*(np.fft.fft2(img_fi).conj())) + (1-self.eta)*self.Ai
        self.Bi = self.eta*(np.fft.fft2(img_fi)*(np.fft.fft2(img_fi).conj())) + (1-self.eta)*self.Bi

        box = np.array([
            box[1] , box[0] , box[3] , box[2]],dtype = np.float32)
        return box

    def track(self, img_files, box , visualize = False) :
        frame_num = len(img_files)

        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)
        for f, img_file in enumerate(img_files):
            begin = time.time()
            if f == 0:
                self.init(img_files[0] , box)
            else:
                boxes[f, :] = self.update(img_file) 
            times[f] = time.time() - begin  
        return boxes, times

