#encoding:utf-8
import segmentation_models as sm
import cv2
import os
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
import numpy as np
from math import ceil
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model = sm.Unet("resnet18", classes=1,activation="sigmoid",encoder_weights=None)
print(model.summary())
model.load_weights('Unet-EMA-Soft-resnet18-focal-pretrained-bs8-newaug-crop-aspect-reflect-color-full400.hdf5')
test_path = "test-cracks500croped"
save_path = "predicted400full"
i=0
print(test_path+'/*.jpg')
for fn in glob(test_path+'/*.jpg',recursive=True):
    i+=1
    for coef in [10]:#[5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60]:
        img = cv2.imread(fn)
        if img is None: continue
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,None,fx=coef/10,fy=coef/10,interpolation=cv2.INTER_LANCZOS4)
        H, W=img.shape[:2]
        WW=ceil(W/32)*32
        HH=ceil(H/32)*32
        xx=np.zeros((1,HH,WW,3),np.float32)
        xx[0,:H,:W,:]=img.astype(np.float32)/256
        xx[0,H:HH,:W,:]=img[::-1,:,:][:HH-H,:,:].astype(np.float32)/256
        xx[0,:H,W:WW,:]=img[:,::-1,:][:,:WW-W,:].astype(np.float32)/256
        xx[0,H:HH,W:WW,:]=img[::-1,::-1,:][:HH-H,:WW-W,:].astype(np.float32)/256
        results = model.predict(xx)
        print(fn)
        res=(255*results[0,:H,:W]).astype(np.uint8)
        res=cv2.resize(res,None,fx=10/coef,fy=10/coef,interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(os.path.join(save_path,os.path.basename(fn)[:-4]+'-predicted.png'),res)
