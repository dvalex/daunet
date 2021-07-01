#encoding:utf-8

from warmup import  WarmUpCosineDecayScheduler
import os
from tensorflow import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
import segmentation_models as sm
from keras_ema import ExponentialMovingAverage

from metrics2 import dice_loss
from metrics2 import weighted_binary_crossentropy
from generator import Generator
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

def trainGenerator(batch_size):
     train_generator = Generator("cracks500","cracks500", batch_size)
     index=0
     for (img, label) in train_generator:
         img=np.asarray(img,np.float32)/255
         label=np.asarray(label,np.float32)/255
         yield (img, label)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



batch_size=8
learning_rate_base=0.001
total_epochs=200
warmup_epochs=10
steps_per_epoch=1896//batch_size

train_data = trainGenerator(batch_size=batch_size)
model = sm.Unet("resnet18", classes=1,activation="sigmoid",encoder_weights="imagenet")
optimizer=keras.optimizers.Adam(learning_rate_base)
lr_metric = get_lr_metric(optimizer)
model.compile(optimizer=optimizer,loss=sm.losses.binary_focal_loss,metrics=[dice_loss,sm.metrics.iou_score,lr_metric],)
print(model.summary())
model_checkpoint1 = keras.callbacks.ModelCheckpoint('Unet-EMA-Soft-resnet18-focal-pretrained-bs8-newaug-crop-aspect-reflect-color-full200.hdf5')
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_epochs*steps_per_epoch,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_epochs*steps_per_epoch,
                                        hold_base_rate_steps=0)
ema=ExponentialMovingAverage()

csv_logger = CSVLogger('trainingUnetEMA-newaug.log', append=True, separator=';')
history = model.fit(train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=total_epochs,
                    callbacks=[model_checkpoint1,csv_logger,warm_up_lr,ema])
