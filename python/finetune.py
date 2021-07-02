#encoding:utf-8

from warmup import  WarmUpCosineDecayScheduler
import os
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger
import segmentation_models as sm
from keras_ema import ExponentialMovingAverage
#import tensorflow_addons as tfa
from metrics2 import dice_loss
from metrics2 import weighted_binary_crossentropy
from generator import Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import cv2
import numpy as np

class BinaryFocalLossWithIgnore(sm.losses.BinaryFocalLoss):

    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(alpha,gamma)

    def __call__(self, gt, pr):
        print("GT",gt)
        print("PR",pr)
        return super().__call__(K.reshape((gt[:,:,:,0]*gt[:,:,:,1]),(8,256,256,1)), pr*K.reshape(gt[:,:,:,1],(8,256,256,1)))
    

def trainGenerator(batch_size):
     train_generator = Generator("../data/cracks500","../data/cracks500", batch_size)
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


batch_size=8 # 16
learning_rate_base=0.01#0.002
total_epochs=20#100
steps_per_epoch=1896//batch_size

train_data = trainGenerator(batch_size=batch_size)
model = sm.Unet("resnet18", classes=1,activation="sigmoid",encoder_weights="imagenet")
optimizer=keras.optimizers.SGD(learning_rate_base)
lr_metric = get_lr_metric(optimizer)
model.load_weights('DAUNet.hdf5')
model.compile(optimizer=optimizer,loss=sm.losses.dice_loss,metrics=[sm.metrics.iou_score,lr_metric],)
print(model.summary())


model_checkpoint1 = keras.callbacks.ModelCheckpoint('DAUNet-finetuned.hdf5')
logName='finetune.log'                
csv_logger = CSVLogger(logName, append=True, separator=';')
history = model.fit(train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=total_epochs,
                    callbacks=[model_checkpoint1,csv_logger])
