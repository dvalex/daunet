from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def weighted_binary_crossentropy(y_true, y_pred):#ponderacion de pesos para darle mayor peso a la clase=1
	zero_weight=1
	one_weight=4
	b_ce = K.binary_crossentropy(y_true, y_pred)
	weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
	weighted_b_ce = weight_vector * b_ce
	return K.mean(weighted_b_ce)
