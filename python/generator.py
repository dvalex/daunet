import cv2
import numpy as np
from glob import glob
from os import path
from tensorflow import keras
from random import shuffle,random,randint
from transform import random_transform_generator, change_transform_origin
from color import random_visual_effect_generator
def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result
def apply_transform(matrix, image):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = cv2.INTER_LINEAR,
        borderMode  = cv2.BORDER_REFLECT, #cv2.BORDER_CONSTANT,
        borderValue = 0,
    )
    return output

class Generator(keras.utils.Sequence):
    def __init__(self,images_path,labels_path,batch_size,shuffle=True,transform=True):
        self.images=[]
        self.labels=[]
        for fn in glob(path.join(images_path,"*.jpg")):
            img=cv2.imread(fn)[:,:,::-1]
            self.images.append(img)
            label=cv2.imread(fn[:-4]+'.png',0)
            label=cv2.GaussianBlur(label,(5,5),cv2.BORDER_DEFAULT) #soft labels
            self.labels.append(label)
        self.batch_size=batch_size
        self.shuffle=shuffle
        if transform:
            self.transform_generator = random_transform_generator(
                min_rotation=-1.57,
                max_rotation=1.57,
                min_translation=(-0.3, -0.3),
                max_translation=(0.3, 0.3),
                min_shear=-0.5,
                max_shear=0.5,
                min_scaling=(0.5, 0.5),
                max_scaling=(1.5, 1.5),
                flip_x_chance=0.5,
                flip_y_chance=0.5,
            )
            self.visual_effect_generator = random_visual_effect_generator(
                contrast_range=(0.9, 1.1),
                brightness_range=(-.1, .1),
                hue_range=(-0.05, 0.05),
                saturation_range=(0.95, 1.05)
            )
        else:
            self.transform_generator=None
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            c=list(zip(self.images, self.labels))
            shuffle(c)
            self.images, self.labels=zip(*c)

    def __len__(self):
        """
        Number of batches for generator.
        """
        return len(self.images)//self.batch_size*2000

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        """
        Keras sequence method for generating batches.
        """
        inputs=np.zeros((self.batch_size,256,256,3),np.float32)
        targets=np.zeros((self.batch_size,256,256),np.float32)
        for i in range(self.batch_size):
            idx=(index*self.batch_size+i)%len(self.images)
            if idx==0:
                self.on_epoch_end()
            img=self.images[idx].copy()
            label=self.labels[idx].copy()
            for j in range(1000):
                s0=randint(0,img.shape[0]-512)
                s1=randint(0,img.shape[1]-512)
                cr=label[s0:s0+512,s1:s1+512].sum()
                if cr>3200000: 
                    break
            label=label[s0:s0+512,s1:s1+512]
            img=img[s0:s0+512,s1:s1+512]
            s0=randint(0,img.shape[0]-256)
            s1=randint(0,img.shape[1]-256)
            if self.transform_generator is not None:
                transform = adjust_transform_for_image(next(self.transform_generator), inputs[i], True)
                # apply visual effect
                visual_effect = next(self.visual_effect_generator)
                inputs[i] = visual_effect(apply_transform(transform, img)[s0:s0+256,s1:s1+256])
                targets[i] = apply_transform(transform, label)[s0:s0+256,s1:s1+256]
            else:
                inputs[i]=img[s0:s0+256,s1:s1+256]
                targets[i]=label[s0:s0+256,s1:s1+256]
     

        return inputs, targets
    def __call__(self):
        return self
