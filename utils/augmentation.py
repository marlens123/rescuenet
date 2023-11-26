import albumentations as A
import numpy as np
import random
import cv2

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

#scale_min: 0.5
#scale_max: 2.0
#zoom_factor: 8  # zoom factor for final prediction during training, be in [1, 2, 4, 8]


def get_training_augmentation(im_size):
    """
    structure inspired by https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
    
    Defines augmentation for training data. Each technique applied with a probability.
    
    Parameters:
    -----------
        im_size : int
            h,w size of image
    
    Return:
    -------
        train_transform : albumentations.compose
    """
    train_transform = [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Rotate(limit=10, interpolation=0),  
        A.GaussNoise(),   
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(p=1),
                A.MotionBlur(p=1),
            ],
            p=1, 
        ),   
        A.RandomSizedCrop(min_max_height=[int(0.5*im_size), int(0.8*im_size)], height=im_size, width=im_size, interpolation=0, p=0.5),  
    ]
    return A.Compose(train_transform)


def get_preprocessing(preprocessing_fn):
    """
    Preprocessing function.
    
    Parameters:
    -----------
        preprocessing_fn : data normalization function 
            (can be specific for each pretrained neural network)
    
    Return:
    -------
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)