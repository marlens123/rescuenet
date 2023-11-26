import numpy as np
from patchify import patchify
from skimage.util import view_as_windows
from sklearn.feature_extraction import image
from sklearn.utils import class_weight
import cv2

def compute_class_weights(train_masks):
    masks_resh = train_masks.reshape(-1,1)
    print(masks_resh.shape)
    masks_resh_list = masks_resh.flatten().tolist()
    print(len(masks_resh_list))
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(masks_resh), y=masks_resh_list)
    print(class_weights.shape)
    return class_weights


def patch_extraction(imgs, masks, size):
    """
    Extracts patches from an image and mask using a sliding window with specified step size.
    """
    if size == 32:
        step = 32
    elif size == 64:
        step = 68
    elif size == 128:
        step = 160
    elif size == 256:
        step == 224
    elif size == 480:
        return imgs, masks
    else:
        print("Unknown patch size. Please enter 32, 64, 128, 256, 480.")

    img_patches = []
    for img in imgs:     
        patches_img = patchify(img, (size, size), step=step)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i,j,:,:]
                img_patches.append(single_patch_img)
    images = np.array(img_patches)

    mask_patches = []
    for img in masks:
        patches_mask = patchify(img, (size, size), step=step)
        
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i,j,:,:]
                mask_patches.append(single_patch_mask)
    masks = np.array(mask_patches)

    return images, masks


def extract_patches(img, rnd_state, nr_patches, patch_size):
    """
    Extracts a given number of random patches from image.

    Parameters:
    -----------
        img : numpy.nd.array
            image to extract patches from
        nr_patches : int
        patch_size : tuple
    """

    patches = image.extract_patches_2d(img, patch_size=patch_size, max_patches=nr_patches, random_state=rnd_state)

    return patches


def patch_pipeline(imgs, masks, size):
    """
    wrapper around random patch function to extract the same patches for image and mask
    """
    if size == 32:
        nr_patches = 320
    elif size == 64:
        nr_patches = 80
    elif size == 128:
        nr_patches = 20
    elif size == 256:
        nr_patches == 5
    elif size == 480:
        return imgs, masks
    else:
        print("Unknown patch size. Please enter 32, 64, 128, 256, 480.")

    patch_size = (patch_size, patch_size)
    
    rnd = 0

    img_patches = []
    for img in imgs:
        patches_img = extract_patches(img, rnd_state=rnd, nr_patches=nr_patches, patch_size=patch_size)
        for i in range(patches_img.shape[0]):
            #for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,:,:]
            img_patches.append(single_patch_img)
        rnd += 1
    
    images = np.array(img_patches)

    rnd = 0

    mask_patches = []
    for mask in masks:
        patches_mask = extract_patches(mask, rnd_state=rnd, nr_patches=nr_patches, patch_size=patch_size)
        for i in range(patches_mask.shape[0]):
            #for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i,:,:]
            mask_patches.append(single_patch_mask)
        rnd += 1
    
    masks = np.array(mask_patches)

    return images, masks


def resize(imgs, masks, im_size):

    images = []
    labels = []

    for i in range(imgs.shape[0]):
        image = cv2.resize(imgs[i], (im_size, im_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(masks[i], (im_size, im_size), interpolation=cv2.INTER_NEAREST)

        images.append(image)
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)

    return images, labels