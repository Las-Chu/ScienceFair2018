import cv2
import os
import numpy as np

import global_defs as gf

def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

def ret_proc_image(image, size=[gf.target_img_width, gf.target_img_height]):
    imageMat = cv2.imread(image)

    im_h, im_w, im_c = imageMat.shape
    # print(im_h, im_w, im_c)
    print(imageMat.shape)

    r_of_w = im_w / gf.small_width
    r_of_h = im_h / gf.small_height

    # See if need to be resized
    if ((gf.small_width == im_w) or (gf.small_height == im_h)):
        imageMat = imageMat
    else:
        r_of_w = 1.0 * im_w / gf.small_width
        r_of_h = 1.0 * im_h / gf.small_height

    if (r_of_h < r_of_w):
        resize_height = gf.small_height
        resize_width = int(resize_height * (1.0 * im_w / im_h))
    else:
        resize_width = gf.small_width
        resize_height = int(resize_width * (1.0 * im_h / im_w))

    imageMat = cv2.resize(imageMat, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    imageMat = crop_center(imageMat, size[0], size[1])

    return imageMat