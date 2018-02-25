import cv2
import os
import numpy as np

import global_defs as gf

def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = (x - cropx) / 2
    starty = (y - cropy) / 2
    return img[starty:starty + cropy, startx:startx + cropx]


# get ImageMat for the given file location in the Data log
def process_image(folder, imageField):
    imagePath = imageField
    fileName = imagePath.split('/')[-1]
    fileNamewithFolder = folder + '/' + fileName
    imageMat = cv2.imread(imageField)

    im_h, im_w, im_c = imageMat.shape
    #print(im_h, im_w, im_c)
    print(fileNamewithFolder, imageMat.shape)

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
    imageMat = crop_center(imageMat, gf.target_img_width, gf.target_img_height)
    cv2.imwrite(fileNamewithFolder, imageMat)



#process_image(gf.target_train_dir, '/home/ramesh/Data/isic2017_data/ISIC-2017_Training_Data/ISIC_0000000.jpg')

import glob
import time
import random
from os.path import basename

def create_train_data(src_folder, isbenign, ismalign, target_folder):


    b_file = open(gf.gt_file_location, 'a')
    images = glob.glob(src_folder+ '/*' + gf.target_img_ext)
    for image in images:
        # create an entry in the trainin_gt_file
        str_val = src_folder + ',' + os.path.splitext(basename(image))[0] + ',' + str(isbenign) + ',' + str(ismalign) + '\n'
        b_file.write(str_val)
        # copy the resized file to the trained data folder
        process_image(target_folder, image)



# Create the new train val data set and ground truth tables
# Opebn GT file for wrting the file name and hot endoded data
if os.path.exists(gf.gt_file_location):
    os.remove(gf.gt_file_location)

# create the folder for the train_val images
if not os.path.exists(gf.gt_train_val_location):
    os.makedirs(gf.gt_train_val_location)

# Now process benign files
create_train_data(gf.training_benign_image_folder, 1.0, 0.0, gf.gt_train_val_location)
# Now process malignant files
create_train_data(gf.training_melanoma_image_folder, 0.0, 1.0, gf.gt_train_val_location)