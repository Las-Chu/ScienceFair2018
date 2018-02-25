import csv
import cv2
import numpy as np
import sklearn
import global_defs as gf
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
import skimage

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# store each line from the GroundTruth.csv from the ISIC training data
def read_gt_data(limit=-1):
    lines = []

    # Open the GroundTruth.csv file as csvfile
    with open(gf.gt_file_location) as csvfile:
        reader = csv.reader(csvfile)
        num_lines = 0
        for line in reader:
            lines.append(line)
            num_lines += 1
            if (limit > 0) and (num_lines > limit):
                break
    print('Number of lines read so far:', len(lines))

    return lines

'''
This function creates some uagmented images for Benign cancer images. As the current set has about
4000 images from the database,  the idea is to create 
1 additinal images for each of the benign so that there are approximately 8000 benign images for training.
'''
def augment_benign_images(csv_file_name, melanoma_img_folder):
    file_index = 0
    effect_index = 0

    # Open the GroundTruth.csv file as csvfile
    with open(csv_file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            effect_index = file_index % 6
            augment_benign_image(melanoma_img_folder, line['name'], effect_index)
            file_index = file_index + 1


def augment_benign_image(benign_img_folder, file_key_name, effect_index):

    #compose the origfile name and dest file name
    orig_file = benign_img_folder + '/' + file_key_name + gf.target_img_ext
    print(orig_file)
    #Load the file
    image_init = cv2.imread(orig_file)

    if (effect_index == 0):
        # Equilize hostograms to enhance the image
        fileNamewithFolder = benign_img_folder + '/' + file_key_name + '_enh' + gf.target_img_ext
        img_yuv = cv2.cvtColor(image_init, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # convert the YUV image back to RGB format
        enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite(fileNamewithFolder, enhanced_img)

    if (effect_index == 1):
        # Add Gaussian noise to images for each class type
        fileNamewithFolder = benign_img_folder + '/' + file_key_name + '_gn' + gf.target_img_ext
        noisy_image = blur = cv2.GaussianBlur(image_init, (5, 5), 0)
        cv2.imwrite(fileNamewithFolder, noisy_image)
    if (effect_index == 2):
        # Add rotation at 10
        fileNamewithFolder = benign_img_folder + '/' + file_key_name + '_rtr' + gf.target_img_ext
        rows, cols, channels = image_init.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
        rotated = cv2.warpAffine(image_init, M, (cols, rows))
        cv2.imwrite(fileNamewithFolder, rotated)
    if (effect_index == 3):
        # Add rotation at -10
        fileNamewithFolder = benign_img_folder + '/' + file_key_name + '_rtl' + gf.target_img_ext
        rows, cols, channels = image_init.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
        rotated1 = cv2.warpAffine(image_init, M, (cols, rows))
        cv2.imwrite(fileNamewithFolder, rotated1)
    if (effect_index == 4):
        # Add shift 100, 100 pixels
        fileNamewithFolder = benign_img_folder + '/' + file_key_name + '_shr' + gf.target_img_ext
        rows, cols, channels = image_init.shape
        M = np.float32([[1, 0, 5], [0, 1, 5]])
        shifted = cv2.warpAffine(image_init, M, (cols, rows))
        cv2.imwrite(fileNamewithFolder, shifted)
    if (effect_index == 5):
        # Add shift -100, -100 pixels
        fileNamewithFolder = benign_img_folder + '/' + file_key_name + '_shl' + gf.target_img_ext
        rows, cols, channels = image_init.shape
        M = np.float32([[1, 0, -5], [0, 1, -5]])
        shifted1 = cv2.warpAffine(image_init, M, (cols, rows))
        cv2.imwrite(fileNamewithFolder, shifted1)

'''
This function creates some uagmented images for Melanoma cancer images. As the current set has only
1120 images from the database and there are about ~4000 benign images from the database, the idea is to create 
6 additinal images for each of the melanoma so that there are approximately 8000 melanoma images.
'''
def augment_melanoma_images(csv_file_name, melanoma_img_folder):
    lines = []

    # Open the GroundTruth.csv file as csvfile
    with open(csv_file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
                augment_melanoma_image(melanoma_img_folder, line['name'])



'''
This function creates some uagmented images for Melanoma cancer images. As the current set has only
1120 images from the database and there are about ~4000 benign images from the database, the idea is to create 
6 additinal images for each of the melanoma so that there are approximately 8000 melanoma images.
'''
def augment_melanoma_image(melanoma_img_folder, file_key_name):

    #compose the origfile name and dest file name
    orig_file = melanoma_img_folder + '/' + file_key_name + gf.target_img_ext
    print(orig_file)
    #Load the file
    image_init = cv2.imread(orig_file)

    # Add Gaussian noise to images for each class type
    #fileNamewithFolder = melanoma_img_folder + '/' + file_key_name + '_gn'+ gf.target_img_ext
    #noisy_image = blur = cv2.GaussianBlur(image_init,(5,5),0)
    #cv2.imwrite(fileNamewithFolder, noisy_image)

    # Add sharpen images to the list as well
    #fileNamewithFolder = gf.target_train_dir_aug + file_key_name + '_gb' + gf.target_img_ext
    #blurred_f = cv2.GaussianBlur(image_init, (3, 3), 10.0)
    #sharpened = cv2.addWeighted(image_init, 2, blurred_f, -1, 0)
    #cv2.imwrite(fileNamewithFolder, sharpened)

    # Add rotation at 15
    fileNamewithFolder = melanoma_img_folder + '/' + file_key_name + '_rtr' + gf.target_img_ext
    rows, cols, channels = image_init.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    rotated = cv2.warpAffine(image_init, M, (cols, rows))
    cv2.imwrite(fileNamewithFolder, rotated)

    # Add rotation at -15
    fileNamewithFolder = melanoma_img_folder + '/' + file_key_name + '_rtl' + gf.target_img_ext
    rows, cols, channels = image_init.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
    rotated1 = cv2.warpAffine(image_init, M, (cols, rows))
    cv2.imwrite(fileNamewithFolder, rotated1)

    # Add shift 100, 100 pixels
    fileNamewithFolder = melanoma_img_folder + '/' + file_key_name + '_shr' + gf.target_img_ext
    rows, cols, channels = image_init.shape
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    shifted = cv2.warpAffine(image_init, M, (cols, rows))
    cv2.imwrite(fileNamewithFolder, shifted)

    # Add shift -100, -100 pixels
    fileNamewithFolder = melanoma_img_folder + '/' + file_key_name + '_shl' + gf.target_img_ext
    rows, cols, channels = image_init.shape
    M = np.float32([[1, 0, -5], [0, 1, -5]])
    shifted1 = cv2.warpAffine(image_init, M, (cols, rows))
    cv2.imwrite(fileNamewithFolder, shifted1)

    # Since most of the images are dark, use
    # Equilize hostograms to enhance the image
    fileNamewithFolder = melanoma_img_folder + '/' + file_key_name + '_enh' + gf.target_img_ext
    img_yuv = cv2.cvtColor(image_init, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(fileNamewithFolder, enhanced_img)


# For ISIC data, two columns are important for classification
# second colun notifies if it is melenoma or benign
# third column notifies if it is benign, whether it is Seborrheic keratosis (look a like to melenoma but not a cancer ) or not
# The returned values are as follows: 0 - Benign, 1 - Seborrheic keratosis, 2 - Melenoma
def measure_target_value_for_isic_data(prim, sub):
    return prim #2*prim+sub

# This function reads the numbers of records from the log file as specified by batch size (which is 32 by default)
# essentially 5*batch_size training records


def generator(samples, batch_size=32):
    #label_binarizer = LabelBinarizer()
    #label_binarizer.fit(gf.lb_fit_array)
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            target = []
            msm_init_count = len(target)
            for batch_sample in batch_samples:

                # Read the center image (first one from the parsed list)
                image_name = batch_sample[1]
                image_with_full_name = gf.gt_train_val_location + '/' + image_name + gf.target_img_ext
                # print('Image with full name', image_with_full_name)
                # imageMat = cv2.imread(image_with_full_name)
                imageMat = image.load_img(image_with_full_name, target_size=(299, 299))
                x = image.img_to_array(imageMat)
                #print("xshape", x.shape)
                # print('image shape', imageMat.shape)

                detected_value = float(batch_sample[2])

                if (detected_value > 0):
                    detect_arr = [1, 0]
                else:
                    detect_arr = [0, 1]

                images.append(x)
                target.append(detect_arr)
                # print(image_name, detect_arr)

            # Convert the images into numpy array
            #print("xshape2", images.shape)
            x = np.array(images)
            #print("xshape2", x.shape)
            x = preprocess_input(x)
            #print("xshape2", x.shape)
            X_train = x
            #print("xshape3", X_train.shape)
            Y_train = np.array(target)

            #y_one_hot = label_binarizer.transform(Y_train)
            #print(Y_train, y_one_hot)

            yield sklearn.utils.shuffle(X_train, Y_train)
