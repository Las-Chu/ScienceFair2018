import csv
import os
from shutil import copyfile
import global_defs as gdefs
import data_generator as dgen

'''
Now augment the melanoma images (for each original images, create 5 more augmented images
'''

dgen.augment_melanoma_images(gdefs.malign_file_name, gdefs.training_melanoma_image_folder )
dgen.augment_benign_images(gdefs.benign_file_name, gdefs.training_benign_image_folder)