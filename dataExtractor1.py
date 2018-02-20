import glob
import time
import random
import json
import csv
import os
import os.path

import global_defs as gdefs

from pprint import pprint

clinical_keys = ['benign_malignant', 'age_approx', 'sex', 'diagnosis', 'diagnosis_confirm_type', 'melanocytic']
acqusition_keys = ['image_type', 'dermoscopic_type', 'pixelsX', 'pixelsY']


def print_images(folder, dumpFileName):

    images = glob.glob('/home/ramesh/vidya/project_cancer/dataset/*.jpg')
    #meta_json = glob.glob('/home/ramesh/vidya/project_cancer/dataset/*.json')
    meta_json = glob.glob(folder+'/*.json')

    IsBenign = False

    temp_key = ""
    temp_value = ""
    temp_key_name = ""
    temp_value_name = ""
    with_header= True


    for jsonPtr in meta_json:
        data_from_json = json.load(open(jsonPtr)) # gets Dict

        #open a file for writing
        csv = open(dumpFileName, 'a')
        #create a csv writer object
        #print data_from_json
        for key1, value1 in data_from_json.iteritems():
            if (key1 == "meta"):
                temp_key = ""
                temp_value = ""
                for key2, value2 in value1.iteritems():
                    if (key2 == "clinical"):
                        for key3 in clinical_keys:
                            temp_key = temp_key + "," + str(key3)
                            try:
                                temp_value = temp_value + ","  + str(value1['clinical'][key3])
                            except:
                                temp_value = temp_value + "," + str(' ')

                    if (key2 == "acquisition"):
                        for key3 in acqusition_keys:
                            temp_key = temp_key + "," + str(key3)
                            try:
                                temp_value = temp_value + ","  + str(value1['acquisition'][key3])
                            except:
                                temp_value = temp_value + "," + str(' ')


            if (key1 == "name"):

                # Add folder name
                temp_key_name = str('Folder') + ","
                temp_value_name = str(folder) + ","

                # Add file name
                temp_key_name   += str(key1)
                temp_value_name += str(value1)


        if (with_header):
            csv.write(temp_key_name + "," + temp_key + '\n')
            csv.write(temp_value_name + "," + temp_value + '\n')
            with_header = False
        else:
            csv.write(temp_value_name + "," + temp_value + '\n')

        # close opened file
    csv.close()

# remove the file if exists
# dumpFileName = "ds_metadata1.csv"
if os.path.exists(gdefs.dumpFileName):
    os.remove(gdefs.dumpFileName)
#rootFolder = '/home/ramesh/Data/ISIC-images/'
for dirName, subdirList, fileList in os.walk(gdefs.rootFolder):
    if ((dirName != gdefs.rootFolder) and (dirName.find(gdefs.ignore_folder) < 0)):
        print(dirName)
        print_images(dirName, gdefs.dumpFileName)