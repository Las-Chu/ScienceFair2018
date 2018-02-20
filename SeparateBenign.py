import glob
import time
import random
import json
import csv
import os
from pprint import pprint
import global_defs as gdefs

def write_tofile(fileHandle, string_to_write):
    fileHandle.write(string_to_write)

# Cleanup the files if exist

if os.path.exists(gdefs.benign_file_name):
    os.remove(gdefs.benign_file_name)

if os.path.exists(gdefs.malign_file_name):
    os.remove(gdefs.malign_file_name)

b_file = open(gdefs.benign_file_name, 'a')
m_file = open(gdefs.malign_file_name, 'a')

# Write headers
str_header = 'Folder' + ',' + 'name' + ',' + 'benign_malignant' + ',' + 'diagnosis' + ',' + 'melanocytic' + ',' + 'pixelsX' + ',' + 'pixelsY' + '\n'
write_tofile(b_file, str_header)
write_tofile(m_file, str_header)

with open(gdefs.dumpFileName) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        str = row['Folder'] + ',' + row['name'] + ',' + row['benign_malignant'] + ',' + row['diagnosis'] + ',' + row['melanocytic'] + ',' + row['pixelsX'] + ',' + row['pixelsY'] + '\n'
        if (row['benign_malignant'].find('malignant') >= 0):
            write_tofile(m_file, str)
        else:
            if row['benign_malignant'].find('benign') >= 0:
                write_tofile(b_file, str)


