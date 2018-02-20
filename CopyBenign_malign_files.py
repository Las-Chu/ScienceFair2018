
import csv
import os
from shutil import copyfile
import global_defs as gdefs

def copyfiles_to_each_folder(file_list, dest_folder):

    # Create the dest_folder if didn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    # Open the csv file
    with open(file_list) as csvfile:
        reader = csv.DictReader(csvfile)
        # For each row in this file, copy the file to a dest folder
        for row in reader:
            #print(row)
            src_file = row['Folder'] + '/' + row['name'] + '.jpg'
            dest_file = dest_folder + '/' + row['name'] + '.jpg'
            # Copy src_file to dest_folder
            copyfile(src_file, dest_file)


# Copy Benign files
copyfiles_to_each_folder(gdefs.benign_file_name, gdefs.training_Files_Folder+'/benign')

# Copy Malignant files
copyfiles_to_each_folder(gdefs.malign_file_name, gdefs.training_Files_Folder+'/malign')
