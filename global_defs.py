dumpFileName = "ds_metadata1.csv"
benign_file_name = 'ds_benign_metadata1.csv'
malign_file_name = 'ds_malign_metadata1.csv'
rootFolder = '/home/ramesh/Data/ISIC-images/'
training_Files_Folder = '/home/ramesh/Data/ISIC-images-train-LC/'
training_melanoma_image_folder = training_Files_Folder + 'malign'
training_benign_image_folder = training_Files_Folder + 'benign'
gt_train_val_location = training_Files_Folder + 'train_val'
gt_file_location = training_Files_Folder + 'isic_data_train_val.csv'
ignore_folder = 'SONIC'
gt_nb_classes = 2

small_width = 400
small_height = 400

project_name='ISIC-2017_Orig_train_data'
data_folder  ='/home/ramesh/Data/isic2017_data/'+project_name+'/'
target_train_dir = data_folder + 'isic2017_laasya_data/'
target_train_dir_aug = data_folder + 'isic2017_laasya_data_aug/'
gt_file = 'ISIC-2017_Training_Part3_GroundTruth_final.csv'
#gt_nb_classes = 3
#lb_fit_array = [0.0, 1.0, 2.0]
sCheckData = 'ISIC'

target_img_ext = '.jpg'
target_img_width = 299
target_img_height = 299