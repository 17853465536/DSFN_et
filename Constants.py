

#========================== public configure ==========================
IMG_SIZE = (960, 960)
TOTAL_EPOCH = 600
INITAL_EPOCH_LOSS = 1000000
NUM_EARLY_STOP = 60
NUM_UPDATE_LR = 100
BINARY_CLASS = 1
BATCH_SIZE = 1
learning_rates =1e-3


# ===================   DRIVE configure =========================
DATA_SET = 'HRF_(UU_C_1)_3_(960_768_512)'
# DATA_SET = 'CHASEDB1_(UU_C_1)_3_(960_900_800_768_600_512)'
visual_samples = '../log/visual_samples/'
saved_path = '../log/weights_save/'+ DATA_SET + '/'
visual_results = '../log/visual_results/'+ DATA_SET + '/'

# resize_drive = 288
# resize_size_drive = (resize_drive, resize_drive)
# size_h, size_w = 300, 300

resize_drive = 960
resize_size_drive = (resize_drive, resize_drive)
# size_h, size_w = 584, 565
# size_h, size_w = 2048, 2048

# size_h, size_w = 960, 999
# size_h, size_w = 605, 700
size_h, size_w = 2336, 3504

# 注意！！！
# 1、现在是584*565的图像文件保存在/tempt目录下，对应运行read_DRIVE_crop.py 文件
# 2、如果是512*512的要去掉路径的/tempt，对应运行read_DRIVE.py 文件
# path_image_drive = '../dataset1/npy/DRIVE/tempt/train_image_save.npy'
# path_label_drive = '../dataset1/npy/DRIVE/tempt/train_label_save.npy'
# path_test_image_drive = '../dataset1/npy/DRIVE/tempt/test_image_save.npy'
# path_test_label_drive = '../dataset1/npy/DRIVE/tempt/test_label_save.npy'
# path_val_image_drive = '../dataset1/npy/DRIVE/tempt/val_image_save.npy'
# path_val_label_drive = '../dataset1/npy/DRIVE/tempt/val_label_save.npy'



# path_image_drive = '../dataset1/npy/CHASE_DB1/tempt/train_image_save.npy'
# path_label_drive = '../dataset1/npy/CHASE_DB1/tempt/train_label_save.npy'
# path_test_image_drive = '../dataset1/npy/CHASE_DB1/tempt/test_image_save.npy'
# path_test_label_drive = '../dataset1/npy/CHASE_DB1/tempt/test_label_save.npy'

#
# path_image_drive = '../dataset1/npy/STARE1_20/tempt/train_image_save.npy'
# path_label_drive = '../dataset1/npy/STARE1_20/tempt/train_label_save.npy'
# path_test_image_drive = '../dataset1/npy/STARE1_20/tempt/test_image_save.npy'
# path_test_label_drive = '../dataset1/npy/STARE1_20/tempt/test_label_save.npy'

path_image_drive = '../dataset1/npy/HRF/tempt/train_image_save.npy'
path_label_drive = '../dataset1/npy/HRF/tempt/train_label_save.npy'
path_test_image_drive = '../dataset1/npy/HRF/tempt/test_image_save.npy'
path_test_label_drive = '../dataset1/npy/HRF/tempt/test_label_save.npy'
# path_val_image_drive = '../dataset1/npy/HRF/tempt/val_image_save.npy'
# path_val_label_drive = '../dataset1/npy/HRF/tempt/val_label_save.npy'

#
# path_image_drive = '../dataset1/npy/DCA1/tempt/train_image_save.npy'
# path_label_drive = '../dataset1/npy/DCA1/tempt/train_label_save.npy'
# path_test_image_drive = '../dataset1/npy/DCA1/tempt/test_image_save.npy'
# path_test_label_drive = '../dataset1/npy/DCA1/tempt/test_label_save.npy'

#
# path_image_drive = '../dataset1/npy/FIVES/tempt/train_image_save.npy'
# path_label_drive = '../dataset1/npy/FIVES/tempt/train_label_save.npy'
# path_test_image_drive = '../dataset1/npy/FIVES/tempt/test_image_save.npy'
# path_test_label_drive = '../dataset1/npy/FIVES/tempt/test_label_save.npy'
# path_val_image_drive = '../dataset1/npy/FIVES/tempt/val_image_save.npy'
# path_val_label_drive = '../dataset1/npy/FIVES/tempt/val_label_save.npy'

# path_image_drive = '/root/My/CS_Net_master/only_for_vessel_seg_1/dataset2/npy/DRIVE_STARE/tempt/train_image_save.npy'
# path_label_drive = '/root/My/CS_Net_master/only_for_vessel_seg_1/dataset2/npy/DRIVE_STARE/tempt/train_label_save.npy'
# path_test_image_drive = '/root/My/CS_Net_master/only_for_vessel_seg_1/dataset2/npy/DRIVE_STARE/tempt/test_image_save.npy'
# path_test_label_drive = '/root/My/CS_Net_master/only_for_vessel_seg_1/dataset2/npy/DRIVE_STARE/tempt/test_label_save.npy'



# path_image_drive = '/root/My/CS_Net_master/only_for_vessel_seg_1/dataset2/npy/STARE_DRIVE/tempt/train_image_save.npy'
# path_label_drive = '/root/My/CS_Net_master/only_for_vessel_seg_1/dataset2/npy/STARE_DRIVE/tempt/train_label_save.npy'
# path_test_image_drive = '/root/My/CS_Net_master/only_for_vessel_seg_1/dataset2/npy/STARE_DRIVE/tempt/test_image_save.npy'
# path_test_label_drive = '/root/My/CS_Net_master/only_for_vessel_seg_1/dataset2/npy/STARE_DRIVE/tempt/test_label_save.npy'



total_drive = 45
Classes_drive_color = 3
###########################################################################################