import sys
sys.path.append('../train_test/')
sys.path.append('../')

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


from only_for_vessel_seg.data_process.retinal_process import rgb2gray, clahe_equalized, dataset_normalized, adjust_gamma
from only_for_vessel_seg.data_process.data_ultils import group_images, visualize, label2rgb
from only_for_vessel_seg import Constants
from only_for_vessel_seg.data_process.ACE import zmIceColor
import warnings
warnings.filterwarnings("ignore")

save_drive0 = '_drive0_'
save_drive1 = '_drive1_'
save_drive2 = '_drive2_'
save_drive3 = '_drive3_'
save_drive4 = '_drive4_'


save_drive = '_drive'
save_drive_color = '_drive_color'
save_mo = '_mo'
save_pylop = '_pylop'
save_tbnc = '_tbnc'


def visual_sample(images, mask, path, per_row =5):
    visualize(group_images(images, per_row), Constants.visual_samples + path + '0')
    visualize(group_images(mask, per_row), Constants.visual_samples + path + '1')


def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def get_drive_data(val_ratio = 0.1, is_train = True):
    images1 = load_from_npy(Constants.path_image_drive)
    mask = load_from_npy(Constants.path_label_drive)

    images_test1 = load_from_npy(Constants.path_test_image_drive)
    mask_test = load_from_npy(Constants.path_test_label_drive)



    images2 = rgb2gray(images1)
    images3 = dataset_normalized(images2)
    images4 = clahe_equalized(images3)
    images = adjust_gamma(images4, 1.0)

    images_test2 = rgb2gray(images_test1)
    images_test3 = dataset_normalized(images_test2)
    images_test4 = clahe_equalized(images_test3)
    images_test = adjust_gamma(images_test4, 1.0)


    images = images/255.                # reduce to 0-1 range
    images_test = images_test / 255.
    # images_val = images_val / 255.
    print(images.shape, mask.shape, '=================', np.max(images), np.max(mask))

    print('========  success load all Drive files ==========')


    visual_sample(images[0:20,:,:,:,], mask[0:20,:,:,:,], save_drive)



    train_list = [images[0:, :, :, :, ], mask[0:, :, :, :, ]]
    val_list = [images_test, mask_test]
    if is_train is True:
        return train_list, val_list
    else:
        return images_test, mask_test



class ImageFolder(data.Dataset):
    '''
    image is RGB original image, mask is one hot GT and label is grey image to visual
    img and mask is necessary while label is alternative
    '''
    def __init__(self,img, mask, label=None):
        self.img = img
        self.mask = mask
        self.label = label

    def __getitem__(self, index):
        imgs  = torch.from_numpy(self.img[index]).float()
        masks = torch.from_numpy(self.mask[index]).float()
        if self.label is not None:
            label = torch.from_numpy(self.label[index]).float()
            return imgs, masks, label
        else:
            return imgs, masks

    def __len__(self):
        assert self.img.shape[0] == self.mask.shape[0], 'The number of images must be equal to labels'
        return self.img.shape[0]


if __name__ == '__main__':

    get_drive_data()
    # get_monuclei_data()
    # get_MRI_chaos_data()
    # get_test_MRI_chaos_data()
    # get_tnbc_data(0.2, is_train = True)
    # get_pylyp_data()
    # get_drive_color_data()

    pass
