import sys
sys.path.append('../')                  # import Constant file
sys.path.append('/root/code/Vessel_Net')
sys.path.append('../data_process/')
sys.path.append('../networks/common/')
sys.path.append('../networks/MESnet/')

import torch
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt
from  PIL import  Image
import cv2
from only_for_vessel_seg import Constants
from only_for_vessel_seg.data_process.data_ultils import read_all_images, data_shuffle, read_all_images1

# path_images_drive = '../dataset1/DRIVE/training/images/'
# path_gt_drive = '../dataset1/DRIVE/training/1st_manual/'
# path_images_test_drive = '../dataset1/DRIVE/test/images/'
# path_gt_test_drive = '../dataset1/DRIVE/test/1st_manual/'
# path_images_val_drive = '../dataset1/DRIVE/val/images/'
# path_gt_val_drive = '../dataset1/DRIVE/val/1st_manual/'


# path_images_drive = '../dataset1/CHASE_DB1/training/images/'
# path_gt_drive = '../dataset1/CHASE_DB1/training/1st_manual/'
# path_images_test_drive = '../dataset1/CHASE_DB1/test/images/'
# path_gt_test_drive = '../dataset1/CHASE_DB1/test/1st_manual/'

# path_images_drive = '../dataset1/STARE1/training/images/'
# path_gt_drive = '../dataset1/STARE1/training/1st_manual/'
# path_images_test_drive = '../dataset1/STARE1/test/images/'
# path_gt_test_drive = '../dataset1/STARE1/test/1st_manual/'

# path_images_drive = '../dataset1/HRF/training/images/'
# path_gt_drive = '../dataset1/HRF/training/1st_manual/'
# path_images_test_drive = '../dataset1/HRF/test/images/'
# path_gt_test_drive = '../dataset1/HRF/test/1st_manual/'
# path_images_val_drive = '../dataset1/HRF/val/images/'
# path_gt_val_drive = '../dataset1/HRF/val/1st_manual/'

# path_images_drive = '../dataset1/DCA1/training/images/'
# path_gt_drive = '../dataset1/DCA1/training/1st_manual/'
# path_images_test_drive = '../dataset1/DCA1/test/images/'
# path_gt_test_drive = '../dataset1/DCA1/test/1st_manual/'

# path_images_drive = '../dataset1/FIVES/test/images/'
# path_gt_drive = '../dataset1/FIVES/test/1st_manual/'
# path_images_test_drive = '../dataset1/FIVES/test/images/'
# path_gt_test_drive = '../dataset1/FIVES/test/1st_manual/'
path_images_test_drive = '../dataset1/FIVES/val/images/'
path_gt_test_drive = '../dataset1/FIVES/val/1st_manual/'


def read_numpy_into_npy(arrays, path):
    np.save(path, arrays)
    print('have saved all arrays in to path ', path)

def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def read_drive_images(size_h,size_w, path_images, path_gt,total_imgs, mask_ch =1):
    all_images = np.empty(shape=(total_imgs, size_h, size_w, 3))
    all_masks  = np.empty(shape=(total_imgs, size_h, size_w, mask_ch))# DRIVE CHASEDB HRF dataset  FIVES——mask为单通道
    # all_masks = np.empty(shape=(total_imgs, size_h, size_w, 3)) # STARE DCA1 FIVES dataset
    all_images = read_all_images(path_images, all_images,size_h, size_w,type ='non_resize')
    all_masks  = read_all_images1(path_gt, all_masks, size_h, size_w,type ='non_resize')

    # all_masks = all_masks[:, :, :, 1, ]  # STARE DCA1   dataset   FIVES——mask为3通道
    # all_masks = np.expand_dims(all_masks, axis=3)  # STARE DCA1  FIVES dataset

    print('============= have read all images ==============')
    return all_images, all_masks

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
    return image, mask

def randomHorizontalFlip(image, mask):
    if np.random.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return  image, mask

def randomVerticleFlip(image, mask):
    if np.random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return  image, mask

def crop_images0(image, mask, crop_size = Constants.resize_drive):
    # select_id = np.random.randint(0, 4)
    select_id = 0
    d_h, d_w, h, w =  image.shape[0] - crop_size, image.shape[1] - crop_size,image.shape[0],image.shape[1]
    crop_lu_im,  crop_lu_ma = image[d_h:h, d_w:w, :,], mask[d_h:h, d_w:w, :,]
    crop_ld_im,  crop_ld_ma = image[d_h:h, 0:w-d_w, :, ], mask[d_h:h, 0:w-d_w, :, ]
    crop_ru_im,  crop_ru_ma = image[0:h - d_h, d_w:w, :, ], mask[0:h - d_h, d_w:w, :, ]
    crop_rd_im,  crop_rd_ma = image[0:h - d_h, 0:w-d_w, :, ], mask[0:h - d_h, 0:w-d_w, :, ]
    # crop_img = np.concatenate([np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_ld_im, axis=0),
    #                 np.expand_dims(crop_ru_im, axis=0),np.expand_dims(crop_rd_im, axis=0)], axis = 0)
    # crop_mask = np.concatenate([np.expand_dims(crop_lu_ma, axis=0), np.expand_dims(crop_lu_ma, axis=0),
    #                 np.expand_dims(crop_lu_ma, axis=0),np.expand_dims(crop_lu_ma, axis=0)], axis = 0)
    crop_img, crop_mask =None, None
    if select_id ==0:
        crop_img, crop_mask = np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_lu_ma, axis=0)
    if select_id ==1:
        crop_img, crop_mask = np.expand_dims(crop_ld_im, axis=0), np.expand_dims(crop_ld_ma, axis=0)
    if select_id ==2:
        crop_img, crop_mask = np.expand_dims(crop_ru_im, axis=0), np.expand_dims(crop_ru_ma, axis=0)
    if select_id ==3:
        crop_img, crop_mask = np.expand_dims(crop_rd_im, axis=0), np.expand_dims(crop_rd_ma, axis=0)
    return crop_img, crop_mask

def crop_images1(image, mask, crop_size = Constants.resize_drive):
    select_id = 1
    d_h, d_w, h, w =  image.shape[0] - crop_size, image.shape[1] - crop_size,image.shape[0],image.shape[1]
    crop_lu_im,  crop_lu_ma = image[d_h:h, d_w:w, :,], mask[d_h:h, d_w:w, :,]
    crop_ld_im,  crop_ld_ma = image[d_h:h, 0:w-d_w, :, ], mask[d_h:h, 0:w-d_w, :, ]
    crop_ru_im,  crop_ru_ma = image[0:h - d_h, d_w:w, :, ], mask[0:h - d_h, d_w:w, :, ]
    crop_rd_im,  crop_rd_ma = image[0:h - d_h, 0:w-d_w, :, ], mask[0:h - d_h, 0:w-d_w, :, ]
    # crop_img = np.concatenate([np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_ld_im, axis=0),
    #                 np.expand_dims(crop_ru_im, axis=0),np.expand_dims(crop_rd_im, axis=0)], axis = 0)
    # crop_mask = np.concatenate([np.expand_dims(crop_lu_ma, axis=0), np.expand_dims(crop_lu_ma, axis=0),
    #                 np.expand_dims(crop_lu_ma, axis=0),np.expand_dims(crop_lu_ma, axis=0)], axis = 0)
    crop_img, crop_mask =None, None
    if select_id ==0:
        crop_img, crop_mask = np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_lu_ma, axis=0)
    if select_id ==1:
        crop_img, crop_mask = np.expand_dims(crop_ld_im, axis=0), np.expand_dims(crop_ld_ma, axis=0)
    if select_id ==2:
        crop_img, crop_mask = np.expand_dims(crop_ru_im, axis=0), np.expand_dims(crop_ru_ma, axis=0)
    if select_id ==3:
        crop_img, crop_mask = np.expand_dims(crop_rd_im, axis=0), np.expand_dims(crop_rd_ma, axis=0)
    return crop_img, crop_mask

def crop_images2(image, mask, crop_size = Constants.resize_drive):
    select_id = 2
    d_h, d_w, h, w =  image.shape[0] - crop_size, image.shape[1] - crop_size,image.shape[0],image.shape[1]
    crop_lu_im,  crop_lu_ma = image[d_h:h, d_w:w, :,], mask[d_h:h, d_w:w, :,]
    crop_ld_im,  crop_ld_ma = image[d_h:h, 0:w-d_w, :, ], mask[d_h:h, 0:w-d_w, :, ]
    crop_ru_im,  crop_ru_ma = image[0:h - d_h, d_w:w, :, ], mask[0:h - d_h, d_w:w, :, ]
    crop_rd_im,  crop_rd_ma = image[0:h - d_h, 0:w-d_w, :, ], mask[0:h - d_h, 0:w-d_w, :, ]
    # crop_img = np.concatenate([np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_ld_im, axis=0),
    #                 np.expand_dims(crop_ru_im, axis=0),np.expand_dims(crop_rd_im, axis=0)], axis = 0)
    # crop_mask = np.concatenate([np.expand_dims(crop_lu_ma, axis=0), np.expand_dims(crop_lu_ma, axis=0),
    #                 np.expand_dims(crop_lu_ma, axis=0),np.expand_dims(crop_lu_ma, axis=0)], axis = 0)
    crop_img, crop_mask =None, None
    if select_id ==0:
        crop_img, crop_mask = np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_lu_ma, axis=0)
    if select_id ==1:
        crop_img, crop_mask = np.expand_dims(crop_ld_im, axis=0), np.expand_dims(crop_ld_ma, axis=0)
    if select_id ==2:
        crop_img, crop_mask = np.expand_dims(crop_ru_im, axis=0), np.expand_dims(crop_ru_ma, axis=0)
    if select_id ==3:
        crop_img, crop_mask = np.expand_dims(crop_rd_im, axis=0), np.expand_dims(crop_rd_ma, axis=0)
    return crop_img, crop_mask

def crop_images3(image, mask, crop_size = Constants.resize_drive):
    select_id = 3
    d_h, d_w, h, w =  image.shape[0] - crop_size, image.shape[1] - crop_size,image.shape[0],image.shape[1]
    crop_lu_im,  crop_lu_ma = image[d_h:h, d_w:w, :,], mask[d_h:h, d_w:w, :,]
    crop_ld_im,  crop_ld_ma = image[d_h:h, 0:w-d_w, :, ], mask[d_h:h, 0:w-d_w, :, ]
    crop_ru_im,  crop_ru_ma = image[0:h - d_h, d_w:w, :, ], mask[0:h - d_h, d_w:w, :, ]
    crop_rd_im,  crop_rd_ma = image[0:h - d_h, 0:w-d_w, :, ], mask[0:h - d_h, 0:w-d_w, :, ]
    # crop_img = np.concatenate([np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_ld_im, axis=0),
    #                 np.expand_dims(crop_ru_im, axis=0),np.expand_dims(crop_rd_im, axis=0)], axis = 0)
    # crop_mask = np.concatenate([np.expand_dims(crop_lu_ma, axis=0), np.expand_dims(crop_lu_ma, axis=0),
    #                 np.expand_dims(crop_lu_ma, axis=0),np.expand_dims(crop_lu_ma, axis=0)], axis = 0)
    crop_img, crop_mask =None, None
    if select_id ==0:
        crop_img, crop_mask = np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_lu_ma, axis=0)
    if select_id ==1:
        crop_img, crop_mask = np.expand_dims(crop_ld_im, axis=0), np.expand_dims(crop_ld_ma, axis=0)
    if select_id ==2:
        crop_img, crop_mask = np.expand_dims(crop_ru_im, axis=0), np.expand_dims(crop_ru_ma, axis=0)
    if select_id ==3:
        crop_img, crop_mask = np.expand_dims(crop_rd_im, axis=0), np.expand_dims(crop_rd_ma, axis=0)
    return crop_img, crop_mask


def crop_order_images(image, mask, crop_size=512, row=4, col=4, mode='orders', rands=16):
    # 将 numpy 转换为 torch 张量
    image, mask = torch.tensor(image.copy()), torch.tensor(mask.copy())
    image = torch.unsqueeze(image.permute((2, 0, 1)), dim=0)  # 转换为 [1, C, H, W]
    mask = torch.unsqueeze(mask.permute((2, 0, 1)), dim=0)    # 转换为 [1, C, H, W]

    # 顺序裁剪逻辑
    if mode == 'orders':
        height, width = image.size(2), image.size(3)
        assert height % crop_size == 0 and width % crop_size == 0, \
            "Image dimensions must be divisible by crop_size without padding in 'orders' mode."

        step_h, step_w = height // row, width // col  # 每个裁剪块的步长
        assert crop_size <= step_h and crop_size <= step_w, \
            "Crop size should be less than or equal to each step size."

        img_set, mask_set = [], []
        # 遍历行列，进行顺序裁剪
        for m in range(row):
            for n in range(col):
                dh = m * step_h
                dw = n * step_w
                img_set.append(image[:, :, dh:dh+crop_size, dw:dw+crop_size])
                mask_set.append(mask[:, :, dh:dh+crop_size, dw:dw+crop_size])

        # 合并所有裁剪块
        return torch.cat(img_set, dim=0), torch.cat(mask_set, dim=0)

    # 随机裁剪逻辑
    elif mode == 'randoms':
        import random
        img_set, mask_set = [], []
        # 随机裁剪
        for _ in range(rands):
            # 随机生成裁剪中心点
            center_y = random.randint(crop_size // 2, image.size(2) - crop_size // 2)
            center_x = random.randint(crop_size // 2, image.size(3) - crop_size // 2)

            # 提取裁剪块
            crops_img = image[:, :, center_y - crop_size // 2:center_y + crop_size // 2, center_x - crop_size // 2:center_x + crop_size // 2]
            crops_mask = mask[:, :, center_y - crop_size // 2:center_y + crop_size // 2, center_x - crop_size // 2:center_x + crop_size // 2]

            img_set.append(crops_img)
            mask_set.append(crops_mask)

        # 合并裁剪块
        return torch.cat(img_set, dim=0), torch.cat(mask_set, dim=0)



# / HRF 裁剪策略
# def crop_order_images(image, mask, crop_size = 512, row = 4, col = 4,  mode = 'randoms', rands = 16):
#     image, mask = torch.tensor(image.copy()),torch.tensor(mask.copy())
#     image = torch.unsqueeze(image.permute((2,0,1)), dim=0)
#     mask = torch.unsqueeze(mask.permute((2, 0, 1)), dim=0)
#     # print(image.size(), mask.size())
#     if mode =='orders':
#         image, s_h, s_w= padding_img(image, crop_size, row, col)
#         mask, s_h, s_w = padding_img(mask, crop_size, row, col)
#         assert (crop_size > s_h and crop_size > s_w)
#         img_set, mask_set = [], []
#         for m in range(0, row):
#             for n in range(0, col):
#                 dh = m * s_h
#                 dw = n * s_w
#                 img_set.append(image[:,:,dh:dh+crop_size, dw:dw+crop_size])
#                 mask_set.append(mask[:, :, dh:dh + crop_size, dw:dw + crop_size])
#         return torch.cat([img_set[i] for i in range(0, len(img_set))], dim=0), \
#                torch.cat([mask_set[i] for i in range(0, len(img_set))], dim=0),
#     elif mode == 'randoms':
#         # random select center point to expand patches ! (compliment)
#         import  random
#         img_set, mask_set = [], []
#         for i in range(0, rands):
#             center_y = random.randint(crop_size//2, image.size()[2] - crop_size//2)
#             center_x = random.randint(crop_size//2, image.size()[3] - crop_size//2)
#             crops_img = image[:,:,center_y - crop_size//2:center_y + crop_size//2, center_x - crop_size//2:center_x + crop_size//2]
#             img_set.append(crops_img)
#             crops_mask = mask[:,:,center_y - crop_size//2:center_y + crop_size//2,center_x - crop_size//2:center_x + crop_size//2]
#             mask_set.append(crops_mask)
#         return torch.cat([img_set[i].permute((0,2,3,1)) for i in range(0, len(img_set))], dim=0)\
#             ,torch.cat([mask_set[i].permute((0,2,3,1)) for i in range(0, len(img_set))], dim=0)
#
# def padding_img(image, crop_size, rows, clos):
#     pad_h, pad_w = (image.size()[2] - crop_size)%(rows-1), (image.size()[3] - crop_size)%(clos-1)
#     image = padding_hw(image, dims='h', ns= 0 if pad_h==0 else rows -1 -pad_h)
#     image = padding_hw(image, dims='w', ns= 0 if pad_w==0 else clos -1 -pad_w)
#     return image.to(device), (image.size()[2] - crop_size)//(rows-1), (image.size()[3] - crop_size)//(clos-1)
#
#
# def padding_hw(img, dims = 'h', ns = 0):
#     if ns ==0:
#         return img
#     else:
#         after_expanding = None
#         if dims == 'h':
#             pad_img = torch.zeros_like(img[:,:,0 : ns,:,])
#             after_expanding = torch.cat([img, pad_img], dim=2)
#         elif dims == 'w':
#             pad_img = torch.zeros_like(img[:,:,:,0:ns])
#             after_expanding = torch.cat([img, pad_img], dim=3)
#         return after_expanding


def deformation_set(image, mask,
                    shift_limit=(-0.2, 0.2),
                    borderMode=cv2.BORDER_CONSTANT):
    """
    对图像和掩码进行平移和裁剪操作。

    参数：
        image: 输入图像 (H, W, C)。
        mask: 输入掩码 (H, W)。
        shift_limit: 平移范围占比 (x, y)，默认(-0.2, 0.2)。
        borderMode: cv2 边界填充模式，默认 cv2.BORDER_CONSTANT。

    返回：
        aug_img: 增强后的图像集合，形状为 (N, C, H, W)。
        aug_mask: 增强后的掩码集合，形状为 (N, 1, H, W)。
    """
    print('deformation_set size check:', image.shape, mask.shape)

    height, width, channel = image.shape
    image_set, mask_set = [], []

    # 随机生成平移偏移量
    dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
    dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

    # 平移图像和掩码
    img_transformed = cv2.warpAffine(image, np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32), (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=(0, 0, 0,))
    mask_transformed = cv2.warpAffine(mask, np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32), (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=(0, 0, 0,))

    # 随机翻转
    img_transformed, mask_transformed = randomHorizontalFlip(img_transformed, mask_transformed)
    img_transformed, mask_transformed = randomVerticleFlip(img_transformed, mask_transformed)

    # 扩展掩码维度
    mask_transformed = np.expand_dims(mask_transformed, axis=2)

    # 使用 `crop_order_images` 进行裁剪
    crop_im, crop_ma = crop_order_images(img_transformed, mask_transformed)  # HRF
    image_set.append(crop_im)
    mask_set.append(crop_ma)

    # 合并所有裁剪块
    aug_img = np.concatenate([image_set[i] for i in range(len(image_set))], axis=0)
    aug_mask = np.concatenate([mask_set[i] for i in range(len(mask_set))], axis=0)

    print('--------------- crop success ---------------')
    return aug_img, aug_mask


#
# def deformation_set(image, mask,
#                            shift_limit=(-0.2, 0.2),
#                            scale_limit=(-0.2, 0.2),
#                            rotate_limit=(-180.0, 180.0),
#                            aspect_limit=(-0.1, 0.1),
#                            borderMode=cv2.BORDER_CONSTANT, u=0.5):
#     print('deformation_set size check: ', image.shape, mask.shape)
#
#     start_angele, per_rotate = -180, 10
#     rotate_num = - start_angele // per_rotate * 2
#     image_set, mask_set = [], []
#     for rotate_id in range(0, rotate_num):
#         masks = mask
#         img = image
#         height, width, channel = img.shape
#         sx, sy = 1., 1.
#         angle = start_angele + rotate_id * per_rotate
#         dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
#         dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
#         # dx = 1.0
#         # dy = 1.0
#         cc = np.cos(angle / 180 * np.pi) * sx
#         ss = np.sin(angle / 180 * np.pi) * sy
#         rotate_matrix = np.array([[cc, -ss], [ss, cc]])
#         box0 = np.array([[0, 0], [width, 0], [width, height], [0, height],])
#         box1 = box0 - np.array([width / 2, height / 2])
#         box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
#         box0 = box0.astype(np.float32)
#         box1 = box1.astype(np.float32)
#         mat = cv2.getPerspectiveTransform(box0, box1)
#         img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
#                                     borderValue=(0, 0, 0,))
#         masks = cv2.warpPerspective(masks, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
#                                    borderValue=(0, 0, 0,))
#         img, masks = randomHorizontalFlip(img, masks)
#         img, masks = randomVerticleFlip(img, masks)
#         masks = np.expand_dims(masks, axis=2)               #
#
#         # image_set.append(np.expand_dims(img, axis=0))
#         # mask_set.append(np.expand_dims(masks, axis=0))
#
#         crop_im, crop_ma = crop_order_images(img, masks) # HRF
#         # crop_im, crop_ma = crop_images0(img, masks)# CHASE_DB1
#         image_set.append(crop_im)
#         mask_set.append(crop_ma)
#
#         # print(img.shape, masks.shape,'====================')
#     aug_img  = np.concatenate([image_set[i] for i in range(0, len(image_set))],axis=0)
#     aug_mask = np.concatenate([mask_set[i] for i in range(0, len(mask_set))], axis=0)
#     print('--------------- crop seccess ---------------')
#     return aug_img, aug_mask

def data_auguments(aug_num,size_h, size_w,path_images, path_gt,total_imgs, mask_ch, augu=True):

    all_images, all_masks = read_drive_images(size_h, size_w,path_images, path_gt,total_imgs, mask_ch)         # original data
    if augu is False:
        return all_images, all_masks
    # print('image and gt shape is:', all_images.shape, all_masks.shape)
    img_list = []
    gt_list = []
    for nums in range(0, aug_num):
        for i_d in range(0, all_images.shape[0]):
            aug_img, aug_gt = deformation_set(all_images[i_d, :, :, :, ], all_masks[i_d, :, :, :, ])
            print(aug_img.shape,'---------',aug_gt.shape)
            img_list.append(aug_img)
            gt_list.append(aug_gt)
    img_au = np.concatenate(img_list, axis=0)
    gt_au = np.concatenate(gt_list, axis=0)
    # print(img_au.shape, gt_au.shape)
    # visualize(group_images(all_masks, 5), './image_test')
    return img_au,gt_au

def data_for_train(aug_num,size_h, size_w,path_images, path_gt,total_imgs, mask_ch,augu):
    all_images, all_masks = data_auguments(aug_num, size_h, size_w, path_images, path_gt,total_imgs, mask_ch,augu)
    print('image and gt shape is:', all_images.shape, all_masks.shape)
    img = np.array(all_images, np.float32).transpose(0, 3, 1, 2)
    # mask = np.array(all_masks, np.float32).transpose(0,3,1,2) / 255.0  # DRIVE
    mask = np.array(all_masks, np.float32).transpose(0,3,1,2)   # CHASE_DB1
    # mask = np.array(all_masks, np.float32).transpose(0, 3, 1, 2) # STARE1
    # mask = np.array(all_masks, np.float32).transpose(0, 3, 1, 2)  # DCA1


    if mask_ch ==1:
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
    #  data shuffle
    index = np.arange(img.shape[0])
    np.random.shuffle(index)
    img  = img[index, :, :, :]
    mask = mask[index, :, :]
    return img, mask

def save_drive_data(mum_arg = 1):
    # images, mask = data_for_train(mum_arg, Constants.size_h,Constants.size_w,
    #                               path_images_drive, path_gt_drive, 20, mask_ch=1, augu=True) # DRIVE CHASE_DB1
    # images, mask = data_for_train(mum_arg, Constants.size_h, Constants.size_w,
    #                               path_images_drive, path_gt_drive, 100, mask_ch=1, augu=True)  # STARE1
    images_test, mask_test = data_for_train(mum_arg, Constants.size_h,Constants.size_w,
                                  path_images_test_drive, path_gt_test_drive, 10, mask_ch=1, augu=False)
    # images_val, mask_val = data_for_train(mum_arg, Constants.size_h, Constants.size_w,
    #                                         path_images_val_drive, path_gt_val_drive, 3, mask_ch=1, augu=False)

    # images_test,mask_test = data_shuffle(images_test,mask_test)

    try:
        # read_numpy_into_npy(images, Constants.path_image_drive)
        # read_numpy_into_npy(mask, Constants.path_label_drive)
        read_numpy_into_npy(images_test, Constants.path_test_image_drive)
        read_numpy_into_npy(mask_test, Constants.path_test_label_drive)
        # read_numpy_into_npy(images_val, Constants.path_val_image_drive)
        # read_numpy_into_npy(mask_val, Constants.path_val_label_drive)
        print('========  all drive train and test data has been saved ! ==========')
    except:
        print(' file save exception has happened! ')

    pass


# def save_drive_data(mum_arg=1):
#     """
#     将数据分为前 300 张和后 300 张进行增强并保存为两个 .npy 文件，然后将这两个文件拼接成一个。
#
#     参数：
#         mum_arg: 数据集的选择，默认为 1。
#
#     返回：
#         无
#     """
#     # 设置前 300 张和后 300 张的样本数量
#     total_samples = 600
#     first_batch_size = 200
#     second_batch_size = total_samples - first_batch_size
#
#     # 第一批数据增强（前 300 张）
#     images_batch_1, mask_batch_1 = data_for_train(mum_arg, Constants.size_h, Constants.size_w,
#                                                   path_images_drive, path_gt_drive, first_batch_size, mask_ch=1, augu=True)
#
#     # 第二批数据增强（后 300 张）
#     # images_batch_2, mask_batch_2 = data_for_train(mum_arg, Constants.size_h, Constants.size_w,
#     #                                               path_images_drive, path_gt_drive, second_batch_size, mask_ch=1, augu=True, start_idx=first_batch_size)
#
#     # 保存第一批和第二批数据到各自的 .npy 文件
#     try:
#         read_numpy_into_npy(images_batch_1, Constants.path_image_drive)
#         read_numpy_into_npy(mask_batch_1, Constants.path_label_drive)
#         # read_numpy_into_npy(images_batch_2, Constants.path_image_drive_batch_2)
#         # read_numpy_into_npy(mask_batch_2, Constants.path_label_drive_batch_2)
#
#         print('========  first batch and second batch data have been saved! ========== ')
#
#     except Exception as e:
#         print(f'Error occurred during saving the batch data: {e}')
#
#     pass
#
#     # 拼接两个 .npy 文件并保存最终文件
#     # try:
#     #     # 加载前 300 张和后 300 张的数据
#     #     images_batch_1 = np.load(Constants.path_image_drive_batch_1)
#     #     mask_batch_1 = np.load(Constants.path_label_drive_batch_1)
#     #     images_batch_2 = np.load(Constants.path_image_drive_batch_2)
#     #     mask_batch_2 = np.load(Constants.path_label_drive_batch_2)
#     #
#     #     # 拼接数据
#     #     final_images = np.concatenate([images_batch_1, images_batch_2], axis=0)
#     #     final_masks = np.concatenate([mask_batch_1, mask_batch_2], axis=0)
#     #
#     #     # 保存拼接后的数据
#     #     read_numpy_into_npy(final_images, Constants.path_image_drive)
#     #     read_numpy_into_npy(final_masks, Constants.path_label_drive)
#     #
#     #     print('========  all data has been successfully concatenated and saved into a single file! ========== ')
#     #
#     # except Exception as e:
#     #     print(f'Error occurred during concatenation and final saving: {e}')




def check_bst_data():
    # a=load_from_npy(Constants.path_image_drive)
    # b=load_from_npy(Constants.path_label_drive)
    c=load_from_npy(Constants.path_test_image_drive)
    d=load_from_npy(Constants.path_test_label_drive)
    # e = load_from_npy(Constants.path_val_image_drive)
    # f = load_from_npy(Constants.path_val_label_drive)
    # print(a.shape, b.shape, c.shape, d.shape)
    # print(np.max(a),np.max(b),np.max(c), np.max(d))
    print(c.shape, d.shape)
    print(np.max(c),np.max(d))


if __name__ == '__main__':
    save_drive_data()
    check_bst_data()
    pass